"""Summarise the mutation-strength screen made by `run_experiments.py --pilot`.

Reads the raw screen (experiment_config.PILOT_ROOT, Git-ignored and exactly
reproducible) and writes to experiment_config.PILOT_SUMMARY_ROOT (tracked):

    pilot_curves.pdf/.png   best-so-far per sigma, every seed and their mean; one
                            panel per sigma, random search in grey in each panel
    pilot_final.pdf/.png    best distance against sigma at two budgets
    pilot_budgets.csv       best distance per condition at several budgets
    pilot_plateau.csv       where ea.py's plateau rule would stop each run
    pilot_survival.csv      share of children kept by (mu+lambda) selection
    pilot_summary.txt       validation, failures and the main numbers

Loading and validation reuse analysis.py, so the screen is checked exactly as
the final experiment will be. Lower fitness is better (metres to the target).
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window
import matplotlib.pyplot as plt
import numpy as np

import experiment_config as cfg
from analysis import Run, aggregate, common_grid, is_baseline, load_runs, validate

BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"  # ordinal pair with BLUE_DARK, validated for print
BLUE_DARK = "#1c5cab"
GREY = "#6b6b6b"
INK = "#2b2b2b"
GRID = "#e4e4e1"

# Candidate stopping rules, applied after the fact exactly as ea.py applies them.
PATIENCES = (25, 50, 100, 200)
DELTAS = (0.001, 0.005, 0.01)
# Generation ranges for the offspring survival rate.
PHASES = ((1, 50), (51, 200), (201, 500), (501, 1000))


def sigma_of(run: Run) -> float | None:
    return json.loads((run.folder / "summary.json").read_text(encoding="utf-8"))["mutation_strength"]


def history_best(run: Run) -> np.ndarray:
    """Best fitness per generation (row g = generation g)."""
    with (run.folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        return np.array([float(row["best_fitness"]) for row in csv.DictReader(stream)])


def plateau_stop(best: np.ndarray, patience: int, delta: float) -> int | None:
    """Generation at which ea.py's plateau rule stops, or None if it never does."""
    reference, without = best[0], 0
    for generation in range(1, len(best)):
        if reference - best[generation] > delta:
            reference, without = best[generation], 0
        else:
            without += 1
        if without >= patience:
            return generation
    return None


def survival_by_phase(database: Path) -> dict[str, float]:
    """Share of each generation's children that (mu+lambda) selection kept."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT time_of_birth, alive OR time_of_death > time_of_birth "
            "FROM individual WHERE time_of_birth >= 1"
        ).fetchall()
    born = np.array([r[0] for r in rows])
    kept = np.array([r[1] for r in rows], dtype=float)
    return {
        f"gen_{low}_{high}": float(kept[(born >= low) & (born <= high)].mean())
        for low, high in PHASES if np.any((born >= low) & (born <= high))
    }


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8,
        "axes.edgecolor": GREY, "axes.labelcolor": INK,
        "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
    })


def tidy(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_curves(runs: list[Run], curves: dict, ea_methods: list[str], sigmas: dict,
                grid: np.ndarray, init_std: float, stem: Path) -> None:
    """One panel per sigma: every seed (thin) and their mean (bold).

    Individual seeds rather than a std band: with three seeds and bimodal
    outcomes, mean +/- std would suggest impossible negative distances.
    """
    rows = int(np.ceil(len(ea_methods) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(7.0, 1.75 * rows + 0.5), sharex=True, sharey=True,
                             squeeze=False)
    rs = curves[cfg.RANDOM_SEARCH]["mean"]
    for ax, method in zip(axes.flat, ea_methods, strict=False):  # spare panels hidden below
        ax.plot(grid, rs, color=GREY, linewidth=1.2, linestyle="--", label="Random search, mean")
        for index, run in enumerate(sorted((r for r in runs if r.method == method), key=lambda r: r.seed)):
            ax.plot(grid, [run.best_at(e) for e in grid], color=BLUE_LIGHT, linewidth=0.8,
                    label="EA, each seed" if index == 0 else None)
        ax.plot(grid, curves[method]["mean"], color=BLUE_DARK, linewidth=1.6,
                label=f"EA, mean of {curves[method]['n']} seeds")
        sigma = sigmas[method]
        ax.set_title(f"\u03c3 = {sigma:g}  ({sigma / init_std:g}\u00d7 initial std)", loc="left")
        ax.set_ylim(bottom=0)
        tidy(ax)
    for ax in axes.flat[len(ea_methods):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Best distance (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Evaluations")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    save(fig, stem)


def plot_final(values: dict, ea_methods: list[str], sigmas: dict, budgets: tuple[int, int],
               stem: Path) -> None:
    """Every seed's best distance (dots) and the mean (bar) against sigma, at two budgets."""
    fig, ax = plt.subplots(figsize=(3.33, 2.8))
    early, full = budgets
    xs = np.array([sigmas[m] for m in ea_methods])
    for budget, colour, shift in ((early, BLUE_LIGHT, 0.93), (full, BLUE_DARK, 1.07)):
        per_seed = np.array([values[m][budget] for m in ea_methods])  # methods x seeds
        for x, seeds in zip(xs * shift, per_seed, strict=True):
            ax.scatter(np.full(len(seeds), x), seeds, s=14, color=colour, linewidths=0, zorder=3)
        ax.scatter(xs * shift, per_seed.mean(axis=1), marker="_", s=110, linewidths=1.6,
                   color=colour, zorder=4, label=f"EA, {budget:,} evals")
        rs = float(np.mean(values[cfg.RANDOM_SEARCH][budget]))
        ax.axhline(rs, color=GREY, linewidth=1, linestyle=":" if budget == early else "--", zorder=1,
                   label=f"Random search, {budget:,} evals")
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{s:g}" for s in xs])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mutation strength \u03c3")
    ax.set_ylabel("Best distance (m)")
    tidy(ax)
    # Seeds as dots, means as bars; random search as its mean (a line).
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(f"EA, {early:,} evals"), labels.index(f"EA, {full:,} evals"),
             labels.index(f"Random search, {early:,} evals"), labels.index(f"Random search, {full:,} evals")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.54, 1.0), columnspacing=1.0)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.87))
    save(fig, stem)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="?", default=cfg.PILOT_ROOT)
    parser.add_argument("--out", type=Path, default=cfg.PILOT_SUMMARY_ROOT)
    parser.add_argument("--force", action="store_true", help="Summarise even if validation fails.")
    args = parser.parse_args()

    root = args.results.resolve()
    runs = load_runs(root)
    seeds = sorted({r.seed for r in runs})
    errors, warnings, budget = validate(runs, len(seeds), "random")
    shown = root.relative_to(cfg.REPO_ROOT) if root.is_relative_to(cfg.REPO_ROOT) else root
    report = [f"Screen: {shown}", f"Runs: {len(runs)}, seeds {seeds}, common budget {budget}", ""]
    report += [f"ERROR    {e}" for e in errors] or ["Validation: no errors."]
    report += [f"WARNING  {w}" for w in warnings]
    if errors and not args.force:
        sys.exit("\n".join(report) + "\nValidation failed; use --force only for incomplete data.")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sigmas = {r.method: sigma_of(r) for r in runs}
    ea_methods = sorted({r.method for r in runs if not is_baseline(r.method, "random")},
                        key=lambda m: sigmas[m])
    methods = ea_methods + [cfg.RANDOM_SEARCH]
    init_std = cfg.EVALUATION.controller.initial_weight_std
    # Generation sizes as recorded with the data, not as currently configured.
    ea = json.loads((root / "experiment.json").read_text(encoding="utf-8"))["resolved"]["ea"]
    size, children = ea["population_size"], ea["number_of_children"]
    checkpoints = [b for b in (size + 10 * children, size + 50 * children, size + 100 * children,
                               size + 200 * children, size + 500 * children, budget) if b <= budget]
    checkpoints = sorted(set(checkpoints))

    # --- best distance at several budgets -------------------------------- #
    values = {m: {b: sorted_by_seed(runs, m, b) for b in checkpoints} for m in methods}
    budget_rows = []
    for m in methods:
        for b in checkpoints:
            v = np.array(values[m][b])
            rs = np.array(values[cfg.RANDOM_SEARCH][b])
            budget_rows.append({
                "method": m, "mutation_strength": "" if sigmas[m] is None else sigmas[m],
                "evaluations": b, "n": len(v),
                "mean": f"{v.mean():.4f}", "std": f"{v.std(ddof=1):.4f}",
                "min": f"{v.min():.4f}", "max": f"{v.max():.4f}",
                "seeds_better_than_random_search": "" if m == cfg.RANDOM_SEARCH
                else f"{int(np.sum(v < rs))}/{len(v)}",
            })
    write_csv(out / "pilot_budgets.csv", budget_rows)

    # --- plateau rule ------------------------------------------------------ #
    plateau_rows = []
    for run in sorted((r for r in runs if r.method in ea_methods), key=lambda r: (sigmas[r.method], r.seed)):
        best = history_best(run)
        for patience in PATIENCES:
            for delta in DELTAS:
                stop = plateau_stop(best, patience, delta)
                at_stop = best[stop] if stop is not None else best[-1]
                plateau_rows.append({
                    "method": run.method, "seed": run.seed, "patience": patience, "delta": delta,
                    "stop_generation": "" if stop is None else stop,
                    "stop_evaluations": "" if stop is None else size + stop * children,
                    "best_at_stop": f"{at_stop:.4f}", "best_at_end": f"{best[-1]:.4f}",
                    "lost_by_stopping": f"{at_stop - best[-1]:.4f}",
                })
    write_csv(out / "pilot_plateau.csv", plateau_rows)

    # --- offspring survival ------------------------------------------------ #
    survival_rows = []
    for run in sorted((r for r in runs if r.method in ea_methods), key=lambda r: (sigmas[r.method], r.seed)):
        survival_rows.append({"method": run.method, "seed": run.seed,
                              **{k: f"{v:.3f}" for k, v in survival_by_phase(run.folder / "population.db").items()}})
    write_csv(out / "pilot_survival.csv", survival_rows)

    # --- figures ------------------------------------------------------------ #
    style()
    grid = common_grid(runs, budget)
    curves = aggregate(runs, methods, grid)
    plot_curves(runs, curves, ea_methods, sigmas, grid, init_std, out / "pilot_curves")
    early = size + 100 * children if size + 100 * children < budget else checkpoints[0]
    plot_final(values, ea_methods, sigmas, (early, budget), out / "pilot_final")

    # --- text summary -------------------------------------------------------- #
    failures = {f"{r.method} seed {r.seed}": json.loads((r.folder / "summary.json").read_text())["failed_evaluations"]
                for r in runs}
    report += ["", f"Failed evaluations: {sum(failures.values())} in total"
               + ("" if not any(failures.values()) else f" ({ {k: v for k, v in failures.items() if v} })")]
    report += ["", f"Best distance (m), mean +/- std over {len(seeds)} seeds:"]
    for m in methods:
        cells = [f"{b:>6}: {np.mean(values[m][b]):.3f}+/-{np.std(values[m][b], ddof=1):.3f}" for b in checkpoints]
        report.append(f"  {m:<16} " + "  ".join(cells))
    (out / "pilot_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote {out}")


def sorted_by_seed(runs: list[Run], method: str, budget: int) -> list[float]:
    return [r.best_at(budget) for r in sorted((r for r in runs if r.method == method), key=lambda r: r.seed)]


if __name__ == "__main__":
    main()
