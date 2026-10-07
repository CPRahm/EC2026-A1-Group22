"""
What it does
------------
1. Validates: all seeds present, no failed runs, budgets reached, settings
   identical across runs, logged best == saved candidate.
2. Fixes one comparison budget: the largest evaluation count that EVERY
   evolutionary run reached. Baseline runs must reach it too. Curves and final
   values are cut at that budget; stopped runs are never padded.
3. Writes to RESULTS_DIR/analysis/:
       convergence.pdf / .png   mean +/- std of best-so-far across seeds
       curves.csv               the numbers behind the plot
       final_fitness.csv        per-method summary at the comparison budget
       per_run.csv              one row per run
       comparisons.csv          pairwise tests and effect sizes
       validation.txt           everything checked, and what failed

Lower fitness is better (planar distance to the target, in metres).
"""

import argparse
import csv
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical colours (colour-blind safe as a set of three), assigned
# in this fixed order. Line styles double-encode identity for greyscale print.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1))]
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e1"

MEAN_COLUMNS = ("mean_fitness", "survivor_mean_fitness")
EXACT_TEST_LIMIT = 2_000_000  # enumeration cap for exact tests


# ============================================================================ #
#  Loading
# ============================================================================ #


@dataclass
class Run:
    method: str
    seed: int
    folder: Path
    evaluations: np.ndarray
    best_raw: np.ndarray  # best_fitness exactly as logged
    best_so_far: np.ndarray  # running minimum of best_raw
    status: str | None
    settings: dict | None
    saved_fitness: float | None
    failed: bool
    notes: list[str] = field(default_factory=list)

    @property
    def final_evaluations(self) -> int:
        return int(self.evaluations[-1])

    def best_at(self, budget: int) -> float:
        """Best-so-far after `budget` evaluations; NaN before the first checkpoint."""
        index = np.searchsorted(self.evaluations, budget, side="right") - 1
        return float(self.best_so_far[index]) if index >= 0 else math.nan


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_from_path(folder: Path, root: Path) -> tuple[str | None, int | None]:
    """Fallback: find 'seed_<n>' in the path; the other part is the method."""
    parts = folder.relative_to(root).parts
    seed, method = None, None
    for part in parts:
        match = re.fullmatch(r"seed[_-]?(\d+)", part)
        if match:
            seed = int(match.group(1))
        else:
            method = part
    return method, seed


def load_run(history_path: Path, root: Path) -> Run:
    folder = history_path.parent
    summary = _read_json(folder / "summary.json") or {}
    candidate = _read_json(folder / "best_candidate.json") or {}

    method = summary.get("method") or candidate.get("method")
    seed = summary.get("seed", candidate.get("seed"))
    if method is None or seed is None:
        path_method, path_seed = _identity_from_path(folder, root)
        method = method or path_method
        seed = seed if seed is not None else path_seed
    if method is None or seed is None:
        raise ValueError(f"Cannot tell method/seed for {folder}; add a summary.json.")

    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{history_path} is empty.")
    for column in ("evaluations", "best_fitness"):
        if column not in rows[0]:
            raise ValueError(f"{history_path} has no '{column}' column.")

    evaluations = np.array([int(float(r["evaluations"])) for r in rows])
    best_raw = np.array([float(r["best_fitness"]) for r in rows])

    return Run(
        method=str(method),
        seed=int(seed),
        folder=folder,
        evaluations=evaluations,
        best_raw=best_raw,
        best_so_far=np.minimum.accumulate(best_raw),
        status=summary.get("status"),
        settings=candidate.get("settings"),
        saved_fitness=candidate.get("fitness"),
        failed=(folder / "failure.json").exists(),
    )


def load_runs(root: Path) -> list[Run]:
    paths = sorted(root.rglob("history.csv"))
    return [load_run(path, root) for path in paths]


# ============================================================================ #
#  Validation
# ============================================================================ #


def is_baseline(method: str, baseline_token: str) -> bool:
    return baseline_token.lower() in method.lower()


def validate(
    runs: list[Run], expected_seeds: int, baseline_token: str
) -> tuple[list[str], list[str], int]:
    """Return (errors, warnings, comparison_budget)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not runs:
        return ["No history.csv files found."], [], 0

    by_method: dict[str, list[Run]] = {}
    for run in runs:
        by_method.setdefault(run.method, []).append(run)

    # --- completeness ---------------------------------------------------- #
    for method, group in sorted(by_method.items()):
        seeds = [r.seed for r in group]
        duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
        if duplicates:
            errors.append(f"{method}: duplicate seeds {duplicates}.")
        if len(set(seeds)) != expected_seeds:
            errors.append(
                f"{method}: {len(set(seeds))} seeds found, {expected_seeds} expected "
                f"(have {sorted(set(seeds))})."
            )

    seed_sets = {m: frozenset(r.seed for r in g) for m, g in by_method.items()}
    if len(set(seed_sets.values())) > 1:
        errors.append(
            "Methods were run on different seeds: "
            + "; ".join(f"{m}={sorted(s)}" for m, s in sorted(seed_sets.items()))
        )

    if not any(is_baseline(m, baseline_token) for m in by_method):
        errors.append(f"No baseline method found (no method name contains '{baseline_token}').")
    if all(is_baseline(m, baseline_token) for m in by_method):
        errors.append("No evolutionary method found.")

    # --- per-run integrity ------------------------------------------------ #
    for run in runs:
        label = f"{run.method} seed {run.seed}"
        if run.failed:
            errors.append(f"{label}: failure.json present - this run crashed.")
        if run.status is not None and run.status != "completed":
            errors.append(f"{label}: status is '{run.status}', not 'completed'.")
        if not np.all(np.isfinite(run.best_raw)):
            errors.append(f"{label}: non-finite fitness in history.")
        if np.any(np.diff(run.evaluations) <= 0):
            errors.append(f"{label}: evaluation counts do not strictly increase.")
        if not is_baseline(run.method, baseline_token) and np.any(np.diff(run.best_raw) > 1e-12):
            errors.append(
                f"{label}: best fitness got worse between generations - elitism is broken."
            )
        if run.saved_fitness is None:
            warnings.append(f"{label}: no best_candidate.json - this run cannot be replayed.")
        elif not math.isclose(run.saved_fitness, float(run.best_so_far[-1]), abs_tol=1e-10):
            errors.append(
                f"{label}: saved candidate fitness {run.saved_fitness:.6f} != "
                f"best in history {run.best_so_far[-1]:.6f}."
            )

    # --- identical settings ----------------------------------------------- #
    with_settings = [r for r in runs if r.settings is not None]
    if with_settings:
        reference = with_settings[0]
        for run in with_settings[1:]:
            if run.settings != reference.settings:
                errors.append(
                    f"{run.method} seed {run.seed}: evaluation settings differ from "
                    f"{reference.method} seed {reference.seed}."
                )

    # --- comparison budget ------------------------------------------------- #
    evolutionary = [r for r in runs if not is_baseline(r.method, baseline_token)]
    budget = min((r.final_evaluations for r in evolutionary), default=0)
    stopped = sorted({r.final_evaluations for r in evolutionary})
    if len(stopped) > 1:
        warnings.append(
            f"Evolutionary runs stopped at different budgets {stopped}; "
            f"all methods are compared at {budget} evaluations."
        )
    for run in runs:
        if is_baseline(run.method, baseline_token):
            if run.final_evaluations < budget:
                errors.append(
                    f"{run.method} seed {run.seed}: baseline ran {run.final_evaluations} "
                    f"evaluations, fewer than the comparison budget {budget}."
                )
            elif run.final_evaluations > budget:
                warnings.append(
                    f"{run.method} seed {run.seed}: baseline ran {run.final_evaluations} "
                    f"evaluations; only the first {budget} are used."
                )

    return errors, warnings, budget


# ============================================================================ #
#  Aggregation
# ============================================================================ #


def method_order(methods: list[str], baseline_token: str) -> list[str]:
    """Evolutionary methods first (alphabetical), baselines last. Fixed colours."""
    return sorted(methods, key=lambda m: (is_baseline(m, baseline_token), m))


def common_grid(runs: list[Run], budget: int) -> np.ndarray:
    """Checkpoints up to the budget at which every run already has data."""
    start = max(int(r.evaluations[0]) for r in runs)
    points = {int(e) for r in runs for e in r.evaluations if start <= e <= budget}
    points.add(budget)
    return np.array(sorted(points))


def aggregate(runs: list[Run], methods: list[str], grid: np.ndarray) -> dict[str, dict]:
    curves = {}
    for method in methods:
        group = sorted((r for r in runs if r.method == method), key=lambda r: r.seed)
        matrix = np.array([[r.best_at(e) for e in grid] for r in group])
        curves[method] = {
            "n": len(group),
            "mean": matrix.mean(axis=0),
            # Sample std across independent runs (ddof=1); 0 for a single run.
            "std": matrix.std(axis=0, ddof=1) if len(group) > 1 else np.zeros(len(grid)),
        }
    return curves


# ============================================================================ #
#  Statistics (exact, no SciPy needed)
# ============================================================================ #


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get the average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def mann_whitney_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact Mann-Whitney U p-value (independent samples)."""
    pooled = np.concatenate([a, b])
    ranks = _ranks(pooled)
    n_a, n_total = len(a), len(pooled)
    if math.comb(n_total, n_a) > EXACT_TEST_LIMIT:
        return math.nan
    centre = n_a * (n_total + 1) / 2
    observed = abs(ranks[:n_a].sum() - centre)
    extreme = total = 0
    for chosen in itertools.combinations(range(n_total), n_a):
        total += 1
        if abs(ranks[list(chosen)].sum() - centre) >= observed - 1e-9:
            extreme += 1
    return extreme / total


def wilcoxon_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value (paired by seed)."""
    differences = a - b
    differences = differences[differences != 0]
    n = len(differences)
    if n == 0:
        return 1.0
    if 2**n > EXACT_TEST_LIMIT:
        return math.nan
    ranks = _ranks(np.abs(differences))
    centre = ranks.sum() / 2
    observed = abs(ranks[differences > 0].sum() - centre)
    extreme = 0
    for signs in itertools.product((False, True), repeat=n):
        if abs(ranks[list(signs)].sum() - centre) >= observed - 1e-9:
            extreme += 1
    return extreme / 2**n


def a12(a: np.ndarray, b: np.ndarray) -> float:
    """Vargha-Delaney A12: chance a run of `a` ends LOWER (better) than a run of `b`."""
    wins = sum((x < y) + 0.5 * (x == y) for x in a for y in b)
    return wins / (len(a) * len(b))


def a12_magnitude(value: float) -> str:
    distance = abs(value - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


# ============================================================================ #
#  Output
# ============================================================================ #


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_name(method: str) -> str:
    return method.replace("_", " ")


def plot(curves: dict, methods: list[str], grid: np.ndarray, path_stem: Path, title: str) -> None:
    # Sized for one GECCO column (3.33 in), readable at print size.
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    })
    fig, ax = plt.subplots(figsize=(3.33, 2.4))

    for index, method in enumerate(methods):
        colour = COLOURS[index % len(COLOURS)]
        mean, std = curves[method]["mean"], curves[method]["std"]
        ax.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.14, linewidth=0)
        ax.plot(
            grid, mean, color=colour, linewidth=1.5, linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=f"{display_name(method)} (n={curves[method]['n']})",
        )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best distance to target (m)")
    ax.set_xlim(grid[0], grid[-1])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Minimisation curves all start high, so the lower-left corner is the
    # spot least likely to cover data.
    ax.legend(frameon=False, loc="lower left", handlelength=2.6)
    if title:
        ax.set_title(title, fontsize=8, color=MUTED, loc="left")
    fig.tight_layout(pad=0.3)
    fig.savefig(path_stem.with_suffix(".pdf"))
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder containing the run folders.")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--baseline", default="random",
                        help="Methods whose name contains this text are baselines (default: random).")
    parser.add_argument("--force", action="store_true",
                        help="Produce outputs even if validation fails (marked INCOMPLETE).")
    args = parser.parse_args()

    root = args.results.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    runs = load_runs(root)
    errors, warnings, budget = validate(runs, args.expected_seeds, args.baseline)

    out = root / "analysis"
    out.mkdir(exist_ok=True)
    report = [f"Results folder: {root}", f"Runs found: {len(runs)}",
              f"Comparison budget: {budget} evaluations", ""]
    report += [f"ERROR    {e}" for e in errors] or ["No errors."]
    report += [f"WARNING  {w}" for w in warnings]
    (out / "validation.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))

    if errors and not args.force:
        sys.exit("\nValidation failed: fix the errors above, or use --force for pilot data.")
    if not runs or budget == 0:
        sys.exit("Nothing to analyse.")

    methods = method_order(sorted({r.method for r in runs}), args.baseline)
    grid = common_grid(runs, budget)
    curves = aggregate(runs, methods, grid)

    # --- curves.csv ------------------------------------------------------- #
    write_csv(out / "curves.csv", [
        {"evaluations": int(e), **{f"{m}_{k}": f"{curves[m][k][i]:.6f}"
                                   for m in methods for k in ("mean", "std")}}
        for i, e in enumerate(grid)
    ])

    # --- per_run.csv ------------------------------------------------------ #
    finals: dict[str, dict[int, float]] = {m: {} for m in methods}
    per_run = []
    for run in sorted(runs, key=lambda r: (methods.index(r.method), r.seed)):
        value = run.best_at(budget)
        finals[run.method][run.seed] = value
        per_run.append({
            "method": run.method, "seed": run.seed,
            "evaluations_run": run.final_evaluations,
            "best_at_budget": f"{value:.6f}",
            "best_at_end": f"{run.best_so_far[-1]:.6f}",
            "stopped_before_others": run.final_evaluations > budget,
        })
    write_csv(out / "per_run.csv", per_run)

    # --- final_fitness.csv ------------------------------------------------ #
    summary = []
    for method in methods:
        values = np.array(list(finals[method].values()))
        summary.append({
            "method": method, "n": len(values),
            "mean": f"{values.mean():.4f}",
            "std": f"{values.std(ddof=1) if len(values) > 1 else 0.0:.4f}",
            "median": f"{np.median(values):.4f}",
            "best": f"{values.min():.4f}", "worst": f"{values.max():.4f}",
        })
    write_csv(out / "final_fitness.csv", summary)

    # --- comparisons.csv -------------------------------------------------- #
    comparisons = []
    for first, second in itertools.combinations(methods, 2):
        shared = sorted(set(finals[first]) & set(finals[second]))
        a = np.array([finals[first][s] for s in shared])
        b = np.array([finals[second][s] for s in shared])
        effect = a12(a, b)
        comparisons.append({
            "method_a": first, "method_b": second, "n_per_method": len(shared),
            "mean_difference_a_minus_b": f"{a.mean() - b.mean():.4f}",
            "a_better_in_seeds": f"{int(np.sum(a < b))}/{len(shared)}",
            "A12_a_beats_b": f"{effect:.3f}",
            "A12_magnitude": a12_magnitude(effect),
            "mann_whitney_p": f"{mann_whitney_exact(a, b):.4f}",
            "wilcoxon_paired_p": f"{wilcoxon_exact(a, b):.4f}",
        })
    if comparisons:
        write_csv(out / "comparisons.csv", comparisons)

    title = "INCOMPLETE - validation failed" if errors else ""
    plot(curves, methods, grid, out / "convergence", title)

    # --- console summary -------------------------------------------------- #
    print(f"\nFinal best distance at {budget} evaluations (lower is better):")
    for row in summary:
        print(f"  {row['method']:<20} n={row['n']}  mean={row['mean']}  std={row['std']}  "
              f"median={row['median']}  best={row['best']}")
    if comparisons:
        n = comparisons[0]["n_per_method"]
        print(f"\nPairwise comparisons (n={n} per method). Smallest p these tests can reach "
              f"at this n: Mann-Whitney {2 / math.comb(2 * n, n):.4f}, "
              f"Wilcoxon {2 / 2 ** n:.4f}.")
        for row in comparisons:
            print(f"  {row['method_a']} vs {row['method_b']}: diff={row['mean_difference_a_minus_b']}, "
                  f"A wins {row['a_better_in_seeds']}, A12={row['A12_a_beats_b']} "
                  f"({row['A12_magnitude']}), MW p={row['mann_whitney_p']}, "
                  f"Wilcoxon p={row['wilcoxon_paired_p']}")
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()