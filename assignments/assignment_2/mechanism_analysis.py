"""Selection pressure, genetic diversity and evolved controllers in one results folder.

Post-processing only: nothing is simulated. From the repository root:

    python assignments/assignment_2/mechanism_analysis.py <results folder>

Writes to <results folder>/analysis/ and prints the three tables:

    survival.csv     per EA run, the share of children kept by (mu+lambda)
                     selection among those born in generations 1-50 and in
                     the run's last 100 generations (G-99..G, where G is the
                     last row of its history.csv); then one row per method
                     with the mean of its runs
    controllers.csv  per best_candidate.json: RMS of the weights, final x and
                     y, and the planar distance walked from the initial to the
                     final position
    diversity.csv    per EA run, the mean over all pairs of survivors of the
                     RMS genotype difference per parameter, after generations
                     0, 10, 50 and G; then one row per method with the mean of
                     its runs

A child counts as kept if it is still alive or outlived its birth generation
(alive = 1 OR time_of_death > time_of_birth), as in
pilot_analysis.survival_by_phase.

The survivors after generation g are the individuals with time_of_birth <= g
< time_of_death, and after G those with alive = 1. Every generation 0..G is
checked against history.csv (best and mean fitness) and against the population
size in experiment.json; a mismatch stops the script, non-zero, before any
file is written. EA runs without a population.db are skipped with a warning.
Standalone: needs only sqlite3, json, csv and numpy.
"""

import argparse
import csv
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

EARLY = (1, 50)  # first window of birth generations, inclusive
LATE = 100  # second window: the run's last LATE generations
DIVERSITY_POINTS = (0, 10, 50)  # generations reported, besides each run's last
BEST_TOLERANCE = 1e-12  # rebuilt survivors against history.csv best_fitness
MEAN_TOLERANCE = 1e-9  # ... and mean_fitness (summed in another order)

SURVIVAL_FIELDS = ("method", "seed", "last_generation", "gen_1_50", "gen_last_100")
CONTROLLER_FIELDS = ("method", "seed", "weight_rms", "final_x", "final_y", "distance_walked")
DIVERSITY_FIELDS = ("method", "seed", "last_generation", *(f"gen_{g}" for g in DIVERSITY_POINTS), "gen_last")


def load_candidates(root: Path) -> list[tuple[Path, dict]]:
    """(run folder, best_candidate.json contents): EA runs by sigma, then baselines."""
    found = [(path.parent, json.loads(path.read_text(encoding="utf-8")))
             for path in root.rglob("best_candidate.json")]
    return sorted(found, key=lambda item: (item[1].get("algorithm") != "ea",
                                           item[1].get("mutation_strength") or 0.0,
                                           item[1]["method"], item[1]["seed"]))


def last_generation(folder: Path) -> int:
    """G: the generation in the last row of the run's history.csv."""
    with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        return int(list(csv.DictReader(stream))[-1]["generation"])


def kept_shares(database: Path, windows: list[tuple[int, int]]) -> list[float]:
    """Share of the children born in each inclusive window that selection kept."""
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT time_of_birth, alive OR time_of_death > time_of_birth "
            "FROM individual WHERE time_of_birth >= 1"
        ).fetchall()
    born = np.array([r[0] for r in rows])
    kept = np.array([r[1] for r in rows], dtype=float)
    shares = []
    for low, high in windows:
        inside = (born >= low) & (born <= high)
        shares.append(float(kept[inside].mean()) if inside.any() else float("nan"))
    return shares


def survival_row(method: str, seed: int | str, last: int | str, early: float, late: float) -> dict:
    return {"method": method, "seed": seed, "last_generation": last,
            "gen_1_50": f"{early:.3f}", "gen_last_100": f"{late:.3f}"}


def survival_table(candidates: list[tuple[Path, dict]]) -> list[dict]:
    """Each EA method's runs, followed by their mean."""
    by_method: dict[str, list[tuple[int, int, float, float]]] = {}
    for folder, candidate in candidates:
        if candidate.get("algorithm") != "ea":
            continue
        method, seed = candidate["method"], candidate["seed"]
        database = folder / "population.db"
        if not database.is_file():
            print(f"WARNING  {method} seed {seed}: no population.db; left out of survival.csv.")
            continue
        last = last_generation(folder)
        early, late = kept_shares(database, [EARLY, (max(1, last - LATE + 1), last)])
        by_method.setdefault(method, []).append((seed, last, early, late))

    rows = []
    for method, runs in by_method.items():
        rows += [survival_row(method, seed, last, early, late) for seed, last, early, late in runs]
        early, late = np.mean([(early, late) for _, _, early, late in runs], axis=0)
        rows.append(survival_row(method, "mean", "", early, late))
    return rows


def controller_table(candidates: list[tuple[Path, dict]]) -> list[dict]:
    rows = []
    for _, candidate in candidates:
        weights = np.asarray(candidate["weights"], dtype=float)
        start = np.asarray(candidate["initial_position"], dtype=float)
        end = np.asarray(candidate["final_position"], dtype=float)
        rows.append({
            "method": candidate["method"], "seed": candidate["seed"],
            "weight_rms": f"{np.sqrt(np.mean(weights ** 2)):.4f}",
            "final_x": f"{end[0]:.4f}", "final_y": f"{end[1]:.4f}",
            "distance_walked": f"{np.hypot(*(end[:2] - start[:2])):.4f}",
        })
    return rows


def experiment_settings(root: Path) -> dict | None:
    """The folder's recorded settings (experiment.json "resolved"), or None if absent."""
    path = root / "experiment.json"
    return json.loads(path.read_text(encoding="utf-8"))["resolved"] if path.is_file() else None


def load_individuals(database: Path) -> tuple[np.ndarray, ...]:
    """alive, time_of_birth, time_of_death, fitness and genotype of every individual."""
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT alive, time_of_birth, time_of_death, fitness_, genotype_ FROM individual"
        ).fetchall()
    alive = np.array([r[0] for r in rows], dtype=bool)
    born = np.array([r[1] for r in rows])
    died = np.array([r[2] for r in rows])
    fitness = np.array([r[3] for r in rows], dtype=float)
    genotypes = np.array([json.loads(r[4]) for r in rows], dtype=float)
    return alive, born, died, fitness, genotypes


def mean_pairwise_rms(genotypes: np.ndarray) -> float:
    """Mean over all pairs of rows of the RMS difference per parameter."""
    differences = genotypes[:, None, :] - genotypes[None, :, :]
    rms = np.sqrt(np.mean(differences ** 2, axis=2))
    return float(rms[np.triu_indices(len(genotypes), k=1)].mean())


def survivor_diversity(folder: Path, population_size: int,
                       label: str) -> tuple[dict[int, float], float, float]:
    """Diversity of the survivors after every generation 0..G, checked against history.csv.

    Returns {generation: diversity} and the largest gaps to history.csv's best
    and mean fitness. Exits non-zero at the first generation that does not match.
    """
    with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    alive, born, died, fitness, genotypes = load_individuals(folder / "population.db")
    last = int(history[-1]["generation"])
    diversity, best_gap, mean_gap = {}, 0.0, 0.0
    for generation, row in enumerate(history):
        where = f"Survivor check failed: {label}, generation {generation}"
        if int(row["generation"]) != generation:
            sys.exit(f"{where}: history.csv row {generation} is generation {row['generation']}.")
        survivors = alive if generation == last else (born <= generation) & (generation < died)
        count = int(survivors.sum())
        if count != population_size:
            sys.exit(f"{where}: {count} survivors, expected {population_size}.")
        best = abs(fitness[survivors].min() - float(row["best_fitness"]))
        mean = abs(fitness[survivors].mean() - float(row["mean_fitness"]))
        if not best <= BEST_TOLERANCE:
            sys.exit(f"{where}: best fitness differs from history.csv by {best:.3g}.")
        if not mean <= MEAN_TOLERANCE:
            sys.exit(f"{where}: mean fitness differs from history.csv by {mean:.3g}.")
        best_gap, mean_gap = max(best_gap, best), max(mean_gap, mean)
        diversity[generation] = mean_pairwise_rms(genotypes[survivors])
    return diversity, best_gap, mean_gap


def diversity_row(method: str, seed: int | str, last: int | str, values: list[float | None]) -> dict:
    cells = ["" if value is None else f"{value:.3f}" for value in values]
    return dict(zip(DIVERSITY_FIELDS, [method, seed, last, *cells], strict=True))


def diversity_table(candidates: list[tuple[Path, dict]],
                    resolved: dict | None) -> tuple[list[dict], dict[str, float], str]:
    """Each EA method's runs, then their mean; also each method's sigma and a check summary.

    A generation past a run's last one is left empty; a method's mean uses the
    runs that reached it.
    """
    population_size = None if resolved is None else resolved["ea"]["population_size"]
    by_method: dict[str, list[tuple[int, int, list[float | None]]]] = {}
    sigmas: dict[str, float] = {}
    generations, best_gap, mean_gap = 0, 0.0, 0.0
    for folder, candidate in candidates:
        if candidate.get("algorithm") != "ea":
            continue
        method, seed = candidate["method"], candidate["seed"]
        if not (folder / "population.db").is_file():
            print(f"WARNING  {method} seed {seed}: no population.db; left out of diversity.csv.")
            continue
        if population_size is None:
            sys.exit("experiment.json is missing: survivors cannot be checked without its population size.")
        diversity, best, mean = survivor_diversity(folder, population_size, f"{method} seed {seed}")
        last = max(diversity)
        values = [diversity.get(g) for g in DIVERSITY_POINTS] + [diversity[last]]
        by_method.setdefault(method, []).append((seed, last, values))
        sigmas[method] = candidate["mutation_strength"]
        generations += len(diversity)
        best_gap, mean_gap = max(best_gap, best), max(mean_gap, mean)

    rows = []
    for method, runs in by_method.items():
        rows += [diversity_row(method, seed, last, values) for seed, last, values in runs]
        means = []
        for column in zip(*(values for _, _, values in runs), strict=True):
            present = [value for value in column if value is not None]
            means.append(float(np.mean(present)) if present else None)
        rows.append(diversity_row(method, "mean", "", means))

    if not by_method:
        return rows, sigmas, "No EA run with a population.db: diversity.csv has no rows."
    runs_checked = sum(len(runs) for runs in by_method.values())
    summary = (f"Survivors rebuilt and checked for {runs_checked} EA runs, {generations} generations: "
               f"{population_size} survivors in each; largest gap to history.csv "
               f"{best_gap:.3g} (best fitness), {mean_gap:.3g} (mean fitness).")
    return rows, sigmas, summary


def reference_levels(resolved: dict, sigmas: dict[str, float]) -> list[str]:
    """Diversity of a random initial population, and of two mutants of one parent."""
    # Two independent N(0, s^2) parameters differ by N(0, 2 s^2).
    s = resolved["evaluation_settings"]["controller"]["initial_weight_std"]
    lines = [f"  random N(0, {s:g}^2) population: {s:g} * sqrt(2) = {s * np.sqrt(2):.3f}"]
    # Each parameter is mutated with probability p by N(0, sigma^2), so two
    # mutants of one parent differ by a mean square of 2 p sigma^2.
    p = resolved["ea"]["mutation_probability"]
    for method, sigma in sigmas.items():
        lines.append(f"  two mutants of one parent, {method}: sqrt(2 * {p:g}) * {sigma:g} "
                     f"= {np.sqrt(2 * p) * sigma:.3f}")
    return lines


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    # Header even without rows, so an old table is never left behind.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, fields: tuple[str, ...], rows: list[dict]) -> None:
    widths = [max([len(f), *(len(str(row[f])) for row in rows)]) for f in fields]
    print(f"\n{title}")
    for cells in [fields, *([row[f] for f in fields] for row in rows)]:
        print("  " + "  ".join(f"{cell!s:<{width}}" for cell, width in zip(cells, widths, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder containing the run folders.")
    args = parser.parse_args()

    root = args.results.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    candidates = load_candidates(root)
    if not candidates:
        sys.exit(f"No best_candidate.json files found under {root}.")

    out = root / "analysis"
    out.mkdir(exist_ok=True)
    survival = survival_table(candidates)
    controllers = controller_table(candidates)
    resolved = experiment_settings(root)
    # Computed before any write: a failed survivor check leaves every file as it was.
    diversity, sigmas, checked = diversity_table(candidates, resolved)
    write_csv(out / "survival.csv", SURVIVAL_FIELDS, survival)
    write_csv(out / "controllers.csv", CONTROLLER_FIELDS, controllers)
    write_csv(out / "diversity.csv", DIVERSITY_FIELDS, diversity)

    print_table(f"Share of children kept by selection: born in generations {EARLY[0]}-{EARLY[1]}, "
                f"and in each run's last {LATE} generations (survival.csv)", SURVIVAL_FIELDS, survival)
    print_table("Best controller of each run (controllers.csv)", CONTROLLER_FIELDS, controllers)
    print_table("Genetic diversity of the survivors: mean over all pairs of the RMS difference per "
                "parameter (diversity.csv)", DIVERSITY_FIELDS, diversity)
    print(f"\n{checked}")
    if resolved is not None and sigmas:
        print("Reference levels (same measure):")
        print("\n".join(reference_levels(resolved, sigmas)))
    print(f"\nWrote {out / 'survival.csv'}, {out / 'controllers.csv'} and {out / 'diversity.csv'}")


if __name__ == "__main__":
    main()
