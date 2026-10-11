"""Selection pressure and evolved controllers in one results folder.

Post-processing only: nothing is simulated. From the repository root:

    python assignments/assignment_2/mechanism_analysis.py <results folder>

Writes to <results folder>/analysis/ and prints both tables:

    survival.csv     per EA run, the share of children kept by (mu+lambda)
                     selection among those born in generations 1-50 and in
                     the run's last 100 generations (G-99..G, where G is the
                     last row of its history.csv); then one row per method
                     with the mean of its runs
    controllers.csv  per best_candidate.json: RMS of the weights, final x and
                     y, and the planar distance walked from the initial to the
                     final position

A child counts as kept if it is still alive or outlived its birth generation
(alive = 1 OR time_of_death > time_of_birth), as in
pilot_analysis.survival_by_phase. EA runs without a population.db are skipped
with a warning. Standalone: needs only sqlite3, json, csv and numpy.
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

SURVIVAL_FIELDS = ("method", "seed", "last_generation", "gen_1_50", "gen_last_100")
CONTROLLER_FIELDS = ("method", "seed", "weight_rms", "final_x", "final_y", "distance_walked")


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
    write_csv(out / "survival.csv", SURVIVAL_FIELDS, survival)
    write_csv(out / "controllers.csv", CONTROLLER_FIELDS, controllers)

    print_table(f"Share of children kept by selection: born in generations {EARLY[0]}-{EARLY[1]}, "
                f"and in each run's last {LATE} generations (survival.csv)", SURVIVAL_FIELDS, survival)
    print_table("Best controller of each run (controllers.csv)", CONTROLLER_FIELDS, controllers)
    print(f"\nWrote {out / 'survival.csv'} and {out / 'controllers.csv'}")


if __name__ == "__main__":
    main()
