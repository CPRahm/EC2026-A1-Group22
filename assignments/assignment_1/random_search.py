"""Equal-budget random search baseline for Assignment 1.

Samples independent random tree genomes from the SAME distribution used to
create the EA's initial population (`create_individual` -> `random_tree`),
using the SAME number of fitness evaluations as the EA variants.

Budget matching
---------------
The EA performs `POPULATION_SIZE + POPULATION_SIZE * GENERATIONS`
= 50 + 50 * 100 = 5050 evaluations per run. This baseline performs exactly
the same number, so the comparison is at equal evaluation budget.

Random search has no generations. To place it on the same x-axis as the EAs,
evaluations are grouped into blocks of POPULATION_SIZE: block 0 corresponds
to the EA's initial population (generation 0), block g to generation g.

Output
------
results/random_search/seed_<seed>.csv with one row per evaluation:

    method,seed,eval_index,generation,fitness,best_so_far,n_nodes

`best_so_far` is the running minimum. Random search applies no selection
pressure, so its per-evaluation fitness is flat in expectation; best_so_far
is the meaningful comparator against the EAs' best-of-generation curves.
"""

import csv
from pathlib import Path

from ea_common import (
    MAX_MODULES,
    create_individual,
    evaluate_individual,
    individual_to_tree,
    load_targets,
    set_seed,
)

# --- Configuration: must match the EA variants exactly --- #
POPULATION_SIZE = 50
GENERATIONS = 100
EVAL_BUDGET = POPULATION_SIZE + POPULATION_SIZE * GENERATIONS  # 5050
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

CURRENT_PATH = Path(__file__).parent
RESULTS_DIR = CURRENT_PATH / "results" / "random_search"

CSV_FIELDS = [
    "method",
    "seed",
    "eval_index",
    "generation",
    "fitness",
    "best_so_far",
    "n_nodes",
]


def run_one_seed(seed, targets):
    """Run one independent random search and return its result rows."""
    set_seed(seed)

    rows = []
    best_so_far = float("inf")

    for eval_index in range(EVAL_BUDGET):
        individual = create_individual(max_modules=MAX_MODULES)

        fitness = evaluate_individual(individual, targets)

        body = individual_to_tree(individual).to_networkx()
        n_nodes = body.number_of_nodes()

        best_so_far = min(best_so_far, fitness)

        rows.append({
            "method": "random_search",
            "seed": seed,
            "eval_index": eval_index,
            "generation": eval_index // POPULATION_SIZE,
            "fitness": fitness,
            "best_so_far": best_so_far,
            "n_nodes": n_nodes,
        })

    return rows


def write_csv(path, rows):
    """Write result rows to a CSV file."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    targets = load_targets()
    print(f"loaded {len(targets)} target bodies")
    print(f"budget  {EVAL_BUDGET} evaluations per seed")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        out_path = RESULTS_DIR / f"seed_{seed}.csv"

        if out_path.exists():
            print(f"seed {seed}: {out_path.name} exists, skipping")
            continue

        rows = run_one_seed(seed, targets)
        write_csv(out_path, rows)

        print(
            f"seed {seed}: best {rows[-1]['best_so_far']:.4f} "
            f"-> {out_path.name}"
        )


if __name__ == "__main__":
    main()