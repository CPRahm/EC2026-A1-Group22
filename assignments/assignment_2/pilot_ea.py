import csv
from pathlib import Path

import numpy as np

from ea import run_evolution
from evaluation import EvaluationSettings


CURRENT_PATH = Path(__file__).parent


# ---------------------------------------------------------------------------
# PILOT SETTINGS ONLY
#
# These are NOT the final experiment settings.
# The purpose is only to compare candidate mutation strengths.
# ---------------------------------------------------------------------------

PILOT_SEEDS = [
    1,
    2,
    3,
]

MUTATION_STRENGTHS = [
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    0.75,
    1.00,
    1.50,
]

POPULATION_SIZE = 10
NUMBER_OF_CHILDREN = 10
PARENT_FRACTION = 0.5
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 15
PLATEAU_PATIENCE = 5
MIN_IMPROVEMENT = 0.001
SIMULATION_DURATION = 15.0


def main() -> None:
    print()
    print("=" * 60)
    print("MUTATION-STRENGTH PILOT")
    print("=" * 60)

    settings = EvaluationSettings(duration=SIMULATION_DURATION)

    output_dir = (
        CURRENT_PATH
        / "results"
        / "pilot"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for mutation_strength in MUTATION_STRENGTHS:
        for seed in PILOT_SEEDS:
            print()
            print("-" * 60)
            print(
                f"Mutation strength = {mutation_strength}, "
                f"seed = {seed}"
            )
            print("-" * 60)

            rng = np.random.default_rng(seed)

            strength_name = str(mutation_strength).replace(".", "_")

            database_path = (
                output_dir
                / f"sigma_{strength_name}_seed_{seed}.db"
            )

            best_weights_path = (
                output_dir
                / f"sigma_{strength_name}_seed_{seed}_best.npy"
            )

            result = run_evolution(
                rng=rng,
                settings=settings,

                population_size=POPULATION_SIZE,
                number_of_children=NUMBER_OF_CHILDREN,

                parent_fraction=PARENT_FRACTION,

                mutation_strength=mutation_strength,
                mutation_probability=MUTATION_PROBABILITY,

                plateau_patience=PLATEAU_PATIENCE,
                min_improvement=MIN_IMPROVEMENT,
                max_generations=MAX_GENERATIONS,

                database_path=database_path,

                db_handling="delete",

                best_weights_path=best_weights_path,
            )

            initial_best = (result.history[0].best_fitness)

            final_best = (result.best_fitness)

            improvement = (
                initial_best
                - final_best
            )

            print(
                f"Initial best fitness: "
                f"{initial_best:.4f}"
            )

            print(
                f"Final best fitness: "
                f"{final_best:.4f}"
            )

            print(
                f"Improvement: "
                f"{improvement:.4f}"
            )

            print(
                f"Evaluations: "
                f"{result.evaluations}"
            )

            print(
                f"Generations completed: "
                f"{result.generations_completed}"
            )

            print(
                f"Stopped on plateau: "
                f"{result.stopped_on_plateau}"
            )

            summary_rows.append(
                {
                    "mutation_strength": mutation_strength,
                    "seed": seed,
                    "initial_best": initial_best,
                    "final_best": final_best,
                    "improvement": improvement,
                    "evaluations": result.evaluations,
                    "generations_completed": result.generations_completed,
                    "stopped_on_plateau": result.stopped_on_plateau,
                }
            )

    summary_path = (
        output_dir
        / "pilot_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mutation_strength",
                "seed",
                "initial_best",
                "final_best",
                "improvement",
                "evaluations",
                "generations_completed",
                "stopped_on_plateau",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)

    for mutation_strength in MUTATION_STRENGTHS:

        rows = [
            row
            for row in summary_rows
            if row["mutation_strength"]
            == mutation_strength
        ]

        final_fitnesses = np.asarray(
            [
                row["final_best"]
                for row in rows
            ],
            dtype=float,
        )

        improvements = np.asarray(
            [
                row["improvement"]
                for row in rows
            ],
            dtype=float,
        )

        print()
        print(
            f"sigma = {mutation_strength}"
        )

        print(
            f"  mean final fitness = "
            f"{np.mean(final_fitnesses):.4f}"
        )

        print(
            f"  std final fitness  = "
            f"{np.std(final_fitnesses):.4f}"
        )

        print(
            f"  mean improvement   = "
            f"{np.mean(improvements):.4f}"
        )

    print()
    print(
        f"Pilot summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()