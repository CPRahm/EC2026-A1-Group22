import csv
from pathlib import Path

import numpy as np

from ea import run_evolution
from evaluation import EvaluationSettings


CURRENT_PATH = Path(__file__).parent


# ---------------------------------------------------------------------------
# STOPPING-RULE PILOT ONLY
#
# We already selected two promising mutation strengths.
# This pilot runs them for longer so we can inspect when improvement slows.
# ---------------------------------------------------------------------------

PILOT_SEEDS = [
    1,
    2,
]

MUTATION_STRENGTHS = [
    0.1,
    0.5,
]

POPULATION_SIZE = 10
NUMBER_OF_CHILDREN = 10

PARENT_FRACTION = 0.5
MUTATION_PROBABILITY = 0.2

# Run substantially longer than the previous 15-generation pilot.
MAX_GENERATIONS = 40

# Disable practical plateau stopping during this pilot.
# We want to SEE the full curve first.
PLATEAU_PATIENCE = MAX_GENERATIONS + 1

# Improvement threshold we will inspect afterwards.
MIN_IMPROVEMENT = 0.001

SIMULATION_DURATION = 15.0


def main() -> None:
    print()
    print("=" * 60)
    print("STOPPING-RULE PILOT")
    print("=" * 60)

    settings = EvaluationSettings(
        duration=SIMULATION_DURATION,
    )

    output_dir = (
        CURRENT_PATH
        / "results"
        / "stopping_pilot"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_rows = []

    for mutation_strength in MUTATION_STRENGTHS:
        for seed in PILOT_SEEDS:
            print()
            print("-" * 60)
            print(
                f"sigma = {mutation_strength}, "
                f"seed = {seed}"
            )
            print("-" * 60)

            rng = np.random.default_rng(
                seed
            )

            strength_name = str(
                mutation_strength
            ).replace(".", "_")

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

                # Effectively disable plateau stopping for this pilot.
                plateau_patience=PLATEAU_PATIENCE,
                min_improvement=MIN_IMPROVEMENT,
                max_generations=MAX_GENERATIONS,

                database_path=database_path,
                db_handling="delete",

                best_weights_path=best_weights_path,
            )

            print(
                f"Final best fitness: "
                f"{result.best_fitness:.4f}"
            )

            print(
                f"Evaluations: "
                f"{result.evaluations}"
            )

            print(
                f"Generations completed: "
                f"{result.generations_completed}"
            )

            reference_best = (result.history[0].best_fitness)

            last_meaningful_improvement = 0

            for record in result.history[1:]:
                improvement = (reference_best - record.best_fitness)
                if improvement > MIN_IMPROVEMENT:
                    reference_best = (record.best_fitness)

                    last_meaningful_improvement = (record.generation)

            print(
                "Last meaningful improvement: "
                f"generation "
                f"{last_meaningful_improvement}"
            )

            for record in result.history:
                history_rows.append(
                    {
                        "mutation_strength":
                            mutation_strength,

                        "seed":
                            seed,

                        "generation":
                            record.generation,

                        "evaluations":
                            record.evaluations,

                        "best_fitness":
                            record.best_fitness,

                        "mean_fitness":
                            record.mean_fitness,

                        "worst_fitness":
                            record.worst_fitness,
                    }
                )

    history_path = (
        output_dir
        / "stopping_pilot_history.csv"
    )

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mutation_strength",
                "seed",
                "generation",
                "evaluations",
                "best_fitness",
                "mean_fitness",
                "worst_fitness",
            ],
        )

        writer.writeheader()
        writer.writerows(
            history_rows
        )

    print()
    print("=" * 60)
    print("STOPPING PILOT COMPLETE")
    print("=" * 60)

    print(
        f"History saved to: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()