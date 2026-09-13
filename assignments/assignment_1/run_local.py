from pathlib import Path

from ariel.ec import EA

from ea_common import (
    create_initial_population,
    evaluate_population,
    load_targets,
    select_parents,
    select_survivors,
    set_seed,
)
from ea_local import reproduce_local

from experiment_config import (
    GENERATIONS,
    PARENT_FRACTION,
    POPULATION_SIZE,
    SEEDS,
)


CURRENT_PATH = Path(__file__).parent


def run_one_seed(seed: int) -> None:
    """Run one independent local-mutation experiment."""

    print()
    print("=" * 50)
    print(f"LOCAL MUTATION - SEED {seed}")
    print("=" * 50)

    set_seed(seed)

    output_dir = CURRENT_PATH / "results" / "local"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_path = (
        output_dir
        / f"seed_{seed}.db"
    )

    targets = load_targets()

    population = create_initial_population(
        population_size=POPULATION_SIZE,
    )

    evaluation_op = evaluate_population(
        targets=targets,
    )

    population = evaluation_op(population)

    operations = [
        select_parents(
            parent_fraction=PARENT_FRACTION,
        ),

        reproduce_local(
            number_of_children=POPULATION_SIZE,
        ),

        evaluation_op,

        select_survivors(
            target_population_size=POPULATION_SIZE,
        ),
    ]

    ea = EA(
        population=population,
        operations=operations,
        num_steps=GENERATIONS,
        is_maximisation=False,
        db_file_path=database_path,
        db_handling="halt",
    )

    ea.run()

    best = ea.get_solution(
        mode="best",
        only_alive=True,
    )

    print()
    print(f"Seed: {seed}")
    print(f"Best fitness: {best.fitness:.4f}")
    print(f"Saved to: {database_path}")


def main() -> None:
    for seed in SEEDS:
        run_one_seed(seed)


if __name__ == "__main__":
    main()