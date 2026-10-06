from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population
from controller import sample_weights
from evaluation import EvaluationResult, EvaluationSettings, evaluate

@dataclass
class EvaluationTracker:
    """Track how many simulations were performed during one EA run."""

    evaluations: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class GenerationRecord:
    """Fitness statistics for one generation."""

    generation: int
    evaluations: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float


@dataclass
class EARunResult:
    """Final information returned by one complete EA run."""

    best_fitness: float
    best_weights: np.ndarray
    evaluations: int
    generations_completed: int
    stopped_on_plateau: bool
    elapsed_evaluation_seconds: float
    history: list[GenerationRecord]


def individual_to_weights(
    individual: Individual,
) -> np.ndarray:
    """Convert an ARIEL individual's genotype to a NumPy weight vector."""

    return np.asarray(
        individual.genotype,
        dtype=np.float64,
    )


def create_individual(
    rng: np.random.Generator,
    settings: EvaluationSettings,
) -> Individual:
    """Create one randomly initialized neural-network controller."""

    weights = np.asarray(
        sample_weights(
            rng,
            settings.controller,
        ),
        dtype=np.float64,
    )

    if weights.ndim != 1:
        raise ValueError(
            "sample_weights must return a one-dimensional vector."
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "sample_weights returned non-finite values."
        )

    individual = Individual()
    individual.genotype = weights.tolist()

    return individual


def create_initial_population(
    population_size: int,
    rng: np.random.Generator,
    settings: EvaluationSettings,
) -> Population:
    """Create the initial random controller population."""

    if population_size < 1:
        raise ValueError(
            "population_size must be at least 1."
        )

    individuals = [
        create_individual(
            rng=rng,
            settings=settings,
        )
        for _ in range(population_size)
    ]

    return Population(individuals)


def evaluate_individual(
    individual: Individual,
    settings: EvaluationSettings,
) -> EvaluationResult:
    """Evaluate one controller using the simulation evaluator."""

    weights = individual_to_weights(individual)

    return evaluate(
        weights,
        settings,
    )


@EAOperation
def evaluate_population(
    population: Population,
    settings: EvaluationSettings,
    tracker: EvaluationTracker,
) -> Population:
    """Evaluate all individuals whose fitness is not yet known."""

    for individual in population:
        if individual.requires_eval:
            result = evaluate_individual(
                individual,
                settings,
            )
            individual.fitness = result.fitness

            tracker.evaluations += 1
            tracker.elapsed_seconds += result.elapsed_seconds

    return population


@EAOperation
def select_parents(
    population: Population,
    parent_fraction: float,
) -> Population:
    """Mark the best fraction of living individuals as parents."""

    if not 0 < parent_fraction <= 1:
        raise ValueError(
            "parent_fraction must be in the interval (0, 1]."
        )

    # Clear old parent-selection tags.
    for individual in population:
        individual.tags["selected_parent"] = False

    # Lower fitness is better.
    ranked = population.alive.sort(
        sort="min",
        attribute="fitness_",
    )

    number_of_parents = max(
        1,
        int(len(ranked) * parent_fraction),
    )

    for individual in ranked[:number_of_parents]:
        individual.tags["selected_parent"] = True

    return population


def get_selected_parents(
    population: Population,
) -> list[Individual]:
    """Return living individuals currently selected as parents."""

    return [
        individual
        for individual in population
        if individual.alive
        and individual.tags.get(
            "selected_parent",
            False,
        )
    ]


def mutate_weights(
    weights: np.ndarray,
    rng: np.random.Generator,
    mutation_strength: float,
    mutation_probability: float,
) -> np.ndarray:
    """Return a mutated copy of a neural-network weight vector."""

    if (
        not np.isfinite(mutation_strength)
        or mutation_strength <= 0
    ):
        raise ValueError(
            "mutation_strength must be finite and positive."
        )

    if (
        not np.isfinite(mutation_probability)
        or not 0 <= mutation_probability <= 1
    ):
        raise ValueError(
            "mutation_probability must be between 0 and 1."
        )

    # Copy first: never mutate the parent directly.
    child_weights = weights.copy()

    # Each weight independently has a chance of being mutated.
    mutation_mask = (
        rng.random(size=child_weights.shape)
        < mutation_probability
    )

    # Gaussian noise:
    # epsilon ~ N(0, mutation_strength^2)
    noise = rng.normal(
        loc=0.0,
        scale=mutation_strength,
        size=child_weights.shape,
    )

    child_weights[mutation_mask] += noise[mutation_mask]

    if not np.all(np.isfinite(child_weights)):
        raise FloatingPointError(
            "Mutation produced non-finite weights."
        )

    return child_weights


@EAOperation
def reproduce(
    population: Population,
    rng: np.random.Generator,
    number_of_children: int,
    mutation_strength: float,
    mutation_probability: float,
) -> Population:
    """Create mutated offspring from selected parents."""

    parents = get_selected_parents(population)

    if not parents:
        raise RuntimeError(
            "No parents were selected before reproduction."
        )

    if number_of_children < 1:
        raise ValueError(
            "number_of_children must be at least 1."
        )

    children: list[Individual] = []

    while len(children) < number_of_children:

        # Randomly choose one eligible parent.
        parent_index = int(
            rng.integers(
                0,
                len(parents),
            )
        )

        parent = parents[parent_index]

        parent_weights = individual_to_weights(
            parent
        )

        child_weights = mutate_weights(
            weights=parent_weights,
            rng=rng,
            mutation_strength=mutation_strength,
            mutation_probability=mutation_probability,
        )

        child = Individual()
        child.genotype = child_weights.tolist()
        child.tags["selected_parent"] = False

        children.append(child)

    # Current individuals + children temporarily coexist.
    population.extend(children)

    return population


@EAOperation
def select_survivors(
    population: Population,
    target_population_size: int,
) -> Population:
    """Keep exactly the best N individuals alive."""

    if target_population_size < 1:
        raise ValueError(
            "target_population_size must be at least 1."
        )

    # Lower fitness is better.
    ranked = population.alive.sort(
        sort="min",
        attribute="fitness_",
    )

    if len(ranked) < target_population_size:
        raise RuntimeError(
            "There are fewer candidates than requested survivors."
        )

    # Best N survive.
    for individual in ranked[:target_population_size]:
        individual.alive = True

    # Everyone else dies.
    for individual in ranked[target_population_size:]:
        individual.alive = False

    if len(population.alive) != target_population_size:
        raise RuntimeError(
            "Survivor selection produced "
            f"{len(population.alive)} living individuals; "
            f"expected {target_population_size}."
        )

    return population


def make_generation_record(
    population: Population,
    generation: int,
    evaluations: int,
) -> GenerationRecord:
    """Calculate fitness statistics for the supplied population."""

    # IMPORTANT:
    # The population passed here has already been fetched from the
    # database with only_alive=True.
    #
    # Therefore we iterate directly over population instead of calling
    # population.alive again.

    fitnesses = np.asarray(
        [
            individual.fitness
            for individual in population
        ],
        dtype=np.float64,
    )

    if len(fitnesses) == 0:
        raise RuntimeError(
            "Cannot summarize an empty population."
        )

    if not np.all(np.isfinite(fitnesses)):
        raise FloatingPointError(
            "Population contains non-finite fitness values."
        )

    return GenerationRecord(
        generation=generation,
        evaluations=evaluations,
        best_fitness=float(np.min(fitnesses)),
        mean_fitness=float(np.mean(fitnesses)),
        worst_fitness=float(np.max(fitnesses)),
    )


def run_evolution(
    *,
    rng: np.random.Generator,
    settings: EvaluationSettings,
    population_size: int,
    number_of_children: int,
    parent_fraction: float,
    mutation_strength: float,
    mutation_probability: float,
    plateau_patience: int,
    min_improvement: float,
    max_generations: int,
    database_path: Path,
    db_handling: str = "halt",
    best_weights_path: Path | None = None,
) -> EARunResult:
    """Run one EA configuration for one random seed."""

    if population_size < 1:
        raise ValueError(
            "population_size must be at least 1."
        )

    if number_of_children < 1:
        raise ValueError(
            "number_of_children must be at least 1."
        )

    if plateau_patience < 1:
        raise ValueError(
            "plateau_patience must be at least 1."
        )

    if (
        not np.isfinite(min_improvement)
        or min_improvement < 0
    ):
        raise ValueError(
            "min_improvement must be finite and non-negative."
        )

    if max_generations < 1:
        raise ValueError(
            "max_generations must be at least 1."
        )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    population = create_initial_population(
        population_size=population_size,
        rng=rng,
        settings=settings,
    )

    tracker = EvaluationTracker()

    evaluation_op = evaluate_population(settings=settings, tracker=tracker)

    # Evaluate the initial population before parent selection.
    population = evaluation_op(population)

    history = [
        make_generation_record(
            population=population,
            generation=0,
            evaluations=tracker.evaluations,
        )
    ]

    operations = [
        select_parents(
            parent_fraction=parent_fraction,
        ),
        reproduce(
            rng=rng,
            number_of_children=number_of_children,
            mutation_strength=mutation_strength,
            mutation_probability=mutation_probability,
        ),
        evaluation_op,
        select_survivors(target_population_size=population_size),
    ]

    ea = EA(
        population=population,
        operations=operations,
        num_steps=max_generations,
        is_maximisation=False,
        db_file_path=database_path,
        db_handling=db_handling,
    )

    reference_best = history[0].best_fitness

    generations_without_improvement = 0
    stopped_on_plateau = False
    generations_completed = 0

    for generation in range(
        1,
        max_generations + 1,
    ):
        ea.step()
        ea.fetch_population(only_alive=True, requires_eval=False)

        generations_completed = generation

        record = make_generation_record(
            population=ea.population,
            generation=generation,
            evaluations=tracker.evaluations,
        )

        history.append(record)

        # Lower fitness is better.
        improvement = (reference_best - record.best_fitness)

        if improvement > min_improvement:
            reference_best = record.best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        if generations_without_improvement >= plateau_patience:
            stopped_on_plateau = True
            break

    best = ea.get_solution(mode="best", only_alive=True)

    best_fitness = float(best.fitness)
    best_weights = individual_to_weights(best).copy()

    if best_weights_path is not None:
        best_weights_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(best_weights_path, best_weights)

    return EARunResult(
        best_fitness=best_fitness,
        best_weights=best_weights,
        evaluations=tracker.evaluations,
        generations_completed=generations_completed,
        stopped_on_plateau=stopped_on_plateau,
        elapsed_evaluation_seconds=tracker.elapsed_seconds,
        history=history,
    )