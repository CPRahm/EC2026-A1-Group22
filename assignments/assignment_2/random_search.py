"""Equal-budget random-search baseline for Assignment 2.

Every candidate is drawn independently from the EA's initial distribution
(controller.sample_weights) and scored by the same evaluator. There is no
selection, so the best-so-far curve is what is compared against the EAs.

Budget: exactly ea_config.max_evaluations evaluations, the EA's cap. The EAs
may stop earlier on a plateau; the analysis then cuts every method at one
common evaluation count, so random search must reach at least that far.

Pairing: run with np.random.default_rng(seed), the stream ea.py also uses.
ea.py draws its initial population first, so for one seed the first
population_size random-search samples ARE the EA's initial population.

Checkpoints: history rows are taken at population_size + g * number_of_children
evaluations, the EA's generation boundaries, so both methods share one x-axis.
best_fitness is the best so far. batch_mean_fitness and batch_worst_fitness
describe only the samples drawn since the previous row; they are named apart
from the EA's survivor mean/worst because they measure something different.

This module only searches. run_experiments.py runs it and writes the files.
"""

import math
from dataclasses import dataclass

import numpy as np

if __package__:
    from .controller import sample_weights
    from .ea import Evaluator
    from .evaluation import EvaluationSettings, evaluate
    from .experiment_config import EAConfig
else:
    from controller import sample_weights
    from ea import Evaluator
    from evaluation import EvaluationSettings, evaluate
    from experiment_config import EAConfig


@dataclass(frozen=True)
class RandomSearchRecord:
    """One checkpoint, aligned with an EA generation boundary."""

    generation: int
    evaluations: int
    best_fitness: float
    batch_mean_fitness: float
    batch_worst_fitness: float


@dataclass
class RandomSearchResult:
    best_fitness: float
    best_weights: np.ndarray
    best_evaluation: int  # 1-based index of the evaluation that found the best
    evaluations: int
    history: list[RandomSearchRecord]


def run_random_search(
    *,
    rng: np.random.Generator,
    settings: EvaluationSettings,
    ea_config: EAConfig,
    evaluator: Evaluator = evaluate,
) -> RandomSearchResult:
    """Sample and score ea_config.max_evaluations independent controllers."""

    checkpoints = {
        ea_config.population_size + g * ea_config.number_of_children: g
        for g in range(ea_config.max_generations + 1)
    }

    best_fitness = math.inf
    best_weights = None
    best_evaluation = 0
    batch: list[float] = []
    history: list[RandomSearchRecord] = []

    for evaluation in range(1, ea_config.max_evaluations + 1):
        weights = np.asarray(sample_weights(rng, settings.controller), dtype=np.float64)
        fitness = float(evaluator(weights, settings).fitness)
        batch.append(fitness)

        # Lower is better; on a tie the earlier candidate is kept.
        if fitness < best_fitness:
            best_fitness = fitness
            best_weights = weights.copy()
            best_evaluation = evaluation

        if evaluation in checkpoints:
            history.append(RandomSearchRecord(
                generation=checkpoints[evaluation],
                evaluations=evaluation,
                best_fitness=best_fitness,
                batch_mean_fitness=float(np.mean(batch)),
                batch_worst_fitness=float(np.max(batch)),
            ))
            batch = []

    if best_weights is None or not math.isfinite(best_fitness):
        raise RuntimeError("Random search produced no finite fitness.")

    return RandomSearchResult(
        best_fitness=best_fitness,
        best_weights=best_weights,
        best_evaluation=best_evaluation,
        evaluations=ea_config.max_evaluations,
        history=history,
    )
