"""Single source of truth for every Assignment 2 experiment setting.

run_experiments.py reads all of its numbers from here; nothing else hard-codes
a population size, mutation strength, budget or seed. Three experiments exist:

FINAL  The comparison reported in the paper: two EA mutation strengths and
       random search, each over FINAL_SEEDS. (run_experiments.py refuses to
       start final runs while FINAL_SIGMAS or FINAL_EA is None.)
PILOT  The mutation-strength screen whose results justify FINAL_SIGMAS and the
       final budget. Plateau stopping is disabled so the full curves are seen.
SMOKE  A tiny pipeline test. Its numbers carry no meaning and its outputs are
       written to the Git-ignored __data__ folder.

Controller, body, world, target and simulation length are Chris's frozen
EvaluationSettings and are identical in all three experiments.
"""

from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .evaluation import EvaluationSettings
else:
    from evaluation import EvaluationSettings

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


@dataclass(frozen=True)
class EAConfig:
    """Settings passed to ea.run_evolution; also fixes random search's budget."""

    population_size: int
    number_of_children: int
    parent_fraction: float
    mutation_probability: float
    max_generations: int
    plateau_patience: int
    min_improvement: float

    def __post_init__(self) -> None:
        for name in ("population_size", "number_of_children", "max_generations", "plateau_patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.parent_fraction <= 1:
            raise ValueError("parent_fraction must be in (0, 1].")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0, 1].")
        if not self.min_improvement >= 0:
            raise ValueError("min_improvement must be non-negative.")

    @property
    def max_evaluations(self) -> int:
        """Evaluation cap: initial population plus offspring of every generation."""
        return self.population_size + self.max_generations * self.number_of_children

    @property
    def plateau_stopping(self) -> bool:
        return self.plateau_patience <= self.max_generations


# --- Shared by every experiment ------------------------------------------- #

EVALUATION = EvaluationSettings()

# Andrada's EA settings, as used in pilot_ea.py and pilot_stopping.py.
POPULATION_SIZE = 10
NUMBER_OF_CHILDREN = 10
PARENT_FRACTION = 0.5
MUTATION_PROBABILITY = 0.2

# A simulation that raises (e.g. MuJoCo warnings, non-finite state) scores
# this many metres: worse than any distance reachable in one evaluation, so a
# failed candidate never survives selection. A run with more failures than
# MAX_FAILED_EVALUATIONS is treated as crashed rather than as a result.
FAILURE_FITNESS = 1000.0
MAX_FAILED_EVALUATIONS = 10

RANDOM_SEARCH = "random_search"


def condition_name(mutation_strength: float) -> str:
    """Folder and method name of an EA condition, e.g. 'ea_sigma_0.1'."""
    return f"ea_sigma_{mutation_strength:g}"


# --- FINAL: approved after the screen (results/pilot_sigma) ---------------- #

# One strength from each regime the screen showed: sigma <= 0.2 either got
# very close or stalled (bimodal across seeds); sigma = 0.5 (= the initial
# weight std) was the only one to beat random search on every seed, with the
# smallest spread. 5x apart: ~9% vs ~45% of the genome's norm per mutation.
FINAL_SIGMAS: tuple[float, float] | None = (0.1, 0.5)
# Fixed budget, no early stopping: in the screen, every EA condition's mean
# curve had made >= 96% of its 10,010-evaluation gain by 7,510 evaluations
# (single runs >= 90%), while ea.py's plateau rule (P=100, delta=0.01) would
# have stopped all 18 screen runs early, one of them 0.90 m short of its
# final result. plateau_patience > max_generations disables the rule.
FINAL_EA: EAConfig | None = EAConfig(
    population_size=POPULATION_SIZE,
    number_of_children=NUMBER_OF_CHILDREN,
    parent_fraction=PARENT_FRACTION,
    mutation_probability=MUTATION_PROBABILITY,
    max_generations=1000,
    plateau_patience=1001,
    min_improvement=0.0,
)
# Disjoint from every pilot seed (teammates' pilots used 0-3).
FINAL_SEEDS = (101, 102, 103, 104, 105)
FINAL_ROOT = HERE / "results" / "final"

# --- PILOT: mutation-strength screen --------------------------------------- #

PILOT_SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
PILOT_MAX_GENERATIONS = 1000
PILOT_EA = EAConfig(
    population_size=POPULATION_SIZE,
    number_of_children=NUMBER_OF_CHILDREN,
    parent_fraction=PARENT_FRACTION,
    mutation_probability=MUTATION_PROBABILITY,
    max_generations=PILOT_MAX_GENERATIONS,
    plateau_patience=PILOT_MAX_GENERATIONS + 1,  # never reached: stopping disabled
    min_improvement=0.0,
)
PILOT_SEEDS = (1, 2, 3)
# Raw screen data is large (one ~27 MB database per EA run) and exactly
# reproducible with `run_experiments.py --pilot`, so it stays out of Git.
# pilot_analysis.py writes the small tables and figure the report cites to
# PILOT_SUMMARY_ROOT, which is tracked.
PILOT_ROOT = REPO_ROOT / "__data__" / "assignment_2" / "pilot_sigma"
PILOT_SUMMARY_ROOT = HERE / "results" / "pilot_sigma"

# --- SMOKE: pipeline test only --------------------------------------------- #

SMOKE_SIGMAS = (0.1, 0.5)  # test-only placeholders, not a selection
SMOKE_EA = EAConfig(
    population_size=4,
    number_of_children=4,
    parent_fraction=PARENT_FRACTION,
    mutation_probability=MUTATION_PROBABILITY,
    max_generations=3,
    plateau_patience=2,
    min_improvement=0.001,
)
SMOKE_SEEDS = (0, 1)
SMOKE_ROOT = REPO_ROOT / "__data__" / "assignment_2" / "smoke"

# Runs are written here first and moved into their results folder only once
# complete, so a results folder never contains a partial run.
STAGING_ROOT = REPO_ROOT / "__data__" / "assignment_2" / "_staging"
