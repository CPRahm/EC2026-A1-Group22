import random
from pathlib import Path

import networkx as nx

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import mean_plus_std_tree_edit_distance

CURRENT_PATH = Path(__file__).parent

TARGET_DIR = CURRENT_PATH / "target_bodies"

MAX_MODULES = 20

# Fraction of the population allowed to reproduce.
PARENT_FRACTION = 0.5

def set_seed(seed: int) -> None:
    """Set the random seed used by the tree operators and our EA."""
    random.seed(seed)

def load_targets(
    target_dir: Path = TARGET_DIR,
) -> list[nx.DiGraph]:
    """Load all target robot bodies."""
    paths = sorted(target_dir.glob("*.json"))

    if not paths:
        raise FileNotFoundError(
            f"No target bodies found in {target_dir}"
        )

    return [load_graph_from_json(path) for path in paths]

def create_individual(
    max_modules: int = MAX_MODULES,
) -> Individual:
    """Create one random tree-genome individual."""

    genome = random_tree(max_modules=max_modules)

    individual = Individual()

    individual.genotype = genome.to_dict()

    return individual


def create_initial_population(
    population_size: int,
    max_modules: int = MAX_MODULES,
) -> Population:
    """Create the initial random population."""

    individuals = [
        create_individual(max_modules)
        for _ in range(population_size)
    ]

    return Population(individuals)

def individual_to_tree(individual: Individual) -> TreeGenome:
    """Reconstruct a TreeGenome from an ARIEL Individual."""
    return TreeGenome.from_dict(individual.genotype)


def evaluate_individual(
    individual: Individual,
    targets: list[nx.DiGraph],
) -> float:
    """Calculate the assignment fitness for one individual."""

    genome = individual_to_tree(individual)

    # For Tree encoding the phenotype is obtained directly from the tree.
    body = genome.to_networkx()

    return mean_plus_std_tree_edit_distance(
        body,
        targets,
    )

@EAOperation
def evaluate_population(
    population: Population,
    targets: list[nx.DiGraph],
) -> Population:
    """Evaluate every individual that currently requires evaluation."""

    for individual in population:
        if individual.requires_eval:
            individual.fitness = evaluate_individual(
                individual,
                targets,
            )

    return population

@EAOperation
def select_parents(
    population: Population,
    parent_fraction: float = PARENT_FRACTION,
) -> Population:
    """Mark the best fraction of living individuals as parents."""

    # Clear parent selection tags from the previous generation.
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
    """Return the living individuals selected for reproduction."""

    return [
        individual
        for individual in population
        if individual.alive
        and individual.tags.get("selected_parent", False)
    ]

@EAOperation
def select_survivors(
    population: Population,
    target_population_size: int,
) -> Population:
    """Keep exactly the best N individuals alive."""

    # Lower fitness is better.
    ranked = population.alive.sort(
        sort="min",
        attribute="fitness_",
    )

    # First N survive.
    for individual in ranked[:target_population_size]:
        individual.alive = True

    # Everyone after position N dies.
    for individual in ranked[target_population_size:]:
        individual.alive = False

    if len(population.alive) != target_population_size:
        raise RuntimeError(
            "Survivor selection produced "
            f"{len(population.alive)} alive individuals; "
            f"expected {target_population_size}."
        )

    return population