import random

from ariel.ec import EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import mutate_replace_node
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from ea_common import get_selected_parents


@EAOperation
def reproduce_local(
    population: Population,
    number_of_children: int,
) -> Population:
    """Create offspring from selected parents using local mutation."""

    parents = get_selected_parents(population)

    if not parents:
        raise RuntimeError(
            "No parents were selected before reproduction."
        )

    children: list[Individual] = []

    while len(children) < number_of_children:

        # Pick one selected parent randomly.
        parent = random.choice(parents)

        # Reconstruct an independent copy of its tree genome.
        child_genome = TreeGenome.from_dict(
            parent.genotype
        )

        # Apply the LOCAL mutation operator.
        mutate_replace_node(child_genome)

        # Store the mutated genome in a new ARIEL Individual.
        child = Individual()
        child.genotype = child_genome.to_dict()

        # This child has not been selected as a parent.
        child.tags["selected_parent"] = False

        children.append(child)

    # Add the new children to the existing population.
    population.extend(children)

    return population