import random

from ariel.ec import EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import mutate_subtree_replacement
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from ea_common import get_selected_parents


@EAOperation
def reproduce_subtree(
    population: Population,
    number_of_children: int,
) -> Population:
    """Create offspring from selected parents using subtree replacement.

    This is `ea_local.reproduce_local` with a single line changed: the
    mutation operator. Everything else - parent pool, copying, one mutation
    per child, tagging, and how children join the population - is identical,
    so the two variants differ in exactly one thing.

    What `mutate_subtree_replacement` actually does, per ARIEL's
    implementation in `ariel.ec.genotypes.tree.operators`:

    1. Pick one non-core node uniformly at random.
    2. Remember its parent and the face it hangs on, then delete that node
       and its entire subtree.
    3. Generate a fresh `random_tree` of 1-3 modules and graft ONE random
       branch of it back onto the remembered parent/face.
    4. Prune invalid edges and validate; if validation fails, the whole
       mutation is rolled back and the child is a copy of its parent.

    Note the asymmetry the name hides: an arbitrarily large subtree can be
    removed, but at most three nodes are ever put back. Unlike
    `mutate_replace_node`, however, this operator CAN increase node count,
    so genome size is free to move in both directions.
    """

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

        # Apply the SUBTREE-REPLACEMENT mutation operator.
        #
        # `max_modules` is left at its default on purpose. The operator
        # computes `random.randint(1, min(3, max_modules))`, so every value
        # >= 3 behaves identically: the grafted branch is always 1-3 nodes.
        # Passing `max_modules=20` here would imply a module budget that
        # this operator does not actually enforce.
        mutate_subtree_replacement(child_genome)

        # Store the mutated genome in a new ARIEL Individual.
        child = Individual()
        child.genotype = child_genome.to_dict()

        # This child has not been selected as a parent.
        child.tags["selected_parent"] = False

        children.append(child)

    # Add the new children to the existing population.
    population.extend(children)

    return population
