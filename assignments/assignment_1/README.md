# Local-Mutation EA

This folder contains the local-mutation variant for Assignment 1.

## Method

The algorithm uses ARIEL's tree-based genome representation.

For each generation:

1. Evaluate the current population using the provided tree-edit-distance fitness.
2. Select the best 50% of living individuals as possible parents.
3. Create one full population of children by randomly copying selected parents.
4. Apply `mutate_replace_node` once to each child.
5. Evaluate the new children.
6. Keep the best `POPULATION_SIZE` individuals from parents + children.

Lower fitness is better.

## Final configuration

- Population size: 50
- Generations: 100
- Parent fraction: 0.5
- Seeds: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Genome representation: Tree
- Local mutation operator: `mutate_replace_node`

With this setup, each run performs:

`50 + 50 * 100 = 5050` fitness evaluations.

## Main files

- `ea_common.py` - shared initialization, fitness evaluation, parent selection, and survivor selection.
- `ea_local.py` - local-mutation reproduction using `mutate_replace_node`.
- `experiment_config.py` - shared experiment parameters.
- `run_local.py` - runs the local-mutation experiments for all configured seeds.

## Running the experiments

From `assignments/assignment_1`:

```powershell
uv run python .\run_local.py
```

Each seed is saved to:

```text
results/local/seed_<seed>.db
```

For example:

```text
results/local/seed_1.db
results/local/seed_2.db
...
results/local/seed_10.db
```

`run_local.py` uses `db_handling="halt"`, so it will stop rather than overwrite an existing database. Delete an old database first only if you intentionally want to rerun that seed.
