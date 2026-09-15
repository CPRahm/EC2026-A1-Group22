"""Analysis tools for Assignment 1.

This file loads results from the two evolutionary algorithms and the
random-search baseline. All result formats are converted into one common
representation for convergence analysis, final-fitness statistics, paired
comparison, and plotting.
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from experiment_config import GENERATIONS, POPULATION_SIZE


CURRENT_PATH = Path(__file__).parent
RESULTS_DIR = CURRENT_PATH / "results"
LOCAL_DIR = RESULTS_DIR / "local"
SUBTREE_DIR = RESULTS_DIR / "subtree"
RANDOM_SEARCH_DIR = RESULTS_DIR / "random_search"

# ============================================================================ #
#  1. LOAD AN ARIEL DATABASE
# ============================================================================ #

def load_ea_database(database_path: Path) -> pd.DataFrame:
    """Load the individual table from one ARIEL experiment database."""

    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    with sqlite3.connect(database_path) as connection:
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """,
            connection,
        )

        if "individual" not in tables["name"].values:
            raise ValueError(
                f"{database_path.name} does not contain an 'individual' table."
            )

        individuals = pd.read_sql_query(
            """
            SELECT *
            FROM individual
            ORDER BY id
            """,
            connection,
        )

    required_columns = {
        "id",
        "alive",
        "time_of_birth",
        "time_of_death",
        "fitness_",
    }
    missing_columns = required_columns - set(individuals.columns)

    if missing_columns:
        raise ValueError(
            "Database is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # All stored individuals should already have a fitness value.
    if individuals["fitness_"].isna().any():
        raise ValueError(
            f"{database_path.name} contains individuals without a fitness value."
        )

    return individuals

# ============================================================================ #
#  2. RECONSTRUCT ONE EA GENERATION
# ============================================================================ #

def get_population_at_generation(
    individuals: pd.DataFrame,
    generation: int,
) -> pd.DataFrame:
    """Return the surviving population after a particular generation."""

    minimum_generation = int(individuals["time_of_birth"].min())
    maximum_generation = int(individuals["time_of_birth"].max())

    if generation < minimum_generation:
        raise ValueError(
            f"Generation {generation} is before the first generation "
            f"({minimum_generation})."
        )

    if generation > maximum_generation:
        raise ValueError(
            f"Generation {generation} is after the final generation "
            f"({maximum_generation})."
        )

    # An individual belongs to generation g when it has already been born and
    # has not died yet. The alive flag is needed for ARIEL's final generation,
    # where final survivors can have time_of_death equal to the final step.
    born = individuals["time_of_birth"] <= generation
    still_alive = (
        (individuals["time_of_death"] > generation)
        | (individuals["alive"] == 1)
    )

    population = individuals[born & still_alive].copy()
    population = population.sort_values("fitness_", ascending=True)

    if len(population) != POPULATION_SIZE:
        raise RuntimeError(
            f"Generation {generation} contains {len(population)} surviving "
            f"individuals; expected {POPULATION_SIZE}."
        )

    return population

# ============================================================================ #
#  3. ANALYSE ONE EA RUN
# ============================================================================ #

def analyse_ea_run(
    individuals: pd.DataFrame,
    method: str,
    seed: int,
) -> pd.DataFrame:
    """Calculate generation-level statistics for one EA run."""

    generations = sorted(
        individuals["time_of_birth"].astype(int).unique()
    )
    rows = []

    for generation in generations:
        population = get_population_at_generation(individuals, generation)
        fitness_values = population["fitness_"].astype(float)

        # Every row with time_of_birth <= g corresponds to an evaluation that
        # has already happened by this checkpoint.
        evaluations = int(
            (individuals["time_of_birth"] <= generation).sum()
        )
        best_individual = population.iloc[0]

        rows.append(
            {
                "method": method,
                "seed": seed,
                "generation": generation,
                "evaluations": evaluations,
                "best_fitness": fitness_values.min(),
                "mean_fitness": fitness_values.mean(),
                "worst_fitness": fitness_values.max(),
                "population_fitness_std": fitness_values.std(ddof=0),
                "best_individual_id": best_individual["id"],
            }
        )

    return pd.DataFrame(rows)

# ============================================================================ #
#  4. LOAD ALL RUNS OF ONE EA METHOD
# ============================================================================ #

def get_seed_from_path(path: Path) -> int:
    """Extract the seed number from a file such as seed_3.db."""

    try:
        return int(path.stem.split("_")[-1])
    except ValueError as error:
        raise ValueError(
            f"Could not extract seed from filename: {path.name}"
        ) from error

def load_all_ea_runs(result_dir: Path, method: str) -> pd.DataFrame:
    """Load and analyse every EA database inside one result directory."""

    database_paths = sorted(
        result_dir.glob("seed_*.db"),
        key=get_seed_from_path,
    )

    if not database_paths:
        raise FileNotFoundError(f"No EA databases found in {result_dir}")

    all_runs = []

    for database_path in database_paths:
        seed = get_seed_from_path(database_path)
        individuals = load_ea_database(database_path)
        run_statistics = analyse_ea_run(
            individuals=individuals,
            method=method,
            seed=seed,
        )
        all_runs.append(run_statistics)

    return pd.concat(all_runs, ignore_index=True)

# ============================================================================ #
#  5. LOAD ONE RANDOM-SEARCH RUN
# ============================================================================ #

def load_random_search_run(csv_path: Path) -> pd.DataFrame:
    """Convert one random-search CSV to generation-equivalent checkpoints."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Random-search file not found: {csv_path}"
        )

    results = pd.read_csv(csv_path)

    required_columns = {
        "method",
        "seed",
        "eval_index",
        "generation",
        "fitness",
        "best_so_far",
        "n_nodes",
    }
    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing_columns)}"
        )

    expected_evaluations = POPULATION_SIZE * (GENERATIONS + 1)

    if len(results) != expected_evaluations:
        raise ValueError(
            f"{csv_path.name} contains {len(results)} evaluations; "
            f"expected {expected_evaluations}."
        )

    # Verify that the stored best_so_far column really is the running minimum.
    calculated_best = results["fitness"].cummin()
    if not calculated_best.equals(results["best_so_far"]):
        raise ValueError(
            f"{csv_path.name} contains an inconsistent 'best_so_far' column."
        )

    rows = []

    # Random search has no surviving population. We therefore keep only the
    # best-so-far value after each block of 50 evaluations, which places it on
    # the same evaluation checkpoints as the EAs.
    for generation, block in results.groupby("generation", sort=True):
        final_row = block.iloc[-1]
        evaluations = int(final_row["eval_index"]) + 1

        rows.append(
            {
                "method": "random_search",
                "seed": int(final_row["seed"]),
                "generation": int(generation),
                "evaluations": evaluations,
                "best_fitness": float(final_row["best_so_far"]),
                "mean_fitness": float("nan"),
                "worst_fitness": float("nan"),
                "population_fitness_std": float("nan"),
                "best_individual_id": float("nan"),
            }
        )

    return pd.DataFrame(rows)

# ============================================================================ #
#  6. LOAD ALL RANDOM-SEARCH RUNS
# ============================================================================ #

def load_all_random_search_runs(result_dir: Path) -> pd.DataFrame:
    """Load every random-search CSV inside the result directory."""

    csv_paths = sorted(
        result_dir.glob("seed_*.csv"),
        key=get_seed_from_path,
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"No random-search CSV files found in {result_dir}"
        )

    all_runs = []

    for csv_path in csv_paths:
        all_runs.append(load_random_search_run(csv_path))

    return pd.concat(all_runs, ignore_index=True)

# ============================================================================ #
#  7. COMBINE ALL METHODS
# ============================================================================ #

def load_all_results() -> pd.DataFrame:
    """Load Local, Subtree and Random Search into one common table."""

    local_results = load_all_ea_runs(
        result_dir=LOCAL_DIR,
        method="local",
    )
    subtree_results = load_all_ea_runs(
        result_dir=SUBTREE_DIR,
        method="subtree",
    )
    random_results = load_all_random_search_runs(RANDOM_SEARCH_DIR)

    return pd.concat(
        [local_results, subtree_results, random_results],
        ignore_index=True,
    )

# ============================================================================ #
#  8. VALIDATE CONVERGENCE DATA
# ============================================================================ #

def validate_convergence_data(results: pd.DataFrame) -> None:
    """Check that every run has a valid best-so-far fitness trajectory."""

    for (method, seed), run in results.groupby(["method", "seed"]):
        run = run.sort_values("evaluations")
        fitness_values = run["best_fitness"].astype(float)

        # Lower fitness is better, so a best-so-far trajectory must never rise.
        fitness_changes = fitness_values.diff()
        if (fitness_changes > 1e-10).any():
            raise ValueError(
                f"Best fitness becomes worse in {method}, seed {seed}."
            )

        if len(run) != GENERATIONS + 1:
            raise ValueError(
                f"{method}, seed {seed} contains {len(run)} checkpoints; "
                f"expected {GENERATIONS + 1}."
            )

# ============================================================================ #
#  9. ACROSS-RUN CONVERGENCE STATISTICS
# ============================================================================ #

def calculate_convergence_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and spread of best fitness across independent runs."""

    return (
        results
        .groupby(
            ["method", "generation", "evaluations"],
            as_index=False,
        )
        .agg(
            mean_best_fitness=("best_fitness", "mean"),
            std_best_fitness=("best_fitness", "std"),
            min_best_fitness=("best_fitness", "min"),
            max_best_fitness=("best_fitness", "max"),
            number_of_runs=("seed", "nunique"),
        )
    )

# ============================================================================ #
#  10. FINAL-FITNESS STATISTICS
# ============================================================================ #

def calculate_final_fitness_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise the final best fitness across independent runs."""

    final_indices = (
        results
        .groupby(["method", "seed"])["evaluations"]
        .idxmax()
    )
    final_runs = results.loc[final_indices].copy()

    return (
        final_runs
        .groupby("method", as_index=False)
        .agg(
            number_of_runs=("seed", "nunique"),
            mean_final_fitness=("best_fitness", "mean"),
            std_final_fitness=("best_fitness", "std"),
            best_run_fitness=("best_fitness", "min"),
            worst_run_fitness=("best_fitness", "max"),
        )
    )

# ============================================================================ #
#  11. PAIRED FINAL-FITNESS COMPARISON
# ============================================================================ #

def compare_local_and_subtree(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare final Local and Subtree fitness using matched seeds."""

    ea_results = results[
        results["method"].isin(["local", "subtree"])
    ].copy()

    final_indices = (
        ea_results
        .groupby(["method", "seed"])["evaluations"]
        .idxmax()
    )
    final_runs = ea_results.loc[
        final_indices,
        ["method", "seed", "best_fitness"],
    ].copy()

    # One row per matched seed makes the paired comparison explicit.
    paired = (
        final_runs
        .pivot(
            index="seed",
            columns="method",
            values="best_fitness",
        )
        .reset_index()
    )

    if paired[["local", "subtree"]].isna().any().any():
        raise ValueError(
            "Some seeds do not contain both Local and Subtree results."
        )

    # Positive means Local has higher (worse) fitness, so Subtree wins.
    paired["local_minus_subtree"] = paired["local"] - paired["subtree"]

    def winner_from_difference(difference: float) -> str:
        if difference > 0:
            return "subtree"
        if difference < 0:
            return "local"
        return "tie"

    paired["winner"] = paired["local_minus_subtree"].apply(
        winner_from_difference
    )
    differences = paired["local_minus_subtree"]

    # Two-sided paired Wilcoxon test. The research question asks whether the
    # mutation strategies differ, rather than assuming one direction in advance.
    test_result = wilcoxon(
        paired["local"],
        paired["subtree"],
        alternative="two-sided",
    )

    statistics = {
        "number_of_pairs": len(paired),
        "mean_difference": differences.mean(),
        "median_difference": differences.median(),
        "std_difference": differences.std(ddof=1),
        "subtree_wins": int((differences > 0).sum()),
        "local_wins": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "wilcoxon_statistic": float(test_result.statistic),
        "wilcoxon_p_value": float(test_result.pvalue),
    }

    return paired, statistics

# ============================================================================ #
#  12. PLOT CONVERGENCE
# ============================================================================ #

def plot_convergence(convergence: pd.DataFrame) -> None:
    """Plot mean best fitness and standard deviation across runs."""

    plt.figure(figsize=(9, 6))

    methods = [
        ("local", "Local mutation"),
        ("subtree", "Subtree replacement"),
        ("random_search", "Random search"),
    ]

    for method, label in methods:
        method_data = convergence[
            convergence["method"] == method
        ].sort_values("evaluations")

        x = method_data["evaluations"].to_numpy()
        mean = method_data["mean_best_fitness"].to_numpy()
        std = method_data["std_best_fitness"].to_numpy()

        plt.plot(x, mean, label=label)
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    plt.xlabel("Fitness evaluations")
    plt.ylabel("Best fitness (lower is better)")
    plt.title("Convergence across independent runs")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = RESULTS_DIR / "convergence.png"
    plt.savefig(output_path, dpi=300)

    print()
    print(f"Convergence plot saved to: {output_path}")

    # Close the figure so the script returns to the terminal immediately.
    plt.close()

# ============================================================================ #
#  13. MAIN
# ============================================================================ #

def main() -> None:
    """Load all results, print statistics and create the convergence plot."""

    results = load_all_results()
    validate_convergence_data(results)

    convergence = calculate_convergence_summary(results)
    final_fitness = calculate_final_fitness_summary(results)
    paired_results, paired_statistics = compare_local_and_subtree(results)

    print()
    print("=" * 70)
    print("FINAL FITNESS ACROSS RUNS")
    print("=" * 70)
    print(final_fitness.round(4).to_string(index=False))

    print()
    print("=" * 70)
    print("CONVERGENCE CHECKPOINTS")
    print("=" * 70)

    # Print only representative checkpoints. The DataFrame still contains all
    # 101 checkpoints and is used in full for the convergence plot.
    selected_generations = [0, 1, 10, 25, 50, 75, 100]
    checkpoints = convergence[
        convergence["generation"].isin(selected_generations)
    ]

    print(
        checkpoints[
            [
                "method",
                "generation",
                "evaluations",
                "mean_best_fitness",
                "std_best_fitness",
                "number_of_runs",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("PAIRED LOCAL VS SUBTREE FINAL FITNESS")
    print("=" * 70)
    print(paired_results.round(4).to_string(index=False))

    print()
    print(
        f"Mean Local - Subtree difference: "
        f"{paired_statistics['mean_difference']:.4f}"
    )
    print(
        f"Median Local - Subtree difference: "
        f"{paired_statistics['median_difference']:.4f}"
    )
    print(
        f"SD of paired differences: "
        f"{paired_statistics['std_difference']:.4f}"
    )
    print(f"Subtree wins: {paired_statistics['subtree_wins']}")
    print(f"Local wins: {paired_statistics['local_wins']}")
    print(f"Ties: {paired_statistics['ties']}")
    print(
        f"Wilcoxon statistic: "
        f"{paired_statistics['wilcoxon_statistic']:.4f}"
    )
    print(
        f"Wilcoxon p-value: "
        f"{paired_statistics['wilcoxon_p_value']:.6f}"
    )

    plot_convergence(convergence)


if __name__ == "__main__":
    main()