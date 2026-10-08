"""Run the Assignment 2 experiments: one command per condition, one process per seed.

Run from the repository root with the project environment:

    python assignments/assignment_2/run_experiments.py --list
    python assignments/assignment_2/run_experiments.py --benchmark 10
    python assignments/assignment_2/run_experiments.py --smoke
    python assignments/assignment_2/run_experiments.py --pilot --workers 11
    python assignments/assignment_2/run_experiments.py --condition ea_sigma_0.1 --workers 5
    python assignments/assignment_2/run_experiments.py --condition all --workers 15

Every setting comes from experiment_config.py. Each (condition, seed) runs in
its own operating-system process: MuJoCo's control callback is process-global,
and a crash then only affects that one run.

One folder per run, <root>/<condition>/seed_<n>/:
    config.json               resolved settings, seed, Git commit, source hashes
    history.csv               one row per generation/checkpoint (analysis.py)
    evaluations.csv           one row per evaluate() call, failures included
    failed_evaluations.jsonl  only if a simulation failed: error and weights
    best_candidate.json       best weights and score (analysis.py, replay.py)
    population.db             ARIEL's database of every EA individual (EA only)
    summary.json              written last, status 'completed'
    run_log.txt               the run's console output

Runs are built in __data__/assignment_2/_staging and moved into <root> only
when complete. A results folder therefore never holds a partial run, finished
runs are never overwritten, and repeating a command runs only missing seeds.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import IO

import ariel
import numpy as np

# Plain imports, as in ea.py: run this file as a script, not as a package.
import experiment_config as cfg
from controller import sample_weights
from ea import run_evolution
from evaluation import EvaluationResult, EvaluationSettings, evaluate
from random_search import run_random_search

SOURCE_FILES = (
    "controller.py", "evaluation.py", "ea.py",
    "random_search.py", "run_experiments.py", "experiment_config.py",
)
REPLAY_SOURCES = ("controller.py", "evaluation.py")  # what replay.py checks

RNG_SCHEME = (
    "One numpy.random.default_rng(seed) per run. EA: ea.py draws the initial "
    "population, then parent choices and mutations, from it. Random search: "
    "successive sample_weights draws, so its first population_size samples equal "
    "the EA's initial population for the same seed. Python's random and NumPy's "
    "legacy global RNG are seeded with the same seed, but no operator uses them. "
    "MuJoCo is deterministic."
)


# ============================================================================ #
#  Experiments
# ============================================================================ #


@dataclass(frozen=True)
class Experiment:
    name: str  # "final", "pilot" or "smoke"
    ea: cfg.EAConfig
    sigmas: tuple[float, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sigmas or any(not (math.isfinite(s) and s > 0) for s in self.sigmas):
            raise ValueError(f"{self.name}: mutation strengths must be finite and positive.")
        if len({cfg.condition_name(s) for s in self.sigmas}) != len(self.sigmas):
            raise ValueError(f"{self.name}: two mutation strengths share one condition name.")

    @property
    def conditions(self) -> dict[str, float | None]:
        """Condition name -> mutation strength (None for random search)."""
        names: dict[str, float | None] = {cfg.condition_name(s): s for s in self.sigmas}
        names[cfg.RANDOM_SEARCH] = None
        return names

    def resolved(self) -> dict:
        """Everything that must be identical for all runs in one results folder."""
        settings = {
            "experiment": self.name,
            "evaluation_settings": asdict(cfg.EVALUATION),
            "genome_length": cfg.EVALUATION.controller.genome_length,
            "ea": {
                **asdict(self.ea),
                "max_evaluations": self.ea.max_evaluations,
                "plateau_stopping": self.ea.plateau_stopping,
            },
            "conditions": {
                name: {
                    "algorithm": "random_search" if sigma is None else "ea",
                    "mutation_strength": sigma,
                }
                for name, sigma in self.conditions.items()
            },
            "failure_fitness": cfg.FAILURE_FITNESS,
            "max_failed_evaluations": cfg.MAX_FAILED_EVALUATIONS,
            "rng": RNG_SCHEME,
        }
        # Normalise exactly as a JSON round trip would (tuples become lists).
        return json.loads(json.dumps(settings, allow_nan=False))


def get_experiment(name: str) -> Experiment:
    if name == "final":
        if cfg.FINAL_SIGMAS is None or cfg.FINAL_EA is None:
            raise SystemExit(
                "The final settings are not frozen yet: set FINAL_SIGMAS and FINAL_EA "
                "in experiment_config.py once the team has approved them."
            )
        return Experiment("final", cfg.FINAL_EA, tuple(cfg.FINAL_SIGMAS), tuple(cfg.FINAL_SEEDS))
    if name == "pilot":
        return Experiment("pilot", cfg.PILOT_EA, tuple(cfg.PILOT_SIGMAS), tuple(cfg.PILOT_SEEDS))
    if name == "smoke":
        return Experiment("smoke", cfg.SMOKE_EA, tuple(cfg.SMOKE_SIGMAS), tuple(cfg.SMOKE_SEEDS))
    raise ValueError(f"Unknown experiment '{name}'.")


# ============================================================================ #
#  Small helpers
# ============================================================================ #


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    """Strict JSON (no NaN/Infinity), written whole via a temporary file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    """Like assert, but never stripped by python -O."""
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest() -> str:
    """One hash over every Assignment 2 source file."""
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(f"{name}:{sha256(cfg.HERE / name)}\n".encode())
    return digest.hexdigest()


def git(*args: str) -> str | None:
    """Output of a read-only Git command, or None if Git is unavailable."""
    try:
        return subprocess.run(
            ["git", "-C", str(cfg.REPO_ROOT), *args],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def ariel_directory() -> Path:
    return Path(ariel.__file__).resolve().parent


def ariel_is_repository_copy() -> bool:
    return ariel_directory() == (cfg.REPO_ROOT / "src" / "ariel").resolve()


def provenance() -> dict:
    status = git("status", "--porcelain")
    framework_changes = git("status", "--porcelain", "--", "src/ariel")
    packages = {}
    for name in ("numpy", "mujoco", "sqlalchemy", "sqlmodel", "pydantic"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "git_commit": (git("rev-parse", "HEAD") or "").strip() or None,
        "git_status_porcelain": None if status is None else status.splitlines(),
        "framework_last_commit": (git("log", "-1", "--format=%H %cI", "--", "src/ariel") or "").strip() or None,
        "framework_unmodified": None if framework_changes is None else framework_changes.strip() == "",
        "ariel_imported_from": str(ariel_directory()),
        "ariel_is_repository_copy": ariel_is_repository_copy(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_sha256": {name: sha256(cfg.HERE / name) for name in SOURCE_FILES},
    }


# ============================================================================ #
#  Logged evaluation
# ============================================================================ #


class TooManyFailures(RuntimeError):
    """More simulations failed than experiment_config.MAX_FAILED_EVALUATIONS."""


def weights_key(weights: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(weights, dtype=np.float64).tobytes()).digest()


class LoggedEvaluator:
    """evaluate() wrapper used by both algorithms; every call counts as budget.

    Logs each call to evaluations.csv. A simulation that raises (including the
    evaluator's non-finite checks) is logged to failed_evaluations.jsonl and
    scored cfg.FAILURE_FITNESS instead of crashing the run. Successful results
    are remembered by their exact weights, so the best candidate's final
    position is known without simulating it again outside the budget.
    """

    FIELDS = (
        "evaluation", "generation", "status", "fitness", "final_x", "final_y", "final_z",
        "simulated_seconds", "elapsed_seconds", "error",
    )

    def __init__(self, folder: Path, ea_config: cfg.EAConfig) -> None:
        self.folder = folder
        self.ea_config = ea_config
        self.calls = 0
        self.failures = 0
        self.best_fitness = math.inf
        self.elapsed: list[float] = []
        self._results: dict[bytes, tuple[int, EvaluationResult]] = {}
        self._stream: IO[str] = (folder / "evaluations.csv").open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def generation_of(self, evaluation: int) -> int:
        """0 for the initial population, then one generation per batch of children."""
        size, children = self.ea_config.population_size, self.ea_config.number_of_children
        return 0 if evaluation <= size else (evaluation - size - 1) // children + 1

    def __call__(self, weights: np.ndarray, settings: EvaluationSettings) -> EvaluationResult:
        self.calls += 1
        index = self.calls
        flat = np.array(weights, dtype=np.float64)
        started = perf_counter()
        error_text = ""
        try:
            result = evaluate(flat, settings)
            if not math.isfinite(result.fitness):
                raise FloatingPointError(f"Non-finite fitness {result.fitness!r}.")
        except Exception as error:
            self.failures += 1
            error_text = f"{type(error).__name__}: {error}"
            self._log_failure(index, flat, error)
            nan3 = (math.nan, math.nan, math.nan)
            result = EvaluationResult(
                fitness=cfg.FAILURE_FITNESS, initial_position=nan3, final_position=nan3,
                simulated_seconds=math.nan, elapsed_seconds=perf_counter() - started,
            )
        else:
            self._results[weights_key(flat)] = (index, result)
            self.best_fitness = min(self.best_fitness, result.fitness)

        self.elapsed.append(result.elapsed_seconds)
        ok = not error_text
        self._writer.writerow({
            "evaluation": index,
            "generation": self.generation_of(index),
            "status": "ok" if ok else "failed",
            "fitness": result.fitness,
            "final_x": result.final_position[0] if ok else "",
            "final_y": result.final_position[1] if ok else "",
            "final_z": result.final_position[2] if ok else "",
            "simulated_seconds": result.simulated_seconds if ok else "",
            "elapsed_seconds": result.elapsed_seconds,
            "error": error_text,
        })
        if index % self.ea_config.number_of_children == 0:
            self._stream.flush()
            self._write_progress()
        if self.failures > cfg.MAX_FAILED_EVALUATIONS:
            raise TooManyFailures(
                f"{self.failures} failed simulations (limit {cfg.MAX_FAILED_EVALUATIONS}); "
                "see failed_evaluations.jsonl."
            )
        return result

    def _log_failure(self, index: int, weights: np.ndarray, error: Exception) -> None:
        record = {
            "evaluation": index,
            "generation": self.generation_of(index),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            # Non-finite values are stored as text so the file stays strict JSON.
            "weights": [float(x) if math.isfinite(x) else repr(float(x)) for x in weights],
        }
        with (self.folder / "failed_evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _write_progress(self) -> None:
        write_json(self.folder / "progress.json", {
            "evaluations": self.calls,
            "max_evaluations": self.ea_config.max_evaluations,
            "best_so_far": self.best_fitness if math.isfinite(self.best_fitness) else None,
            "failed_evaluations": self.failures,
            "updated_utc": utc_now(),
        })

    def lookup(self, weights: np.ndarray) -> tuple[int, EvaluationResult]:
        """The logged evaluation of exactly these weights."""
        try:
            return self._results[weights_key(np.asarray(weights, dtype=np.float64))]
        except KeyError:
            raise RuntimeError("The returned best weights were never evaluated successfully.") from None

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


# ============================================================================ #
#  One run (executed in its own process)
# ============================================================================ #


@dataclass
class Outcome:
    history: list[dict]
    best_fitness: float
    best_weights: np.ndarray
    evaluations: int
    generations_completed: int
    stopped_on_plateau: bool


def count_database(path: Path) -> tuple[int, float]:
    """(number of individuals, best fitness ever) in an ARIEL database."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows, best = connection.execute(
            "SELECT COUNT(*), MIN(fitness_) FROM individual WHERE requires_eval = 0"
        ).fetchone()
        total = connection.execute("SELECT COUNT(*) FROM individual").fetchone()[0]
    require(rows == total, f"{total - rows} individuals in the database were never evaluated.")
    return rows, best


def run_ea(experiment: Experiment, sigma: float, rng: np.random.Generator,
           logged: LoggedEvaluator, folder: Path) -> Outcome:
    ea = experiment.ea
    result = run_evolution(
        rng=rng,
        settings=cfg.EVALUATION,
        population_size=ea.population_size,
        number_of_children=ea.number_of_children,
        parent_fraction=ea.parent_fraction,
        mutation_strength=sigma,
        mutation_probability=ea.mutation_probability,
        plateau_patience=ea.plateau_patience,
        min_improvement=ea.min_improvement,
        max_generations=ea.max_generations,
        database_path=folder / "population.db",
        db_handling="halt",
        best_weights_path=None,  # *.npy is Git-ignored; weights go into best_candidate.json
        evaluator=logged,
    )
    history = [asdict(record) for record in result.history]
    db_rows, db_best = count_database(folder / "population.db")
    require(
        result.evaluations == logged.calls == db_rows == history[-1]["evaluations"],
        f"Evaluation counts disagree: ea.py {result.evaluations}, logged {logged.calls}, "
        f"database {db_rows}, history {history[-1]['evaluations']}.",
    )
    require(
        result.best_fitness == history[-1]["best_fitness"] == db_best,
        "The returned best, the last history row and the database disagree.",
    )
    return Outcome(
        history=history,
        best_fitness=result.best_fitness,
        best_weights=result.best_weights,
        evaluations=result.evaluations,
        generations_completed=result.generations_completed,
        stopped_on_plateau=result.stopped_on_plateau,
    )


def run_baseline(experiment: Experiment, rng: np.random.Generator,
                 logged: LoggedEvaluator) -> Outcome:
    result = run_random_search(
        rng=rng, settings=cfg.EVALUATION, ea_config=experiment.ea, evaluator=logged,
    )
    history = [asdict(record) for record in result.history]
    require(
        result.evaluations == logged.calls == experiment.ea.max_evaluations
        == history[-1]["evaluations"],
        "Random search did not use exactly the EA's evaluation cap.",
    )
    require(result.best_fitness == history[-1]["best_fitness"], "History and best disagree.")
    return Outcome(
        history=history,
        best_fitness=result.best_fitness,
        best_weights=result.best_weights,
        evaluations=result.evaluations,
        generations_completed=experiment.ea.max_generations,
        stopped_on_plateau=False,
    )


def run_worker(experiment: Experiment, condition: str, seed: int, folder: Path) -> None:
    if condition not in experiment.conditions:
        raise SystemExit(f"Unknown condition '{condition}' for the {experiment.name} experiment.")
    sigma = experiment.conditions[condition]
    algorithm = "random_search" if sigma is None else "ea"
    started_utc, started = utc_now(), perf_counter()

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    write_json(folder / "config.json", {
        "condition": condition,
        "algorithm": algorithm,
        "mutation_strength": sigma,
        "seed": seed,
        "started_utc": started_utc,
        "resolved": experiment.resolved(),
        "provenance": provenance(),
    })
    print(f"{experiment.name}: {condition}, seed {seed}, cap {experiment.ea.max_evaluations} "
          f"evaluations. Folder: {folder}", flush=True)

    logged = LoggedEvaluator(folder, experiment.ea)
    try:
        if sigma is None:
            outcome = run_baseline(experiment, rng, logged)
        else:
            outcome = run_ea(experiment, sigma, rng, logged, folder)
        best_index, best = logged.lookup(outcome.best_weights)
        require(best.fitness == outcome.best_fitness, "Logged score of the best candidate differs.")
    except BaseException as error:
        logged.close()
        write_json(folder / "failure.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "method": condition,
            "seed": seed,
            "evaluations_done": logged.calls,
            "failed_evaluations": logged.failures,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "utc": utc_now(),
        })
        raise
    logged.close()

    with (folder / "history.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcome.history[0]))
        writer.writeheader()
        writer.writerows(outcome.history)

    write_json(folder / "best_candidate.json", {
        "method": condition,
        "algorithm": algorithm,
        "mutation_strength": sigma,
        "seed": seed,
        "fitness": outcome.best_fitness,
        "weights": [float(x) for x in outcome.best_weights],
        "settings": asdict(cfg.EVALUATION),
        "initial_position": list(best.initial_position),
        "final_position": list(best.final_position),
        "found_at_evaluation": best_index,
        "search_evaluations": outcome.evaluations,
        "source_sha256": {name: sha256(cfg.HERE / name) for name in REPLAY_SOURCES},
    })

    (folder / "progress.json").unlink(missing_ok=True)  # running state only
    elapsed = np.asarray(logged.elapsed)
    initial_best = outcome.history[0]["best_fitness"]
    # Written last: its presence with status 'completed' marks a finished run.
    write_json(folder / "summary.json", {
        "status": "completed",
        "experiment": experiment.name,
        "method": condition,
        "algorithm": algorithm,
        "mutation_strength": sigma,
        "seed": seed,
        "evaluations": outcome.evaluations,
        "max_evaluations": experiment.ea.max_evaluations,
        "generations_completed": outcome.generations_completed,
        "stopped_on_plateau": outcome.stopped_on_plateau,
        "failed_evaluations": logged.failures,
        "initial_best": initial_best,
        "final_best": outcome.best_fitness,
        "improvement": initial_best - outcome.best_fitness,
        "best_found_at_evaluation": best_index,
        "evaluation_seconds_median": float(np.median(elapsed)),
        "evaluation_seconds_p95": float(np.percentile(elapsed, 95)),
        "wall_seconds": perf_counter() - started,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
    })
    print(f"Completed: best {outcome.best_fitness:.4f} after {outcome.evaluations} evaluations, "
          f"{logged.failures} failed.", flush=True)


# ============================================================================ #
#  Launcher
# ============================================================================ #


@dataclass
class Task:
    condition: str
    seed: int
    final: Path
    staging: Path
    status: str = "pending"
    process: subprocess.Popen | None = None
    log: IO[str] | None = None
    started: float = 0.0
    notes: list[str] = field(default_factory=list)


def is_complete(folder: Path) -> bool:
    summary = read_json(folder / "summary.json")
    return summary is not None and summary.get("status") == "completed"


def staging_root(root: Path) -> Path:
    """A staging folder on the same filesystem as root, so the final move is atomic."""
    try:
        relative = root.resolve().relative_to(cfg.REPO_ROOT)
    except ValueError:
        raise SystemExit(f"The results folder must be inside the repository: {root}") from None
    return cfg.STAGING_ROOT / "__".join(relative.parts)


def claim_root(root: Path, experiment: Experiment) -> None:
    """Refuse to mix runs made with different settings in one results folder."""
    path = root / "experiment.json"
    expected = experiment.resolved()
    existing = read_json(path)
    if existing is None:
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, {"resolved": expected, "created_utc": utc_now()})
    elif existing.get("resolved") != expected:
        raise SystemExit(
            f"{root} already holds runs made with different settings (see its "
            "experiment.json). Use a new results folder instead of mixing settings."
        )


def promote(task: Task, experiment: Experiment) -> None:
    """Move a completed staging folder into the results folder."""
    config = read_json(task.staging / "config.json")
    require(config is not None and config.get("resolved") == experiment.resolved(),
            "This run's settings differ from the experiment's; it was not moved.")
    require(not task.final.exists(), f"{task.final} already exists; nothing was overwritten.")
    task.final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(task.staging, task.final)


def acquire_locks(staging: Path, conditions: list[str]) -> list[Path]:
    """One launcher per (results folder, condition) at a time."""
    staging.mkdir(parents=True, exist_ok=True)
    locks: list[Path] = []
    for condition in conditions:
        path = staging / f"{condition}.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            release_locks(locks)
            raise SystemExit(
                f"Another launcher is already running '{condition}' for this results folder "
                f"({path}). If none is, delete that file and try again."
            ) from None
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"process {os.getpid()}, since {utc_now()}\n")
        locks.append(path)
    return locks


def release_locks(locks: list[Path]) -> None:
    for path in locks:
        path.unlink(missing_ok=True)


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # One thread per process: parallelism comes from processes only.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def start(task: Task, experiment: Experiment, environment: dict[str, str], digest: str) -> None:
    task.staging.mkdir(parents=True)
    task.log = (task.staging / "run_log.txt").open("x", encoding="utf-8")
    task.process = subprocess.Popen(
        [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--experiment", experiment.name, "--condition", task.condition,
            "--seed", str(task.seed), "--staging", str(task.staging),
            "--source-digest", digest,
        ],
        stdout=task.log, stderr=subprocess.STDOUT, env=environment, cwd=cfg.REPO_ROOT,
    )
    task.started = perf_counter()
    task.status = "running"


def describe(folder: Path) -> str:
    summary = read_json(folder / "summary.json") or {}
    if not summary:
        return ""
    return (f"best {summary['final_best']:.4f}, {summary['evaluations']} evaluations, "
            f"{summary['failed_evaluations']} failed, {summary['wall_seconds'] / 60:.1f} min")


def launch(experiment: Experiment, conditions: list[str], seeds: list[int],
           root: Path, workers: int | None) -> int:
    staging = staging_root(root)  # checks root is inside the repository, before creating it
    claim_root(root, experiment)
    locks = acquire_locks(staging, conditions)
    try:
        return run_tasks(experiment, conditions, seeds, root, staging, workers, source_digest())
    finally:
        release_locks(locks)


def run_tasks(experiment: Experiment, conditions: list[str], seeds: list[int],
              root: Path, staging: Path, workers: int | None, digest: str) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tasks: list[Task] = []
    for condition in conditions:
        for seed in seeds:
            task = Task(condition, seed, root / condition / f"seed_{seed}",
                        staging / condition / f"seed_{seed}")
            tasks.append(task)
            if is_complete(task.final):
                task.status = "skipped (already complete)"
            elif task.final.exists():
                raise SystemExit(f"{task.final} exists but is not a completed run; inspect it by hand.")
            elif is_complete(task.staging):
                # Finished earlier, but the launcher stopped before moving it.
                try:
                    promote(task, experiment)
                    task.status = "completed (moved from staging)"
                except (RuntimeError, OSError) as error:
                    task.status = f"FAILED to move: {error}"
            elif task.staging.exists():
                aside = task.staging.with_name(f"{task.staging.name}.aborted-{stamp}")
                task.staging.rename(aside)
                task.notes.append(f"unfinished earlier attempt kept at {aside}")

    pending = [t for t in tasks if t.status == "pending"]
    workers = workers or min(len(pending), max(1, (os.cpu_count() or 2) - 2)) or 1
    print(f"Experiment '{experiment.name}' -> {root}")
    print(f"Staging: {staging}")
    print(f"{len(pending)} runs to do, {len(tasks) - len(pending)} already complete; "
          f"{workers} parallel processes; cap {experiment.ea.max_evaluations} evaluations per run.")
    for task in tasks:
        for note in task.notes:
            print(f"  {task.condition} seed {task.seed}: {note}")

    environment = child_environment()
    running: list[Task] = []
    last_heartbeat = perf_counter()
    try:
        while pending or running:
            while pending and len(running) < workers:
                task = pending.pop(0)
                start(task, experiment, environment, digest)
                running.append(task)
            time.sleep(0.5)
            for task in list(running):
                if task.process.poll() is None:
                    continue
                running.remove(task)
                task.log.close()
                if task.process.returncode == 0 and is_complete(task.staging):
                    try:
                        promote(task, experiment)
                        task.status = "completed"
                    except (RuntimeError, OSError) as error:
                        task.status = f"FAILED to move: {error}"
                else:
                    task.status = f"FAILED (exit code {task.process.returncode})"
                folder = task.final if task.status == "completed" else task.staging
                print(f"[{sum(t.status != 'running' and t.status != 'pending' for t in tasks)}/"
                      f"{len(tasks)}] {task.condition} seed {task.seed}: {task.status}. "
                      f"{describe(folder)} ({folder})", flush=True)
            if running and perf_counter() - last_heartbeat > 300:
                last_heartbeat = perf_counter()
                for task in running:
                    progress = read_json(task.staging / "progress.json") or {}
                    print(f"    running {task.condition} seed {task.seed}: "
                          f"{progress.get('evaluations', 0)}/{experiment.ea.max_evaluations} "
                          f"evaluations, best so far {progress.get('best_so_far')}", flush=True)
    except KeyboardInterrupt:
        # Ctrl-C in a terminal also reaches the workers; give them a moment to
        # record failure.json before stopping any that are still running.
        for task in running:
            try:
                task.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                task.process.terminate()
                task.process.wait()
            task.log.close()
            task.status = "interrupted"
        print("\nInterrupted. Unfinished runs stay in staging; the same command restarts them.")
        return 130

    print("\nSummary")
    failed = 0
    for task in tasks:
        ok = task.status.startswith(("completed", "skipped"))
        failed += not ok
        folder = task.final if ok else task.staging
        print(f"  {task.condition:<20} seed {task.seed:<5} {task.status:<32} {describe(folder)}")
    if failed:
        print(f"\n{failed} run(s) failed; see failure.json and run_log.txt in their staging folders.")
    return 1 if failed else 0


# ============================================================================ #
#  Checks, listing and benchmark
# ============================================================================ #


def check_code_state(require_committed_sources: bool) -> None:
    if not ariel_is_repository_copy():
        raise SystemExit(
            f"ariel is imported from {ariel_directory()}, not from this repository's src/ariel; "
            "the recorded framework revision would be wrong. Fix the environment first."
        )
    framework_changes = git("status", "--porcelain", "--", "src/ariel")
    if framework_changes is None:
        raise SystemExit("Git is unavailable, so src/ariel cannot be confirmed unmodified.")
    if framework_changes.strip():
        raise SystemExit("src/ariel has local changes; the framework must not be modified:\n"
                         + framework_changes)
    if require_committed_sources:
        sources = [f"assignments/assignment_2/{name}" for name in SOURCE_FILES]
        changes = git("status", "--porcelain", "--", *sources) or ""
        if changes.strip():
            raise SystemExit(
                "Commit the Assignment 2 sources first, so each run's recorded Git commit "
                "identifies its code exactly (or pass --allow-dirty):\n" + changes
            )


def list_experiments() -> None:
    for name in ("final", "pilot", "smoke"):
        try:
            experiment = get_experiment(name)
        except SystemExit as reason:
            print(f"{name}: not set. {reason}\n")
            continue
        details = {"seeds": list(experiment.seeds), **experiment.resolved()}
        print(f"{name}:\n{json.dumps(details, indent=2)}\n")


def benchmark(count: int) -> None:
    settings = cfg.EVALUATION
    rng = np.random.default_rng(0)
    print(f"Timing {count} sequential evaluations of random controllers "
          f"({settings.duration:g} simulated seconds each)...")
    times = []
    for index in range(count):
        result = evaluate(sample_weights(rng, settings.controller), settings)
        times.append(result.elapsed_seconds)
        print(f"  {index + 1:3d}: {result.elapsed_seconds:.3f} s, fitness {result.fitness:.4f}")
    median = float(np.median(times))
    print(f"Median {median:.3f} s, mean {np.mean(times):.3f} s, max {np.max(times):.3f} s.")

    still = evaluate(np.zeros(settings.controller.genome_length), settings)
    print(f"Reference: all-zero weights (every hinge held at 0, robot stands still) "
          f"score {still.fitness:.4f} m.")

    for name in ("pilot", "final"):
        try:
            experiment = get_experiment(name)
        except SystemExit:
            continue
        runs = len(experiment.conditions) * len(experiment.seeds)
        total = runs * experiment.ea.max_evaluations
        print(f"{name}: {runs} runs x {experiment.ea.max_evaluations} evaluations = {total}; "
              f"{total * median / 3600:.2f} CPU-hours at this median (sequential). Wall time is "
              f"about that divided by the number of workers, plus slowdown under parallel load.")


# ============================================================================ #
#  Command line
# ============================================================================ #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Tiny pipeline test into a new __data__ folder.")
    mode.add_argument("--pilot", action="store_true", help="Mutation-strength screen.")
    mode.add_argument("--list", action="store_true", help="Print the resolved settings and exit.")
    mode.add_argument("--benchmark", type=int, metavar="N", help="Time N evaluations and exit.")
    parser.add_argument("--condition", nargs="+", metavar="NAME",
                        help="Condition(s) to run, or 'all'. Required for final runs.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: the experiment's seeds.")
    parser.add_argument("--workers", type=int, help="Parallel processes (one run each).")
    parser.add_argument("--root", type=Path, help="Smoke/pilot only: results folder to create or resume.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Final runs only: allow uncommitted Assignment 2 sources.")
    # Internal: one run inside a child process.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--experiment", choices=("final", "pilot", "smoke"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--staging", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-digest", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress stays visible when redirected

    if args.worker:
        # All runs of one batch must use the code the batch started with.
        if args.source_digest != source_digest():
            raise SystemExit("Assignment 2 source files changed after this batch started; "
                             "this run was not started.")
        run_worker(get_experiment(args.experiment), args.condition[0], args.seed, args.staging)
        return
    if args.list:
        list_experiments()
        return
    if args.benchmark is not None:
        if args.benchmark < 1:
            parser.error("--benchmark needs a positive count.")
        benchmark(args.benchmark)
        return

    name = "smoke" if args.smoke else "pilot" if args.pilot else "final"
    if name == "final" and not args.condition:
        parser.error("Final runs need --condition NAME (or --condition all).")
    if name == "final" and args.root:
        parser.error("Final runs always go to experiment_config.FINAL_ROOT.")
    experiment = get_experiment(name)

    available = list(experiment.conditions)
    if not args.condition or args.condition == ["all"]:
        conditions = available
    else:
        unknown = [c for c in args.condition if c not in available]
        if unknown:
            parser.error(f"Unknown condition(s) {unknown}; available: {available}.")
        conditions = list(dict.fromkeys(args.condition))
    seeds = list(args.seeds) if args.seeds else list(experiment.seeds)
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("Seeds must be distinct, non-negative integers.")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1.")

    check_code_state(require_committed_sources=name == "final" and not args.allow_dirty)

    if name == "final":
        root = cfg.FINAL_ROOT
    elif name == "pilot":
        root = args.root or cfg.PILOT_ROOT
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = args.root or cfg.SMOKE_ROOT / stamp
    sys.exit(launch(experiment, conditions, seeds, root.resolve(), args.workers))


if __name__ == "__main__":
    main()
