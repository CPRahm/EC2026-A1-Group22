"""

Run this file to compare a mutation-only EA with equal-budget random search.
Both use the existing controller/evaluator and ARIEL's persistence engine.
This is a development check, not the final multi-configuration experiment.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import ariel
import numpy as np
from ariel.ec import EA, EAOperation, Individual, Population

# No postponed annotations: this ARIEL version inspects Population annotations.
if __package__:
    from .controller import ControllerSettings, sample_weights
    from .evaluation import EvaluationSettings, evaluate
else:
    from controller import ControllerSettings, sample_weights
    from evaluation import EvaluationSettings, evaluate


def save_json(path: Path, data: dict) -> None:
    """Reject NaN/Infinity rather than writing ambiguous experiment records."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(start: Path) -> dict:
    """Best-effort metadata only; never change Git configuration or files."""
    root = next((p for p in (start, *start.parents) if (p / ".git").exists()), None)
    executable = shutil.which("git")
    if root is None or executable is None:
        return {"available": False, "reason": "Git executable or checkout not found"}
    try:
        commit = subprocess.check_output(
            [executable, "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
        status = subprocess.check_output(
            [executable, "-C", str(root), "status", "--porcelain"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip()
        return {"available": True, "root": str(root), "commit": commit, "status": status}
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error)}


def provenance(root: Path) -> dict:
    sources = root / "source_snapshot"
    sources.mkdir()
    hashes = {}
    for name in ("controller.py", "evaluation.py", "learning_pilot.py"):
        path = Path(__file__).resolve().with_name(name)
        hashes[name] = file_hash(path)
        shutil.copyfile(path, sources / name)
    packages = {}
    for name in ("numpy", "mujoco", "ariel", "sqlmodel"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "distribution version unavailable"
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "source_sha256": hashes,
        "framework_git": git_state(Path(ariel.__file__).resolve().parent),
    }


def run_method(method: str, seed: int, args: argparse.Namespace,
               settings: EvaluationSettings, root: Path, source_hashes: dict) -> dict:
    """One seed/method, with independent initialization and proposal RNG streams."""
    folder = root / f"seed_{seed}" / method
    folder.mkdir(parents=True, exist_ok=False)
    initial_seed, mutation_seed, random_seed = np.random.SeedSequence(seed).spawn(3)
    initial_rng = np.random.default_rng(initial_seed)
    proposal_rng = np.random.default_rng(mutation_seed if method == "ea" else random_seed)
    generation, count = 0, 0
    durations = []
    history = []
    engine = None
    started = perf_counter()
    budget = args.population * (args.generations + 1)
    row_fields = [
        "evaluation", "generation", "fitness", "parent_fitness", "improved_parent",
        "final_x", "final_y", "final_z", "elapsed_seconds", "simulated_seconds",
    ]

    def new_individual(weights, parent=None) -> Individual:
        individual = Individual()
        individual.genotype = np.asarray(weights, dtype=float).tolist()
        if parent is not None:
            individual.tags = {"parent_id": parent.id, "parent_fitness": parent.fitness}
        return individual

    def ranked(population):
        # Explicit tie-breaker avoids relying on database row ordering.
        return sorted(population.alive, key=lambda ind: (ind.fitness, ind.tags["evaluation"]))

    with (folder / "evaluations.csv").open("x", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=row_fields)
        writer.writeheader()

        def score_candidates(population: Population) -> Population:
            nonlocal count
            for individual in population.unevaluated:
                if count >= budget:
                    raise RuntimeError("Pilot attempted to exceed its evaluation budget.")
                count += 1
                try:
                    result = evaluate(individual.genotype, settings)
                except Exception as error:
                    save_json(folder / "failure.json", {
                        "status": "failed", "method": method, "seed": seed,
                        "evaluation": count, "generation": generation,
                        "weights": individual.genotype,
                        "settings": asdict(settings), "error": repr(error),
                    })
                    raise
                individual.fitness = result.fitness
                individual.tags = {
                    "evaluation": count, "generation": generation,
                    "final_position": list(result.final_position),
                    "elapsed_seconds": result.elapsed_seconds,
                }
                parent_fitness = individual.tags.get("parent_fitness")
                writer.writerow({
                    "evaluation": count, "generation": generation,
                    "fitness": result.fitness,
                    "parent_fitness": "" if parent_fitness is None else parent_fitness,
                    "improved_parent": "" if parent_fitness is None else int(result.fitness < parent_fitness),
                    "final_x": result.final_position[0], "final_y": result.final_position[1],
                    "final_z": result.final_position[2], "elapsed_seconds": result.elapsed_seconds,
                    "simulated_seconds": result.simulated_seconds,
                })
                log.flush()
                durations.append(result.elapsed_seconds)
            return population

        def propose_candidates(population: Population) -> Population:
            if method == "ea":
                parents = ranked(population)[:max(1, args.population // 2)]
                for _ in range(args.population):
                    parent = parents[int(proposal_rng.integers(len(parents)))]
                    # Fresh array: parents are never mutated in place.
                    child = np.asarray(parent.genotype) + proposal_rng.normal(
                        0.0, args.sigma, size=settings.controller.genome_length,
                    )
                    population.append(new_individual(child, parent))
            else:
                # Selection never affects these independent random proposals.
                for _ in range(args.population):
                    population.append(new_individual(sample_weights(proposal_rng, settings.controller)))
            return population

        def keep_best(population: Population) -> Population:
            for index, individual in enumerate(ranked(population)):
                individual.alive = index < args.population
            return population

        def record_generation() -> None:
            # ARIEL commits can expire objects; obtain a fresh database snapshot.
            engine.fetch_population()
            population = ranked(engine.population)
            if len(population) != args.population:
                raise RuntimeError("Unexpected survivor count.")
            row = {
                "generation": generation, "evaluations": count,
                "best_fitness": population[0].fitness,
                "survivor_mean_fitness": float(np.mean([ind.fitness for ind in population])),
            }
            if history and row["best_fitness"] > history[-1]["best_fitness"]:
                raise RuntimeError("Elitism failed: best fitness became worse.")
            history.append(row)
            print(f"seed={seed} {method:6s} generation={generation:2d} "
                  f"evaluations={count:4d}/{budget} best={row['best_fitness']:.4f}", flush=True)
            save_json(folder / "progress.json", {"status": "running", "history": history})

        try:
            # Paired EA/RS runs start with identical genotypes, separately scored.
            initial = Population([
                new_individual(sample_weights(initial_rng, settings.controller))
                for _ in range(args.population)
            ])
            score_candidates(initial)
            engine = EA(
                initial,
                [EAOperation(propose_candidates), EAOperation(score_candidates), EAOperation(keep_best)],
                num_steps=args.generations, first_generation_id=0,
                is_maximisation=False, quiet=True,
                db_file_path=folder / "population.db", db_handling="halt",
            )
            record_generation()
            for generation in range(1, args.generations + 1):
                engine.step()
                record_generation()
            if count != budget:
                raise RuntimeError("Unexpected final evaluation count.")

            best = engine.get_solution("best")
            candidate = {
                "method": method, "seed": seed, "fitness": best.fitness,
                "weights": list(best.genotype), "settings": asdict(settings),
                "final_position": best.tags["final_position"],
                "search_evaluations": count, "source_sha256": source_hashes,
            }
            save_json(folder / "best_candidate.json", candidate)
            # One additional diagnostic simulation, reported outside the search budget.
            loaded = json.loads((folder / "best_candidate.json").read_text(encoding="utf-8"))
            recheck = evaluate(loaded["weights"], settings)
            if not np.isclose(recheck.fitness, best.fitness, rtol=0, atol=1e-10):
                raise RuntimeError("Saved candidate did not reproduce its original score.")
            if not np.allclose(recheck.final_position, best.tags["final_position"], rtol=0, atol=1e-10):
                raise RuntimeError("Saved candidate did not reproduce its final position.")

            summary = {
                "status": "completed", "method": method, "seed": seed,
                "initial_best": history[0]["best_fitness"], "final_best": best.fitness,
                "improvement_from_initial": history[0]["best_fitness"] - best.fitness,
                "search_evaluations": count, "verification_evaluations": 1,
                "median_evaluation_seconds": float(np.median(durations)),
                "p95_evaluation_seconds": float(np.percentile(durations, 95)),
                "wall_seconds": perf_counter() - started,
                "saved_candidate_reproduced": True,
                "best_candidate": str((folder / "best_candidate.json").resolve()),
            }
            with (folder / "history.csv").open("x", newline="", encoding="utf-8") as stream:
                table = csv.DictWriter(stream, fieldnames=list(history[0]))
                table.writeheader()
                table.writerows(history)
            save_json(folder / "summary.json", summary)
            save_json(folder / "progress.json", {"status": "completed", "history": history})
            return summary
        finally:
            if engine is not None:
                engine.engine.dispose()


def replay(path: Path, show_viewer: bool) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    for name in ("controller.py", "evaluation.py"):
        if file_hash(Path(__file__).resolve().with_name(name)) != saved["source_sha256"][name]:
            raise ValueError(f"{name} changed since this candidate was saved. Use the recorded version.")
    raw = dict(saved["settings"])
    raw["controller"] = ControllerSettings(**raw["controller"])
    raw["spawn_position"] = tuple(raw["spawn_position"])
    raw["target_position"] = tuple(raw["target_position"])
    settings = EvaluationSettings(**raw)
    result = evaluate(saved["weights"], settings)
    if not np.isclose(result.fitness, saved["fitness"], rtol=0, atol=1e-10):
        raise RuntimeError(f"Replay differs: saved={saved['fitness']}, current={result.fitness}.")
    if not np.allclose(result.final_position, saved["final_position"], rtol=0, atol=1e-10):
        raise RuntimeError("Replay final position differs from the saved candidate.")
    print(f"Saved candidate reproduced: seed={saved['seed']} method={saved['method']} "
          f"fitness={result.fitness:.4f}", flush=True)
    if show_viewer:
        print("Opening the saved candidate for 15 simulated seconds. Do not perturb the robot.", flush=True)
        visual = evaluate(saved["weights"], settings, show_viewer=True)
        if not np.isclose(visual.fitness, result.fitness, rtol=0, atol=1e-10):
            raise RuntimeError("Visual replay differs from the headless score.")
        print(f"Visual replay completed: fitness={visual.fitness:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="New output folder; existing folders are refused.")
    parser.add_argument("--replay", type=Path, help="Re-evaluate a saved best_candidate.json.")
    parser.add_argument("--view", action="store_true", help="With --replay, also show its movement.")
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.view)
        return
    if args.view:
        parser.error("--view requires --replay.")
    if args.population < 2 or args.generations < 1:
        parser.error("Use population >= 2 and generations >= 1.")
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("sigma must be finite and positive.")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct, non-negative integers.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = (args.output or Path.cwd() / "__data__" / "assignment_2" / f"pilot_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = EvaluationSettings()
    metadata = provenance(root)
    budget = args.population * (args.generations + 1)
    metadata.update({
        "purpose": "development pilot; not the final assignment experiment",
        "status": "running", "started_utc": stamp,
        "evaluation_settings": asdict(settings), "genome_length": settings.controller.genome_length,
        "seeds": args.seeds, "population": args.population, "offspring_per_generation": args.population,
        "generations": args.generations, "mutation_sigma": args.sigma,
        "mutated_parameters": "all weights and biases; additive independent Gaussian noise",
        "parent_selection": "uniform among best floor(population/2), minimum one",
        "survival": "best population candidates from parents plus offspring, minimizing fitness",
        "random_search": "independent initial-distribution samples; shared initial population per seed",
        "rng": "NumPy SeedSequence(seed).spawn(3): initialization, EA proposals, RS proposals",
        "search_budget_per_method_seed": budget,
        "verification_evaluations_per_method_seed": 1,
        "failure_policy": "save failing candidate and abort; no silent retries or dropped failures",
    })
    save_json(root / "config.json", metadata)
    print(f"Output: {root}", flush=True)
    print(f"Pilot: {len(args.seeds)} seeds x 2 methods x {budget} search evaluations, "
          f"plus {2 * len(args.seeds)} saved-candidate rechecks.", flush=True)
    summaries = []
    try:
        for seed in args.seeds:
            for method in ("ea", "random"):
                summaries.append(run_method(method, seed, args, settings, root, metadata["source_sha256"]))
                save_json(root / "summary.json", {"status": "running", "runs": summaries})
        pairs = []
        for seed in args.seeds:
            ea, rs = [row for row in summaries if row["seed"] == seed]
            if not np.isclose(ea["initial_best"], rs["initial_best"], rtol=0, atol=1e-10):
                raise RuntimeError("Paired initial populations produced different best scores.")
            pairs.append({"seed": seed, "initial_best": ea["initial_best"],
                          "ea_final": ea["final_best"], "random_final": rs["final_best"],
                          "ea_better_than_random": ea["final_best"] < rs["final_best"]})
        best_ea = min((row for row in summaries if row["method"] == "ea"), key=lambda row: row["final_best"])
        shutil.copyfile(best_ea["best_candidate"], root / "best_ea.json")
        save_json(root / "summary.json", {"status": "completed", "pairs": pairs, "runs": summaries,
                                          "interpretation": "Preliminary development evidence; no significance claim."})
        with (root / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
            table = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            table.writeheader()
            table.writerows(pairs)
        metadata["status"] = "completed"
        save_json(root / "config.json", metadata)
        print("\nPilot completed. Lower fitness is better.", flush=True)
        for pair in pairs:
            print(f"seed={pair['seed']}: initial={pair['initial_best']:.4f}, "
                  f"EA={pair['ea_final']:.4f}, random={pair['random_final']:.4f}")
        print(f"Best EA candidate for replay: {root / 'best_ea.json'}")
        print("Development check only: do not treat these runs as the final experiment.")
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=repr(error))
        save_json(root / "config.json", metadata)
        save_json(root / "summary.json", {"status": metadata["status"], "runs": summaries,
                                          "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
