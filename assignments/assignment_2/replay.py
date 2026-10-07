import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .controller import ControllerSettings
    from .evaluation import EvaluationSettings, evaluate
else:
    from controller import ControllerSettings
    from evaluation import EvaluationSettings, evaluate

HERE = Path(__file__).resolve().parent
CHECKED_SOURCES = ("controller.py", "evaluation.py")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings_from_json(raw_settings: dict) -> EvaluationSettings:
    """Rebuild the frozen settings object exactly as the run stored it."""
    raw = dict(raw_settings)
    raw["controller"] = ControllerSettings(**raw["controller"])
    raw["spawn_position"] = tuple(raw["spawn_position"])
    raw["target_position"] = tuple(raw["target_position"])
    return EvaluationSettings(**raw)


def changed_sources(saved: dict) -> list[str]:
    """Names of evaluator files that differ from the version used for the run."""
    recorded = saved.get("source_sha256") or {}
    return [
        name for name in CHECKED_SOURCES
        if name in recorded and file_hash(HERE / name) != recorded[name]
    ]


def replay_one(path: Path, tolerance: float, ignore_source_changes: bool) -> dict:
    saved = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "method": saved.get("method", "?"), "seed": saved.get("seed", "?"),
        "saved_fitness": saved["fitness"], "replayed_fitness": "",
        "difference": "", "status": "", "file": str(path),
    }

    changed = changed_sources(saved)
    if changed and not ignore_source_changes:
        row["status"] = f"SKIPPED: {', '.join(changed)} changed since this run"
        return row

    result = evaluate(saved["weights"], settings_from_json(saved["settings"]))
    difference = abs(result.fitness - saved["fitness"])
    same_position = np.allclose(
        result.final_position, saved["final_position"], rtol=0, atol=tolerance
    )
    row["replayed_fitness"] = result.fitness
    row["difference"] = difference
    if difference <= tolerance and same_position:
        row["status"] = "OK"
    elif difference <= tolerance:
        row["status"] = "FAIL: same score, different final position"
    else:
        row["status"] = "FAIL: score differs"
    return row


def view(path: Path) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    settings = settings_from_json(saved["settings"])
    print(f"Showing {saved.get('method', '?')} seed {saved.get('seed', '?')} "
          f"(saved fitness {saved['fitness']:.4f}) for {settings.duration:g} simulated seconds. "
          "Do not touch the robot in the viewer.", flush=True)
    result = evaluate(saved["weights"], settings, show_viewer=True)
    print(f"Viewer run finished: fitness {result.fitness:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="?", help="Folder to search for best_candidate.json files.")
    parser.add_argument("--tolerance", type=float, default=1e-10,
                        help="Allowed fitness difference. 1e-10 on the same machine; "
                             "loosen (e.g. 1e-6) for runs made on another computer.")
    parser.add_argument("--ignore-source-changes", action="store_true",
                        help="Replay even if controller.py/evaluation.py changed since the run.")
    parser.add_argument("--view", type=Path, help="Open the viewer for one best_candidate.json.")
    args = parser.parse_args()

    if args.view:
        view(args.view)
        return
    if args.results is None:
        parser.error("Give a results folder, or --view with one best_candidate.json.")

    root = args.results.resolve()
    paths = sorted(p for p in root.rglob("best_candidate.json"))
    if not paths:
        sys.exit(f"No best_candidate.json files found under {root}.")

    print(f"Replaying {len(paths)} saved controllers (tolerance {args.tolerance:g})...\n", flush=True)
    rows = []
    for path in paths:
        row = replay_one(path, args.tolerance, args.ignore_source_changes)
        rows.append(row)
        detail = (f"replayed={row['replayed_fitness']:.6f}  diff={row['difference']:.1e}"
                  if row["replayed_fitness"] != "" else "")
        print(f"  {row['method']:<20} seed {row['seed']!s:<4} saved={row['saved_fitness']:.6f}  "
              f"{detail}  {row['status']}", flush=True)

    out = root / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "replay_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row["status"] == "OK" for row in rows)
    print(f"\n{passed}/{len(rows)} reproduced. Report: {out / 'replay_report.csv'}")
    if passed != len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()