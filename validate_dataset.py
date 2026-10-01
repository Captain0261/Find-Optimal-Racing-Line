"""Run complete planning on the supplied dataset and retain validation results."""
import argparse
import json
import time
from pathlib import Path
from racing_line import load_track, plan_racing_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root.parent / "fsd_racetrack_dataset-main" / "dataset")
    parser.add_argument("--output", type=Path, default=root / "results" / "dataset_validation.json")
    parser.add_argument("--objective", choices=("curvature", "lap_time"), default="lap_time")
    parser.add_argument("--controls", type=int, default=48)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--tracks", type=int, nargs="+", help="Only validate these track IDs")
    parser.add_argument("--max-iterations", type=int, default=200)
    args = parser.parse_args()
    reports = []
    for file in sorted(args.data_dir.glob("boundaries_*.yaml")):
        track_id = int(file.stem.split("_")[-1])
        if args.tracks and track_id not in args.tracks:
            continue
        start = time.perf_counter()
        try:
            plan = plan_racing_line(load_track(args.data_dir, track_id), objective=args.objective,
                                   control_points=args.controls, samples=args.samples,
                                   max_iterations=args.max_iterations)
            report = {"track": track_id, "valid": True, **plan.metrics, "solver": plan.solver}
        except (ValueError, RuntimeError) as exc:
            report = {"track": track_id, "valid": False, "error": str(exc)}
        report["runtime_s"] = time.perf_counter() - start
        reports.append(report)
        print(json.dumps(report), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if not reports or not all(r["valid"] and all(s["success"] for s in r["solver"]) for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
