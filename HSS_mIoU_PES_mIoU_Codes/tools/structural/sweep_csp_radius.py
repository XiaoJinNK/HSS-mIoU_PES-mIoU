import argparse
import csv
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))

from structural_metrics import evaluate_dataset_from_dirs


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated list of non-negative integers."""
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError("All values must be >= 0")
        values.append(value)

    if not values:
        raise ValueError("At least one value must be provided")
    return values


def _to_serializable(value: Any) -> Any:
    """Convert numpy-heavy outputs into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one flat record into a CSV file, creating the header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_radius_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    """Run CSP evaluation across multiple radius/boundary combinations and save records."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or "csp_radius_sweep"
    sweep_dir = Path(args.output_dir) / f"{timestamp}_{experiment_name}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "pred_dir": str(Path(args.pred_dir).resolve()),
        "gt_dir": str(Path(args.gt_dir).resolve()),
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "radii": args.radii,
        "boundary_widths": args.boundary_widths,
        "kernel_type": args.kernel_type,
        "preview_pairs": args.preview_pairs,
        "num_workers": args.num_workers,
        "notes": args.notes,
    }

    summary_rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []
    history_csv = Path(args.output_dir) / "history.csv"

    combinations = list(itertools.product(args.radii, args.boundary_widths))

    print(f"[Sweep] output directory: {sweep_dir}")
    print(f"[Sweep] radii to evaluate: {args.radii}")
    print(f"[Sweep] boundary_widths to evaluate: {args.boundary_widths}")
    print(f"[Sweep] total combinations: {len(combinations)}")

    for radius, boundary_width in combinations:
        print(f"[Sweep] evaluating max_radius={radius}, boundary_width={boundary_width} ...")
        start_time = time.perf_counter()
        metrics = evaluate_dataset_from_dirs(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            max_radius=radius,
            boundary_width=boundary_width,
            kernel_type=args.kernel_type,
            preview_pairs=args.preview_pairs,
            num_workers=args.num_workers,
        )
        elapsed_sec = time.perf_counter() - start_time

        row = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "radius": radius,
            "boundary_width": boundary_width,
            "kernel_type": args.kernel_type,
            "num_classes": args.num_classes,
            "ignore_index": args.ignore_index,
            "num_workers": args.num_workers,
            "num_images": int(metrics["num_images"]),
            "csp_image_mean": float(metrics["csp_image_mean"]),
            "persist_image_mean": float(metrics["persist_image_mean"]),
            "edge_image_mean": float(metrics["edge_image_mean"]),
            "csp_class_mean": float(metrics["csp_class_mean"]),
            "persist_class_mean": float(metrics["persist_class_mean"]),
            "edge_class_mean": float(metrics["edge_class_mean"]),
            "miou": float(metrics["miou"]),
            "elapsed_sec": elapsed_sec,
            "pred_dir": str(Path(args.pred_dir).resolve()),
            "gt_dir": str(Path(args.gt_dir).resolve()),
            "notes": args.notes,
        }
        summary_rows.append(row)
        _append_csv_row(history_csv, row)

        detailed_record = {
            "config": config,
            "radius": radius,
            "boundary_width": boundary_width,
            "elapsed_sec": elapsed_sec,
            "metrics": _to_serializable(metrics),
        }
        detailed_results.append(detailed_record)

        detail_path = sweep_dir / f"radius_{radius:03d}_bw_{boundary_width:03d}.json"
        detail_path.write_text(json.dumps(detailed_record, indent=2, ensure_ascii=False), encoding="utf-8")

        print(
            "[Sweep] "
            f"r={radius} | bw={boundary_width} | mIoU={row['miou']:.6f} | "
            f"CSP(image_mean)={row['csp_image_mean']:.6f} | "
            f"Persist(image_mean)={row['persist_image_mean']:.6f} | "
            f"Edge(image_mean)={row['edge_image_mean']:.6f} | "
            f"time={elapsed_sec:.2f}s"
        )

    summary_csv = sweep_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = sweep_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "config": config,
                "summary_rows": summary_rows,
                "detailed_files": [
                    f"radius_{radius:03d}_bw_{boundary_width:03d}.json"
                    for radius, boundary_width in combinations
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[Sweep] finished. Summary CSV: {summary_csv}")
    print(f"[Sweep] finished. Summary JSON: {summary_json}")
    print(f"[Sweep] appended experiment rows to: {history_csv}")

    return {
        "sweep_dir": str(sweep_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "history_csv": str(history_csv),
        "rows": summary_rows,
        "details": detailed_results,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for CSP radius/boundary sweep experiments."""
    parser = argparse.ArgumentParser("CSP radius sweep runner")
    parser.add_argument("--pred-dir", type=str, required=True, help="Prediction directory")
    parser.add_argument("--gt-dir", type=str, required=True, help="Ground-truth directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument(
        "--radii",
        type=parse_int_list,
        required=True,
        help="Comma-separated max_radius values, e.g. 0,1,3,5,7",
    )
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignored GT label")
    parser.add_argument(
        "--boundary-widths",
        type=parse_int_list,
        default=[1],
        help="Comma-separated boundary_width values, e.g. 1,2,3",
    )
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"], help="Morphology kernel type")
    parser.add_argument("--preview-pairs", type=int, default=0, help="How many pred/gt pairs to preview")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker count for dataset evaluation")
    parser.add_argument("--output-dir", type=str, default="outputs/csp_radius_sweeps", help="Directory for summaries and logs")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional experiment name suffix")
    parser.add_argument("--notes", type=str, default="", help="Optional free-form notes recorded with the run")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run_radius_sweep(parser.parse_args())
