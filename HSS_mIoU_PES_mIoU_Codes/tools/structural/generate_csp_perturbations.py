import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))

from csp_perturbation_utils import (
    apply_perturbation,
    build_perturbation_specs,
    parse_name_list,
    select_class_masks,
)
from structural_metrics import binary_iou


def _name_seed_offset(name: str) -> int:
    """Create a deterministic integer offset from a perturbation name."""
    return sum(ord(ch) for ch in name)


def _to_serializable(value: Any) -> Any:
    """Convert nested numpy-heavy values into JSON-safe types."""
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


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for perturbation sample generation."""
    parser = argparse.ArgumentParser("Generate controlled CSP perturbation samples")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory of GT semantic label maps")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignored GT label value")
    parser.add_argument("--output-dir", type=str, default="outputs/csp_perturbations", help="Directory to save generated samples")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional name suffix for the run directory")
    parser.add_argument("--perturbations", type=parse_name_list, default=None, help="Comma-separated perturbation names")
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"], help="Morphology kernel type")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of GT images to scan")
    parser.add_argument("--max-instances", type=int, default=100, help="Maximum number of class masks to sample")
    parser.add_argument("--max-classes-per-image", type=int, default=2, help="Maximum number of sampled classes per image")
    parser.add_argument("--min-mask-pixels", type=int, default=64, help="Minimum class area to keep")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and perturbations")
    parser.add_argument("--notes", type=str, default="", help="Optional notes stored in metadata")
    return parser


def main(args: argparse.Namespace) -> None:
    """Generate perturbation samples and save metadata for later analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or "csp_perturbations"
    run_dir = Path(args.output_dir) / f"{timestamp}_{experiment_name}"
    gt_mask_dir = run_dir / "gt_masks"
    pred_mask_dir = run_dir / "perturbed_masks"
    gt_mask_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    specs = build_perturbation_specs(args.perturbations)
    mask_records = select_class_masks(
        gt_dir=args.gt_dir,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_images=args.max_images,
        max_instances=args.max_instances,
        max_classes_per_image=args.max_classes_per_image,
        min_mask_pixels=args.min_mask_pixels,
        seed=args.seed,
    )

    metadata_rows: List[Dict[str, Any]] = []
    perturbation_counts: Dict[str, int] = {name: 0 for name in specs}

    print(f"[Generate] output directory: {run_dir}")
    print(f"[Generate] sampled class masks: {len(mask_records)}")

    for record in mask_records:
        gt_mask = record["mask"].astype(bool, copy=False)
        instance_stem = f"instance_{record['instance_id']:05d}_class_{record['class_id']:03d}"
        gt_path = gt_mask_dir / f"{instance_stem}_gt.npy"
        np.save(gt_path, gt_mask.astype(np.uint8))

        for perturbation_name, spec in specs.items():
            levels = spec["levels"]
            family = spec["family"]
            for level_index, level_value in enumerate(levels):
                seed = args.seed + int(record["instance_id"]) * 1000 + level_index * 37 + _name_seed_offset(perturbation_name)
                perturbed = apply_perturbation(
                    gt_mask,
                    perturbation_name=perturbation_name,
                    level_value=level_value,
                    seed=seed,
                    kernel_type=args.kernel_type,
                )
                pred_path = pred_mask_dir / f"{instance_stem}_{perturbation_name}_lvl{level_index + 1}.npy"
                np.save(pred_path, perturbed.astype(np.uint8))

                row = {
                    "instance_id": int(record["instance_id"]),
                    "image_index": int(record["image_index"]),
                    "image_path": record["image_path"],
                    "class_id": int(record["class_id"]),
                    "height": int(gt_mask.shape[0]),
                    "width": int(gt_mask.shape[1]),
                    "gt_area": int(gt_mask.sum()),
                    "pred_area": int(perturbed.sum()),
                    "perturbation_name": perturbation_name,
                    "perturbation_family": family,
                    "level_index": level_index + 1,
                    "level_value": float(level_value),
                    "kernel_type": args.kernel_type,
                    "gt_mask_path": str(gt_path.resolve()),
                    "pred_mask_path": str(pred_path.resolve()),
                    "binary_iou": float(binary_iou(gt_mask, perturbed)),
                    "notes": args.notes,
                }
                metadata_rows.append(row)
                perturbation_counts[perturbation_name] += 1

    metadata_csv = run_dir / "metadata.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [])
        if metadata_rows:
            writer.writeheader()
            writer.writerows(metadata_rows)

    config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "gt_dir": str(Path(args.gt_dir).resolve()),
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "kernel_type": args.kernel_type,
        "max_images": args.max_images,
        "max_instances": args.max_instances,
        "max_classes_per_image": args.max_classes_per_image,
        "min_mask_pixels": args.min_mask_pixels,
        "seed": args.seed,
        "notes": args.notes,
        "perturbations": {name: _to_serializable(spec) for name, spec in specs.items()},
        "num_sampled_masks": len(mask_records),
        "num_generated_pairs": len(metadata_rows),
        "perturbation_counts": perturbation_counts,
    }
    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Generate] metadata CSV: {metadata_csv}")
    print(f"[Generate] summary JSON: {summary_json}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
