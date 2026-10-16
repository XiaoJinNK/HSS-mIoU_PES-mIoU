import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
from structural_metrics import (
    _compute_class_score_components,
    binary_iou,
)


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
    """Create the CLI parser for CSP parameter calibration."""
    parser = argparse.ArgumentParser("Calibrate CSP radius/boundary parameters with controlled perturbations")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory of GT semantic label maps")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument("--radii", type=parse_int_list, required=True, help="Comma-separated radius list, e.g. 0,1,3,5")
    parser.add_argument(
        "--boundary-widths",
        type=parse_int_list,
        required=True,
        help="Comma-separated boundary_width list, e.g. 1,2,3",
    )
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignored GT label value")
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"], help="Morphology kernel type")
    parser.add_argument("--perturbations", type=parse_name_list, default=None, help="Comma-separated perturbation names")
    parser.add_argument("--output-dir", type=str, default="outputs/csp_param_calibration", help="Directory to save calibration logs")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional name suffix for the run directory")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of GT images to scan")
    parser.add_argument("--max-instances", type=int, default=100, help="Maximum number of class masks to sample")
    parser.add_argument("--max-classes-per-image", type=int, default=2, help="Maximum number of sampled classes per image")
    parser.add_argument("--min-mask-pixels", type=int, default=64, help="Minimum class area to keep")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--notes", type=str, default="", help="Optional notes stored with the run")
    return parser


def build_perturbation_dataset(args: argparse.Namespace, specs: Dict[str, Dict[str, object]]) -> List[Dict[str, Any]]:
    """Sample binary masks from GT and generate perturbed variants in memory."""
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

    samples: List[Dict[str, Any]] = []
    for record in mask_records:
        gt_mask = record["mask"].astype(bool, copy=False)
        for perturbation_name, spec in specs.items():
            family = spec["family"]
            for level_index, level_value in enumerate(spec["levels"]):
                seed = args.seed + int(record["instance_id"]) * 1000 + level_index * 37 + _name_seed_offset(perturbation_name)
                pred_mask = apply_perturbation(
                    gt_mask,
                    perturbation_name=perturbation_name,
                    level_value=level_value,
                    seed=seed,
                    kernel_type=args.kernel_type,
                )
                samples.append(
                    {
                        "instance_id": int(record["instance_id"]),
                        "image_path": record["image_path"],
                        "class_id": int(record["class_id"]),
                        "perturbation_name": perturbation_name,
                        "family": family,
                        "level_index": level_index + 1,
                        "level_value": float(level_value),
                        "gt_mask": gt_mask,
                        "pred_mask": pred_mask,
                        "binary_iou": float(binary_iou(gt_mask, pred_mask)),
                    }
                )
    return samples


def evaluate_parameter_pair(
    samples: List[Dict[str, Any]],
    radius: int,
    boundary_width: int,
    kernel_type: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate one `(radius, boundary_width)` pair on all perturbed samples."""
    evaluated: List[Dict[str, Any]] = []

    for sample in samples:
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        persist, edge, struct = _compute_class_score_components(
            gt_mask,
            pred_mask,
            max_radius=radius,
            boundary_width=boundary_width,
            kernel_type=kernel_type,
        )
        row = {
            "instance_id": sample["instance_id"],
            "image_path": sample["image_path"],
            "class_id": sample["class_id"],
            "perturbation_name": sample["perturbation_name"],
            "family": sample["family"],
            "level_index": sample["level_index"],
            "level_value": sample["level_value"],
            "binary_iou": sample["binary_iou"],
        }
        row["persist"] = persist
        row["edge"] = edge
        row["struct"] = struct
        evaluated.append(row)

    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[(row["instance_id"], row["perturbation_name"])].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda x: x["level_index"])

    monotone_struct = []
    monotone_target = []
    separability_struct = []
    separability_target = []
    stability_ratios = []
    edge_decoupling = []
    persist_decoupling = []
    structure_advantage = []

    family_means: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    perturbation_means: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for group_rows in groups.values():
        family = group_rows[0]["family"]
        name = group_rows[0]["perturbation_name"]

        struct_seq = np.asarray([row["struct"] for row in group_rows], dtype=np.float64)
        persist_seq = np.asarray([row["persist"] for row in group_rows], dtype=np.float64)
        edge_seq = np.asarray([row["edge"] for row in group_rows], dtype=np.float64)
        iou_seq = np.asarray([row["binary_iou"] for row in group_rows], dtype=np.float64)

        if family == "edge":
            target_seq = edge_seq
        elif family == "persist":
            target_seq = persist_seq
        else:
            target_seq = struct_seq

        monotone_struct.append(float(np.all(np.diff(struct_seq) <= 1e-9)))
        monotone_target.append(float(np.all(np.diff(target_seq) <= 1e-9)))

        separability_struct.append(float(np.clip(struct_seq[0] - struct_seq[-1], 0.0, 1.0)))
        separability_target.append(float(np.clip(target_seq[0] - target_seq[-1], 0.0, 1.0)))

        target_drop = 1.0 - target_seq
        if target_drop[-1] > 1e-8:
            stability_ratios.append(float(np.clip(1.0 - min(target_drop[0] / target_drop[-1], 1.0), 0.0, 1.0)))

        for row in group_rows:
            family_means[family]["persist"].append(row["persist"])
            family_means[family]["edge"].append(row["edge"])
            family_means[family]["struct"].append(row["struct"])
            family_means[family]["iou"].append(row["binary_iou"])

            perturbation_means[name]["persist"].append(row["persist"])
            perturbation_means[name]["edge"].append(row["edge"])
            perturbation_means[name]["struct"].append(row["struct"])
            perturbation_means[name]["iou"].append(row["binary_iou"])

            persist_drop = 1.0 - row["persist"]
            edge_drop = 1.0 - row["edge"]
            struct_drop = 1.0 - row["struct"]
            iou_drop = 1.0 - row["binary_iou"]

            if family == "edge":
                edge_decoupling.append(float(edge_drop > persist_drop))
            elif family == "persist":
                persist_decoupling.append(float(persist_drop > edge_drop))
                structure_advantage.append(float(struct_drop > iou_drop))

    monotonicity_score = 0.5 * float(np.mean(monotone_struct)) + 0.5 * float(np.mean(monotone_target))
    decoupling_terms = []
    if edge_decoupling:
        decoupling_terms.append(float(np.mean(edge_decoupling)))
    if persist_decoupling:
        decoupling_terms.append(float(np.mean(persist_decoupling)))
    decoupling_score = float(np.mean(decoupling_terms)) if decoupling_terms else 0.0

    separability_score = 0.5 * float(np.mean(separability_struct)) + 0.5 * float(np.mean(separability_target))
    stability_score = float(np.mean(stability_ratios)) if stability_ratios else 0.0
    complementarity_score = float(np.mean(structure_advantage)) if structure_advantage else 0.0

    calibration_score = (
        0.30 * monotonicity_score
        + 0.25 * decoupling_score
        + 0.20 * separability_score
        + 0.15 * stability_score
        + 0.10 * complementarity_score
    )

    summary = {
        "radius": float(radius),
        "boundary_width": float(boundary_width),
        "calibration_score": float(calibration_score),
        "monotonicity_score": float(monotonicity_score),
        "decoupling_score": float(decoupling_score),
        "separability_score": float(separability_score),
        "stability_score": float(stability_score),
        "complementarity_score": float(complementarity_score),
        "mean_struct": float(np.mean([row["struct"] for row in evaluated])) if evaluated else 0.0,
        "mean_persist": float(np.mean([row["persist"] for row in evaluated])) if evaluated else 0.0,
        "mean_edge": float(np.mean([row["edge"] for row in evaluated])) if evaluated else 0.0,
        "mean_binary_iou": float(np.mean([row["binary_iou"] for row in evaluated])) if evaluated else 0.0,
    }

    for family, values in family_means.items():
        for metric_name, metric_values in values.items():
            summary[f"{family}_{metric_name}_mean"] = float(np.mean(metric_values))
    for name, values in perturbation_means.items():
        summary[f"{name}_struct_mean"] = float(np.mean(values["struct"]))
        summary[f"{name}_persist_mean"] = float(np.mean(values["persist"]))
        summary[f"{name}_edge_mean"] = float(np.mean(values["edge"]))

    return summary, evaluated


def main(args: argparse.Namespace) -> None:
    """Run the full CSP parameter calibration experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or "csp_param_calibration"
    run_dir = Path(args.output_dir) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = build_perturbation_specs(args.perturbations)
    samples = build_perturbation_dataset(args, specs)
    if not samples:
        raise RuntimeError("No perturbation samples were generated. Try increasing max_images/max_instances or lowering min_mask_pixels.")

    config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "gt_dir": str(Path(args.gt_dir).resolve()),
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "kernel_type": args.kernel_type,
        "radii": args.radii,
        "boundary_widths": args.boundary_widths,
        "max_images": args.max_images,
        "max_instances": args.max_instances,
        "max_classes_per_image": args.max_classes_per_image,
        "min_mask_pixels": args.min_mask_pixels,
        "seed": args.seed,
        "notes": args.notes,
        "perturbations": _to_serializable(specs),
        "num_samples": len(samples),
    }

    print(f"[Calibrate] output directory: {run_dir}")
    print(f"[Calibrate] perturbation samples: {len(samples)}")

    summary_rows: List[Dict[str, Any]] = []
    detailed_files: List[str] = []

    for radius, boundary_width in itertools.product(args.radii, args.boundary_widths):
        print(f"[Calibrate] evaluating radius={radius}, boundary_width={boundary_width} ...")
        start_time = time.perf_counter()
        summary, evaluated = evaluate_parameter_pair(
            samples=samples,
            radius=radius,
            boundary_width=boundary_width,
            kernel_type=args.kernel_type,
        )
        elapsed_sec = time.perf_counter() - start_time
        summary["elapsed_sec"] = float(elapsed_sec)
        summary_rows.append(summary)

        detail_name = f"radius_{radius:03d}_bw_{boundary_width:03d}.json"
        detail_path = run_dir / detail_name
        detail_path.write_text(
            json.dumps(
                {
                    "config": config,
                    "summary": summary,
                    "samples": _to_serializable(evaluated),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        detailed_files.append(detail_name)
        print(
            "[Calibrate] "
            f"r={radius} | bw={boundary_width} | "
            f"score={summary['calibration_score']:.4f} | "
            f"mono={summary['monotonicity_score']:.4f} | "
            f"decouple={summary['decoupling_score']:.4f} | "
            f"separate={summary['separability_score']:.4f} | "
            f"stability={summary['stability_score']:.4f} | "
            f"complement={summary['complementarity_score']:.4f} | "
            f"time={elapsed_sec:.2f}s"
        )

    summary_rows.sort(key=lambda row: row["calibration_score"], reverse=True)

    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = run_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "config": config,
                "summary_rows": _to_serializable(summary_rows),
                "top3": _to_serializable(summary_rows[:3]),
                "detailed_files": detailed_files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sample_manifest = run_dir / "sample_manifest.csv"
    with sample_manifest.open("w", newline="", encoding="utf-8") as f:
        base_fields = [
            "instance_id",
            "image_path",
            "class_id",
            "perturbation_name",
            "family",
            "level_index",
            "level_value",
            "binary_iou",
        ]
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample[key] for key in base_fields})

    print(f"[Calibrate] summary CSV: {summary_csv}")
    print(f"[Calibrate] summary JSON: {summary_json}")
    print("[Calibrate] Top 3 parameter pairs:")
    for row in summary_rows[:3]:
        print(
            f"  radius={int(row['radius'])}, boundary_width={int(row['boundary_width'])}, "
            f"calibration_score={row['calibration_score']:.4f}"
        )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
