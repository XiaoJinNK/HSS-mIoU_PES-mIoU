#!/usr/bin/env python3
"""
Analyze structural metric variants on a real prediction dataset.

This script is a diagnostic tool for understanding why the current
Persistence / Edge definitions behave the way they do on full datasets.

Compared metrics:
1. Current Persistence:
   mean_r IoU(Erode(G, r), Erode(P, r)), r in [0, R]
2. Variant Persistence-no-r0:
   mean_r IoU(Erode(G, r), Erode(P, r)), r in [1, R]
3. Current Edge:
   IoU(boundary_band(G), boundary_band(P))
4. Variant Edge-BF1:
   tolerant boundary F1 on binary class masks

Outputs:
- per-image CSV for the variant metrics
- JSON / Markdown summary with:
  - image-level means
  - dataset-level class-macro metrics
  - image-level correlations to mIoU
  - class-level correlations to per-class IoU
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import find_boundaries


ROOT = Path(__file__).resolve().parents[2]
STRUCT_PATH = ROOT / "metrics" / "structural_metrics.py"

_STRUCT_SPEC = importlib.util.spec_from_file_location("structural_metrics", STRUCT_PATH)
if _STRUCT_SPEC is None or _STRUCT_SPEC.loader is None:
    raise RuntimeError(f"Failed to load structural_metrics from {STRUCT_PATH}")
structural_metrics = importlib.util.module_from_spec(_STRUCT_SPEC)
_STRUCT_SPEC.loader.exec_module(structural_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze structural metric variants")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--max-radius", type=int, default=3)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--bf1-tolerance", type=int, default=2)
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--official-summary-json", type=Path, default=None)
    parser.add_argument("--official-image-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def match_pred_gt_files(pred_dir: Path, gt_dir: Path) -> List[Tuple[Path, Path, str]]:
    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and "_pred_mask" in p.stem])
    gt_by_stem = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}
    pairs: List[Tuple[Path, Path, str]] = []
    for pred_path in pred_files:
        if "_pred_mask" not in pred_path.stem:
            continue
        image_id = pred_path.stem.split("_pred_mask")[0]
        gt_path = gt_by_stem.get(image_id)
        if gt_path is not None:
            pairs.append((pred_path, gt_path, image_id))
    if not pairs:
        raise RuntimeError(f"No matched prediction/GT pairs found in {pred_dir} vs {gt_dir}")
    return pairs


def boundary_f1_binary(gt_mask: np.ndarray, pred_mask: np.ndarray, tolerance: int) -> Tuple[float, int, int, int, int]:
    gt_b = find_boundaries(gt_mask.astype(np.int8, copy=False), mode="thick")
    pred_b = find_boundaries(pred_mask.astype(np.int8, copy=False), mode="thick")

    gt_count = int(gt_b.sum())
    pred_count = int(pred_b.sum())
    if gt_count == 0 and pred_count == 0:
        return float("nan"), 0, 0, 0, 0

    if tolerance > 0:
        footprint = disk(tolerance)
        gt_d = ndi.binary_dilation(gt_b, structure=footprint)
        pred_d = ndi.binary_dilation(pred_b, structure=footprint)
    else:
        gt_d = gt_b
        pred_d = pred_b

    pred_match = int((pred_b & gt_d).sum())
    gt_match = int((gt_b & pred_d).sum())
    precision = float(pred_match / max(1, pred_count))
    recall = float(gt_match / max(1, gt_count))
    if precision + recall == 0.0:
        return 0.0, pred_match, pred_count, gt_match, gt_count
    score = 2.0 * precision * recall / (precision + recall)
    return float(score), pred_match, pred_count, gt_match, gt_count


def compute_image_miou(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int) -> float:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    structural_metrics.update_confusion_matrix(confusion, gt, pred, num_classes, ignore_index)
    return float(structural_metrics.miou_from_confusion(confusion)["miou"])


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a_arr) & np.isfinite(b_arr)
    if int(valid.sum()) < 2:
        return float("nan")
    a_valid = a_arr[valid]
    b_valid = b_arr[valid]
    if np.allclose(a_valid, a_valid[0]) or np.allclose(b_valid, b_valid[0]):
        return float("nan")
    return float(np.corrcoef(a_valid, b_valid)[0, 1])


def summarize_distribution(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    bins = [
        ("=0", lambda x: x == 0.0),
        ("(0,0.01)", lambda x: (x > 0.0) & (x < 0.01)),
        ("[0.01,0.05)", lambda x: (x >= 0.01) & (x < 0.05)),
        ("[0.05,0.1)", lambda x: (x >= 0.05) & (x < 0.1)),
        ("[0.1,0.2)", lambda x: (x >= 0.1) & (x < 0.2)),
        ("[0.2,0.4)", lambda x: (x >= 0.2) & (x < 0.4)),
        ("[0.4,0.6)", lambda x: (x >= 0.4) & (x < 0.6)),
        ("[0.6,0.8)", lambda x: (x >= 0.6) & (x < 0.8)),
        ("[0.8,1]", lambda x: (x >= 0.8) & (x <= 1.0)),
    ]
    out = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for name, fn in bins:
        out[name] = int(fn(arr).sum())
    return out


def _worker(task: Tuple[Path, Path, str, int, int, int, int, int, str]) -> Dict[str, Any]:
    pred_path, gt_path, image_id, num_classes, ignore_index, max_radius, boundary_width, bf1_tolerance, kernel_type = task
    pred = structural_metrics.load_label_map(pred_path)
    gt = structural_metrics.load_label_map(gt_path)

    current = structural_metrics.compute_multiclass_struct_score(
        gt=gt,
        pred=pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_radius=max_radius,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )
    image_miou = compute_image_miou(gt, pred, num_classes=num_classes, ignore_index=ignore_index)

    valid_pixels = gt != ignore_index
    gt_present = np.unique(gt[valid_pixels])
    pred_present = np.unique(pred[valid_pixels])
    active_classes = np.union1d(gt_present, pred_present)
    active_classes = active_classes[(active_classes >= 0) & (active_classes < num_classes)].astype(np.int64)

    persist_no_r0 = np.full((num_classes,), np.nan, dtype=np.float64)
    edge_bf1 = np.full((num_classes,), np.nan, dtype=np.float64)
    struct_variant = np.full((num_classes,), np.nan, dtype=np.float64)
    persist_no_r0_valid = np.zeros((num_classes,), dtype=bool)
    edge_bf1_valid = np.zeros((num_classes,), dtype=bool)
    struct_variant_valid = np.zeros((num_classes,), dtype=bool)

    bf1_pred_match = np.zeros((num_classes,), dtype=np.uint32)
    bf1_pred_total = np.zeros((num_classes,), dtype=np.uint32)
    bf1_gt_match = np.zeros((num_classes,), dtype=np.uint32)
    bf1_gt_total = np.zeros((num_classes,), dtype=np.uint32)

    for class_id in active_classes.tolist():
        persist_inter = np.asarray(current["per_class_persist_inter"][class_id], dtype=np.float64)
        persist_union = np.asarray(current["per_class_persist_union"][class_id], dtype=np.float64)
        if persist_union.shape[0] > 1:
            valid_r = persist_union[1:] > 0
            if np.any(valid_r):
                persist_no_r0[class_id] = float(np.mean(persist_inter[1:][valid_r] / persist_union[1:][valid_r]))
                persist_no_r0_valid[class_id] = True

        gt_mask = (gt == class_id) & valid_pixels
        pred_mask = (pred == class_id) & valid_pixels
        bf1, pred_match, pred_total, gt_match, gt_total = boundary_f1_binary(
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            tolerance=bf1_tolerance,
        )
        if math.isfinite(bf1):
            edge_bf1[class_id] = bf1
            edge_bf1_valid[class_id] = True
            bf1_pred_match[class_id] = pred_match
            bf1_pred_total[class_id] = pred_total
            bf1_gt_match[class_id] = gt_match
            bf1_gt_total[class_id] = gt_total

        if persist_no_r0_valid[class_id] and edge_bf1_valid[class_id]:
            struct_variant[class_id] = float(math.sqrt(max(persist_no_r0[class_id], 0.0) * max(edge_bf1[class_id], 0.0)))
            struct_variant_valid[class_id] = True

    image_persist_no_r0 = float(np.nanmean(persist_no_r0[persist_no_r0_valid])) if np.any(persist_no_r0_valid) else 0.0
    image_edge_bf1 = float(np.nanmean(edge_bf1[edge_bf1_valid])) if np.any(edge_bf1_valid) else 0.0
    image_struct_variant = float(np.nanmean(struct_variant[struct_variant_valid])) if np.any(struct_variant_valid) else 0.0

    return {
        "image_id": image_id,
        "miou": image_miou,
        "persist_no_r0": image_persist_no_r0,
        "edge_bf1": image_edge_bf1,
        "struct_variant": image_struct_variant,
        "persist_inter": np.asarray(current["per_class_persist_inter"], dtype=np.uint32),
        "persist_union": np.asarray(current["per_class_persist_union"], dtype=np.uint32),
        "bf1_pred_match": bf1_pred_match,
        "bf1_pred_total": bf1_pred_total,
        "bf1_gt_match": bf1_gt_match,
        "bf1_gt_total": bf1_gt_total,
    }


def aggregate_persist_no_r0(persist_inter_sum: np.ndarray, persist_union_sum: np.ndarray) -> Dict[str, Any]:
    num_classes = persist_inter_sum.shape[0]
    per_class = np.full((num_classes,), np.nan, dtype=np.float64)
    valid_mask = np.zeros((num_classes,), dtype=bool)
    for class_id in range(num_classes):
        valid_r = persist_union_sum[class_id, 1:] > 0
        if np.any(valid_r):
            per_class[class_id] = np.mean(persist_inter_sum[class_id, 1:][valid_r] / persist_union_sum[class_id, 1:][valid_r])
            valid_mask[class_id] = True
    score = float(np.nanmean(per_class[valid_mask])) if np.any(valid_mask) else 0.0
    return {"score": score, "per_class": per_class, "valid_mask": valid_mask}


def aggregate_boundary_f1(
    pred_match_sum: np.ndarray,
    pred_total_sum: np.ndarray,
    gt_match_sum: np.ndarray,
    gt_total_sum: np.ndarray,
) -> Dict[str, Any]:
    num_classes = pred_match_sum.shape[0]
    per_class = np.full((num_classes,), np.nan, dtype=np.float64)
    valid_mask = np.zeros((num_classes,), dtype=bool)
    for class_id in range(num_classes):
        pred_total = int(pred_total_sum[class_id])
        gt_total = int(gt_total_sum[class_id])
        if pred_total == 0 and gt_total == 0:
            continue
        precision = float(pred_match_sum[class_id] / max(1, pred_total))
        recall = float(gt_match_sum[class_id] / max(1, gt_total))
        if precision + recall == 0.0:
            per_class[class_id] = 0.0
        else:
            per_class[class_id] = float(2.0 * precision * recall / (precision + recall))
        valid_mask[class_id] = True
    score = float(np.nanmean(per_class[valid_mask])) if np.any(valid_mask) else 0.0
    return {"score": score, "per_class": per_class, "valid_mask": valid_mask}


def write_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    metrics = summary["dataset_metrics"]
    corr = summary["correlations"]

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(value_f):
            return "n/a"
        return f"{value_f:.6f}"

    lines = [
        f"# {summary['dataset_name']} structural metric variant analysis",
        "",
        "## Dataset-level metrics",
        "",
        "| Metric | Current | Variant |",
        "| --- | ---: | ---: |",
        f"| Persistence | {fmt(metrics.get('current_persistence'))} | {fmt(metrics.get('persist_no_r0'))} |",
        f"| Edge | {fmt(metrics.get('current_edge'))} | {fmt(metrics.get('edge_bf1'))} |",
        f"| Struct | {fmt(metrics.get('current_struct'))} | {fmt(metrics.get('struct_variant'))} |",
        "",
        "## Correlations",
        "",
        "| Relation | Current | Variant |",
        "| --- | ---: | ---: |",
        f"| image mIoU vs persistence | {fmt(corr.get('image_miou_vs_current_persistence'))} | {fmt(corr.get('image_miou_vs_persist_no_r0'))} |",
        f"| image mIoU vs edge | {fmt(corr.get('image_miou_vs_current_edge'))} | {fmt(corr.get('image_miou_vs_edge_bf1'))} |",
        f"| image mIoU vs struct | {fmt(corr.get('image_miou_vs_current_struct'))} | {fmt(corr.get('image_miou_vs_struct_variant'))} |",
        f"| class IoU vs persistence | {fmt(corr.get('class_iou_vs_current_persistence'))} | {fmt(corr.get('class_iou_vs_persist_no_r0'))} |",
        f"| class IoU vs edge | {fmt(corr.get('class_iou_vs_current_edge'))} | {fmt(corr.get('class_iou_vs_edge_bf1'))} |",
        f"| class IoU vs struct | {fmt(corr.get('class_iou_vs_current_struct'))} | {fmt(corr.get('class_iou_vs_struct_variant'))} |",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = match_pred_gt_files(args.pred_dir, args.gt_dir)
    if args.max_images is not None:
        pairs = pairs[: args.max_images]

    current_summary = None
    current_images = None
    if args.official_summary_json is not None:
        current_summary = json.loads(args.official_summary_json.read_text(encoding="utf-8"))
    if args.official_image_csv is not None:
        current_images = pd.read_csv(args.official_image_csv)

    tasks = [
        (
            pred_path,
            gt_path,
            image_id,
            args.num_classes,
            args.ignore_index,
            args.max_radius,
            args.boundary_width,
            args.bf1_tolerance,
            args.kernel_type,
        )
        for pred_path, gt_path, image_id in pairs
    ]

    persist_inter_sum = np.zeros((args.num_classes, args.max_radius + 1), dtype=np.uint64)
    persist_union_sum = np.zeros((args.num_classes, args.max_radius + 1), dtype=np.uint64)
    bf1_pred_match_sum = np.zeros((args.num_classes,), dtype=np.uint64)
    bf1_pred_total_sum = np.zeros((args.num_classes,), dtype=np.uint64)
    bf1_gt_match_sum = np.zeros((args.num_classes,), dtype=np.uint64)
    bf1_gt_total_sum = np.zeros((args.num_classes,), dtype=np.uint64)
    image_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            item = future.result()
            image_rows.append(
                {
                    "image_id": item["image_id"],
                    "miou": item["miou"],
                    "persist_no_r0": item["persist_no_r0"],
                    "edge_bf1": item["edge_bf1"],
                    "struct_variant": item["struct_variant"],
                }
            )
            persist_inter_sum += item["persist_inter"]
            persist_union_sum += item["persist_union"]
            bf1_pred_match_sum += item["bf1_pred_match"]
            bf1_pred_total_sum += item["bf1_pred_total"]
            bf1_gt_match_sum += item["bf1_gt_match"]
            bf1_gt_total_sum += item["bf1_gt_total"]

    image_df = pd.DataFrame(image_rows).sort_values("image_id").reset_index(drop=True)
    image_df.to_csv(args.output_dir / "image_level_variant_metrics.csv", index=False)

    persist_no_r0 = aggregate_persist_no_r0(persist_inter_sum, persist_union_sum)
    edge_bf1 = aggregate_boundary_f1(
        pred_match_sum=bf1_pred_match_sum,
        pred_total_sum=bf1_pred_total_sum,
        gt_match_sum=bf1_gt_match_sum,
        gt_total_sum=bf1_gt_total_sum,
    )

    struct_variant_per_class = np.full((args.num_classes,), np.nan, dtype=np.float64)
    struct_variant_valid = persist_no_r0["valid_mask"] & edge_bf1["valid_mask"]
    struct_variant_per_class[struct_variant_valid] = np.sqrt(
        np.maximum(persist_no_r0["per_class"][struct_variant_valid], 0.0)
        * np.maximum(edge_bf1["per_class"][struct_variant_valid], 0.0)
    )
    struct_variant_score = (
        float(np.nanmean(struct_variant_per_class[struct_variant_valid]))
        if np.any(struct_variant_valid)
        else 0.0
    )

    summary: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "num_images": int(len(image_df)),
        "num_classes": int(args.num_classes),
        "config": {
            "pred_dir": str(args.pred_dir),
            "gt_dir": str(args.gt_dir),
            "ignore_index": int(args.ignore_index),
            "max_radius": int(args.max_radius),
            "boundary_width": int(args.boundary_width),
            "bf1_tolerance": int(args.bf1_tolerance),
            "kernel_type": args.kernel_type,
            "num_workers": int(args.num_workers),
        },
        "dataset_metrics": {
            "persist_no_r0": float(persist_no_r0["score"]),
            "edge_bf1": float(edge_bf1["score"]),
            "struct_variant": float(struct_variant_score),
            "persist_no_r0_image_mean": float(image_df["persist_no_r0"].mean()),
            "edge_bf1_image_mean": float(image_df["edge_bf1"].mean()),
            "struct_variant_image_mean": float(image_df["struct_variant"].mean()),
        },
        "correlations": {
            "image_miou_vs_persist_no_r0": safe_corr(image_df["miou"], image_df["persist_no_r0"]),
            "image_miou_vs_edge_bf1": safe_corr(image_df["miou"], image_df["edge_bf1"]),
            "image_miou_vs_struct_variant": safe_corr(image_df["miou"], image_df["struct_variant"]),
        },
        "distributions": {
            "persist_no_r0": summarize_distribution(persist_no_r0["per_class"][persist_no_r0["valid_mask"]]),
            "edge_bf1": summarize_distribution(edge_bf1["per_class"][edge_bf1["valid_mask"]]),
            "struct_variant": summarize_distribution(struct_variant_per_class[struct_variant_valid]),
        },
        "variant_valid_counts": {
            "persist_no_r0_classes": int(persist_no_r0["valid_mask"].sum()),
            "edge_bf1_classes": int(edge_bf1["valid_mask"].sum()),
            "struct_variant_classes": int(struct_variant_valid.sum()),
        },
    }

    if current_summary is not None:
        summary_row = current_summary["summary_row"]
        struct = current_summary["dataset_metrics"]["struct"]
        current_image_df = current_images.copy() if current_images is not None else None
        if current_image_df is not None:
            merged = current_image_df.merge(image_df, on="image_id", how="inner", validate="one_to_one")
            summary["dataset_metrics"].update(
                {
                    "current_persistence": float(summary_row["persistence"]),
                    "current_edge": float(summary_row["edge"]),
                    "current_struct": float(summary_row["structscore"]),
                    "current_persistence_image_mean": float(summary_row["persistence_image_mean"]),
                    "current_edge_image_mean": float(summary_row["edge_image_mean"]),
                    "current_struct_image_mean": float(summary_row["structscore_image_mean"]),
                }
            )
            summary["correlations"].update(
                {
                    "image_miou_vs_current_persistence": safe_corr(merged["miou_x"], merged["persist"]),
                    "image_miou_vs_current_edge": safe_corr(merged["miou_x"], merged["edge"]),
                    "image_miou_vs_current_struct": safe_corr(merged["miou_x"], merged["structscore"]),
                }
            )
        else:
            summary["dataset_metrics"].update(
                {
                    "current_persistence": float(summary_row["persistence"]),
                    "current_edge": float(summary_row["edge"]),
                    "current_struct": float(summary_row["structscore"]),
                    "current_persistence_image_mean": float(summary_row["persistence_image_mean"]),
                    "current_edge_image_mean": float(summary_row["edge_image_mean"]),
                    "current_struct_image_mean": float(summary_row["structscore_image_mean"]),
                }
            )

        per_class_iou = np.asarray(current_summary["dataset_metrics"]["mIoU"]["per_class_iou"], dtype=np.float64)
        current_persist = np.asarray(struct["miou_style"]["per_class_persist_mean"], dtype=np.float64)
        current_edge = np.asarray(struct["miou_style"]["per_class_edge_mean"], dtype=np.float64)
        current_struct = np.asarray(struct["miou_style"]["per_class_mean"], dtype=np.float64)
        summary["correlations"].update(
            {
                "class_iou_vs_current_persistence": safe_corr(per_class_iou, current_persist),
                "class_iou_vs_current_edge": safe_corr(per_class_iou, current_edge),
                "class_iou_vs_current_struct": safe_corr(per_class_iou, current_struct),
                "class_iou_vs_persist_no_r0": safe_corr(per_class_iou, persist_no_r0["per_class"]),
                "class_iou_vs_edge_bf1": safe_corr(per_class_iou, edge_bf1["per_class"]),
                "class_iou_vs_struct_variant": safe_corr(per_class_iou, struct_variant_per_class),
            }
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
