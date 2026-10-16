#!/usr/bin/env python3
"""
Batch mIoU evaluation for the six paper datasets.

This script focuses on mIoU only and keeps VOC20b aligned with the
21-class "20 foreground + 1 background" setup.

Notes for VOC20b:
- Preferred predictions live in `visualization/all_datasets_raw/voc20b_21/pred_raw`.
- If only the legacy `voc20b/pred_raw` directory exists, the script will still
  evaluate it, but will collapse all predicted labels >= 20 to background (20)
  to match the repository's `sem_seg_background` evaluator semantics.
- That legacy fallback is useful for inspection, but it is not a replacement for
  re-running inference with the corrected 21-class prompt set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "metrics"
sys.path.insert(0, str(METRICS_DIR))

from semantic_metrics import compute_confusion_matrix, compute_miou_from_confusion  # type: ignore  # noqa: E402


SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".npy"}
PREDICTION_SUFFIXES = (
    "_pred_mask",
    "_pred",
    "_prediction",
    "_seg",
    "_mask",
    "_label",
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    gt_dir: Path
    pred_dirs: Tuple[Path, ...]
    num_classes: int
    ignore_index: int
    collapse_pred_ge_to: Optional[int] = None
    collapse_target: Optional[int] = None
    notes: str = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_prediction_suffixes(stem: str) -> str:
    out = stem
    changed = True
    while changed:
        changed = False
        for suffix in PREDICTION_SUFFIXES:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def relative_stem(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def list_mask_files(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=lambda p: p.relative_to(folder).as_posix().lower())


def choose_existing_dir(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No prediction directory exists among: {joined}")


def build_prediction_index(pred_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for pred_path in list_mask_files(pred_dir):
        key = strip_prediction_suffixes(relative_stem(pred_path, pred_dir).lower())
        index.setdefault(key, []).append(pred_path)
    return index


def match_gt_pred_files(gt_dir: Path, pred_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[str], List[str]]:
    gt_files = list_mask_files(gt_dir)
    pred_index = build_prediction_index(pred_dir)

    matched: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    ambiguous: List[str] = []

    for gt_path in gt_files:
        key = relative_stem(gt_path, gt_dir).lower()
        candidates = pred_index.get(key, [])
        if not candidates:
            missing.append(relative_stem(gt_path, gt_dir))
            continue
        if len(candidates) > 1:
            ambiguous.append(relative_stem(gt_path, gt_dir))
            continue
        matched.append((gt_path, candidates[0]))

    return matched, missing, ambiguous


def load_label_map(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        array = np.array(Image.open(path))
    return array.astype(np.int64, copy=False)


def remap_prediction(pred: np.ndarray, config: DatasetConfig) -> Tuple[np.ndarray, int]:
    if config.collapse_pred_ge_to is None or config.collapse_target is None:
        return pred, 0
    out = pred.copy()
    mask = (out >= config.collapse_pred_ge_to) & (out != config.ignore_index)
    collapsed_pixels = int(mask.sum())
    out[mask] = config.collapse_target
    return out, collapsed_pixels


def compute_detectron2_style_miou_from_confusion(confusion: np.ndarray) -> Dict[str, object]:
    """
    Match Detectron2 SemSegEvaluator mIoU semantics.

    Our local metrics.py averages over classes with union > 0. The official
    semantic segmentation evaluator averages over classes that appear in GT.
    This distinction matters a lot for long-tail datasets such as PC459,
    where many absent classes may still be predicted and should not expand the
    mIoU denominator for the paper's official numbers.
    """
    tp = np.diag(confusion).astype(np.float64)
    pos_gt = confusion.sum(axis=1).astype(np.float64)
    pos_pred = confusion.sum(axis=0).astype(np.float64)
    union = pos_gt + pos_pred - tp

    gt_present = pos_gt > 0
    per_class_iou = np.full(confusion.shape[0], np.nan, dtype=np.float64)
    valid_union = gt_present & (union > 0)
    per_class_iou[valid_union] = tp[valid_union] / union[valid_union]

    miou = float(np.nanmean(per_class_iou[gt_present])) if np.any(gt_present) else 0.0
    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "tp": tp,
        "pos_gt": pos_gt,
        "pos_pred": pos_pred,
        "union": union,
        "valid_classes": gt_present,
        "num_valid_classes": int(gt_present.sum()),
    }


def make_default_configs() -> Dict[str, DatasetConfig]:
    raw_base = Path(os.environ.get("PREDICTION_ROOT", str(ROOT / "data" / "predictions" / "all_datasets_raw")))
    return {
        "ADE150": DatasetConfig(
            name="ADE150",
            gt_dir=Path(os.environ.get("ADE150_GT_DIR", str(ROOT / "data" / "ground_truth" / "ADE150"))),
            pred_dirs=(raw_base / "ade150" / "pred_raw",),
            num_classes=150,
            ignore_index=255,
        ),
        "ADE847": DatasetConfig(
            name="ADE847",
            gt_dir=Path(os.environ.get("ADE847_GT_DIR", str(ROOT / "data" / "ground_truth" / "ADE847"))),
            pred_dirs=(raw_base / "ade847" / "pred_raw",),
            num_classes=847,
            ignore_index=65535,
        ),
        "PC59": DatasetConfig(
            name="PC59",
            gt_dir=Path(os.environ.get("PC59_GT_DIR", str(ROOT / "data" / "ground_truth" / "PC59"))),
            pred_dirs=(raw_base / "pc59" / "pred_raw",),
            num_classes=59,
            ignore_index=255,
        ),
        "PC459": DatasetConfig(
            name="PC459",
            gt_dir=Path(os.environ.get("PC459_GT_DIR", str(ROOT / "data" / "ground_truth" / "PC459"))),
            pred_dirs=(raw_base / "pc459" / "pred_raw",),
            num_classes=459,
            ignore_index=459,
        ),
        "VOC20": DatasetConfig(
            name="VOC20",
            gt_dir=Path(os.environ.get("VOC20_GT_DIR", str(ROOT / "data" / "ground_truth" / "VOC20"))),
            pred_dirs=(raw_base / "voc20" / "pred_raw",),
            num_classes=20,
            ignore_index=255,
        ),
        "VOC20b": DatasetConfig(
            name="VOC20b",
            gt_dir=Path(os.environ.get("VOC20B_GT_DIR", str(ROOT / "data" / "ground_truth" / "VOC20b"))),
            pred_dirs=(
                raw_base / "voc20b_21" / "pred_raw",
                raw_base / "voc20b" / "pred_raw",
            ),
            num_classes=21,
            ignore_index=255,
            collapse_pred_ge_to=20,
            collapse_target=20,
            notes=(
                "Preferred path is voc20b_21/pred_raw. If legacy voc20b/pred_raw is used, "
                "predicted labels >= 20 are collapsed to background (20)."
            ),
        ),
    }


def evaluate_dataset(config: DatasetConfig) -> Dict[str, object]:
    pred_dir = choose_existing_dir(config.pred_dirs)
    matched, missing, ambiguous = match_gt_pred_files(config.gt_dir, pred_dir)
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)

    collapsed_pixels_total = 0
    invalid_pred_pixels_total = 0
    pred_max_before_collapse = -1
    pred_max_after_collapse = -1

    for gt_path, pred_path in matched:
        gt = load_label_map(gt_path)
        pred = load_label_map(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch for {gt_path.name}: GT {gt.shape} vs Pred {pred.shape} ({pred_path.name})"
            )

        pred_max_before_collapse = max(pred_max_before_collapse, int(pred.max()))
        pred, collapsed_pixels = remap_prediction(pred, config)
        pred_max_after_collapse = max(pred_max_after_collapse, int(pred.max()))
        collapsed_pixels_total += collapsed_pixels

        valid_gt = (gt != config.ignore_index) & (gt >= 0) & (gt < config.num_classes)
        invalid_pred_pixels_total += int(
            (valid_gt & ((pred < 0) | (pred >= config.num_classes) | (pred == config.ignore_index))).sum()
        )
        confusion += compute_confusion_matrix(gt, pred, config.num_classes, config.ignore_index)

    official_miou_result = compute_detectron2_style_miou_from_confusion(confusion)
    union_valid_miou_result = compute_miou_from_confusion(confusion)
    using_legacy_voc20b = config.name == "VOC20b" and pred_dir.name == "pred_raw" and "voc20b_21" not in pred_dir.as_posix()

    return {
        "dataset": config.name,
        "gt_dir": str(config.gt_dir),
        "pred_dir": str(pred_dir),
        "matched_images": len(matched),
        "missing_predictions": len(missing),
        "ambiguous_predictions": len(ambiguous),
        "num_classes": config.num_classes,
        "ignore_index": config.ignore_index,
        "miou": float(official_miou_result["miou"]),
        "miou_detectron2_style": float(official_miou_result["miou"]),
        "miou_union_valid_style": float(union_valid_miou_result["miou"]),
        "num_valid_classes": int(official_miou_result["num_valid_classes"]),
        "pred_max_before_collapse": pred_max_before_collapse,
        "pred_max_after_collapse": pred_max_after_collapse,
        "collapsed_pred_pixels": collapsed_pixels_total,
        "invalid_pred_pixels_after_remap": invalid_pred_pixels_total,
        "using_legacy_voc20b_predictions": using_legacy_voc20b,
        "notes": config.notes,
        "missing_prediction_examples": missing[:10],
        "ambiguous_prediction_examples": ambiguous[:10],
        "per_class_iou": [None if np.isnan(x) else float(x) for x in official_miou_result["per_class_iou"]],
        "valid_classes": [bool(x) for x in official_miou_result["valid_classes"]],
    }


def save_json(data: object, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "dataset",
        "miou",
        "miou_detectron2_style",
        "miou_union_valid_style",
        "matched_images",
        "missing_predictions",
        "ambiguous_predictions",
        "num_classes",
        "ignore_index",
        "pred_dir",
        "pred_max_before_collapse",
        "pred_max_after_collapse",
        "collapsed_pred_pixels",
        "invalid_pred_pixels_after_remap",
        "using_legacy_voc20b_predictions",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mIoU on the six paper datasets.")
    parser.add_argument(
        "--datasets",
        default="ADE150,ADE847,PC59,PC459,VOC20,VOC20b",
        help="Comma-separated dataset names to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "miou_6datasets"),
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = make_default_configs()
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    rows: List[Dict[str, object]] = []
    for name in dataset_names:
        if name not in configs:
            raise KeyError(f"Unknown dataset: {name}. Available: {', '.join(configs)}")
        result = evaluate_dataset(configs[name])
        rows.append(result)
        legacy_tag = " [legacy-voc20b-preds]" if bool(result["using_legacy_voc20b_predictions"]) else ""
        print(
            f"{result['dataset']:>6} | mIoU={result['miou']:.6f} | "
            f"matched={result['matched_images']} | classes={result['num_classes']}{legacy_tag}"
        )

    output_dir = ensure_dir(Path(args.output_dir))
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    save_json(rows, json_path)
    save_csv(rows, csv_path)

    print()
    print(f"Saved JSON summary to: {json_path}")
    print(f"Saved CSV summary to:  {csv_path}")


if __name__ == "__main__":
    main()
