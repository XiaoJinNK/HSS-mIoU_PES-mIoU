#!/usr/bin/env python3
"""
Systematic evaluation for real prediction directories.

This script reuses the metric implementations in:
- metrics/semantic_metrics.py
- metrics/structural_metrics.py

It evaluates multiple prediction directories on:
- mIoU
- hierIoU
- StructScore

and exports tables, JSON summaries, plots, rankings, and case studies.
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import math
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
ensure_mpl_dir = ROOT / ".mplconfig"
ensure_cache_dir = ROOT / ".cache"
ensure_mpl_dir.mkdir(parents=True, exist_ok=True)
ensure_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ensure_mpl_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(ensure_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    class _DummyTqdm:
        def __init__(self, iterable: Optional[Iterable[Any]] = None, **_: Any) -> None:
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, _: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

        def set_postfix(self, *_: Any, **__: Any) -> None:
            return None

        def __enter__(self) -> "_DummyTqdm":
            return self

        def __exit__(self, *_: Any) -> None:
            self.close()

    def tqdm(iterable: Optional[Iterable[Any]] = None, **kwargs: Any) -> _DummyTqdm:
        return _DummyTqdm(iterable=iterable, **kwargs)


METRICS_DIR = ROOT / "metrics"

sys.path.insert(0, str(METRICS_DIR))

from semantic_metrics import (  # type: ignore  # noqa: E402
    compute_confusion_matrix,
    compute_hier_soft_iou_from_confusion,
    compute_miou_from_confusion,
)
from structural_metrics import (  # type: ignore  # noqa: E402
    _aggregate_struct_counts,
    _aggregate_struct_dataset_results,
    compute_multiclass_struct_score,
    load_label_map as _load_label_map,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
PREDICTION_SUFFIXES = (
    "_pred_mask",
    "_pred",
    "_prediction",
    "_seg",
    "_mask",
    "_label",
)
VISUALIZATION_HINTS = (
    "overlay",
    "labeled",
    "labelled",
    "visual",
    "vis",
    "color",
    "colour",
)
GT_SUMMARY_SUFFIX = "_gt_summary"
DEFAULT_CASE_GT_VIS_DIR = ROOT / "data" / "gt_visual_summaries"
IMAGE_WORKER_CONTEXT: Dict[str, Any] = {}


@dataclass
class MatchedPair:
    image_id: str
    gt_path: Path
    pred_path: Path
    gt_rel: str
    pred_rel: str


@dataclass
class GtSource:
    image_id: str
    gt_path: Path
    gt_rel: str


def warn(message: str, warnings: List[str]) -> None:
    text = f"[warning] {message}"
    print(text)
    warnings.append(message)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_slug(name: str) -> str:
    slug = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            slug.append(ch)
        else:
            slug.append("_")
    value = "".join(slug).strip("._")
    return value or "item"


def relative_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.with_suffix("").as_posix()


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


def strip_gt_summary_suffix(stem: str) -> str:
    if stem.endswith(GT_SUMMARY_SUFFIX):
        return stem[: -len(GT_SUMMARY_SUFFIX)]
    return stem


def normalized_relative_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = strip_prediction_suffixes(rel.stem.lower())
    parts = list(rel.parts[:-1]) + [stem]
    return Path(*parts).as_posix() if parts else stem


def is_label_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_visualization_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(token in lower_name for token in VISUALIZATION_HINTS)


def list_label_files(folder: Path, skip_visualizations: bool = False) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [p for p in folder.rglob("*") if is_label_file(p)]
    if skip_visualizations:
        files = [p for p in files if not is_visualization_file(p)]
    return sorted(files, key=lambda p: p.relative_to(folder).as_posix().lower())


def list_gt_sources(gt_dir: Path, warnings: List[str]) -> List[GtSource]:
    summary_files = sorted(gt_dir.rglob(f"*{GT_SUMMARY_SUFFIX}.json"))
    if summary_files:
        gt_sources: List[GtSource] = []
        for summary_path in summary_files:
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception as exc:  # pragma: no cover - safety path
                warn(f"Failed to read GT summary '{summary_path}': {type(exc).__name__}: {exc}", warnings)
                continue

            image_id = str(summary.get("image_id") or strip_gt_summary_suffix(summary_path.stem))
            mask_path_value = summary.get("mask_path")
            if not mask_path_value:
                warn(f"GT summary '{summary_path}' is missing 'mask_path'. Skipped.", warnings)
                continue

            mask_path = Path(mask_path_value)
            if not mask_path.exists():
                warn(f"GT mask_path from '{summary_path}' does not exist: {mask_path}", warnings)
                continue

            gt_sources.append(
                GtSource(
                    image_id=image_id,
                    gt_path=mask_path,
                    gt_rel=image_id,
                )
            )

        if gt_sources:
            return sorted(gt_sources, key=lambda item: item.gt_rel.lower())

    gt_files = list_label_files(gt_dir, skip_visualizations=False)
    return [
        GtSource(
            image_id=relative_stem(gt_path, gt_dir),
            gt_path=gt_path,
            gt_rel=relative_stem(gt_path, gt_dir),
        )
        for gt_path in gt_files
    ]


def _pick_unique_match(
    candidate_groups: Sequence[Tuple[str, Sequence[Path]]],
    gt_rel: str,
    pred_dir: Path,
    warnings: List[str],
) -> Optional[Path]:
    for strategy, candidates in candidate_groups:
        unique_candidates = sorted(
            {c.resolve(): c for c in candidates}.values(),
            key=lambda p: p.relative_to(pred_dir).as_posix().lower(),
        )
        if not unique_candidates:
            continue
        if len(unique_candidates) == 1:
            return unique_candidates[0]
        warn(
            f"Ambiguous prediction match for GT '{gt_rel}' using {strategy}: "
            f"{[c.relative_to(pred_dir).as_posix() for c in unique_candidates[:5]]}",
            warnings,
        )
        return None
    return None


def match_gt_pred_files(gt_dir: Path, pred_dir: Path, warnings: List[str]) -> List[MatchedPair]:
    gt_sources = list_gt_sources(gt_dir, warnings)
    pred_files = list_label_files(pred_dir, skip_visualizations=True)

    if not gt_sources:
        raise RuntimeError(f"No GT label files found in {gt_dir}")
    if not pred_files:
        warn(f"No prediction label files found in {pred_dir}", warnings)
        return []

    pred_by_rel: Dict[str, List[Path]] = {}
    pred_by_stem: Dict[str, List[Path]] = {}
    pred_by_norm_rel: Dict[str, List[Path]] = {}
    pred_by_norm_stem: Dict[str, List[Path]] = {}

    for pred_path in pred_files:
        rel_key = relative_stem(pred_path, pred_dir).lower()
        stem_key = pred_path.stem.lower()
        norm_rel_key = normalized_relative_key(pred_path, pred_dir).lower()
        norm_stem_key = strip_prediction_suffixes(pred_path.stem.lower())

        pred_by_rel.setdefault(rel_key, []).append(pred_path)
        pred_by_stem.setdefault(stem_key, []).append(pred_path)
        pred_by_norm_rel.setdefault(norm_rel_key, []).append(pred_path)
        pred_by_norm_stem.setdefault(norm_stem_key, []).append(pred_path)

    matched: List[MatchedPair] = []

    for gt_source in gt_sources:
        gt_rel = gt_source.gt_rel
        gt_rel_key = gt_rel.lower()
        gt_stem_key = Path(gt_rel).stem.lower()
        norm_gt_rel_key = strip_prediction_suffixes(gt_rel_key)
        norm_gt_stem_key = strip_prediction_suffixes(gt_stem_key)

        rel_suffix_candidates = []
        stem_suffix_candidates = []
        for suffix in PREDICTION_SUFFIXES:
            rel_suffix_candidates.extend(pred_by_rel.get(f"{gt_rel_key}{suffix}", []))
            stem_suffix_candidates.extend(pred_by_stem.get(f"{gt_stem_key}{suffix}", []))

        candidate_groups = [
            ("exact_relative", pred_by_rel.get(gt_rel_key, [])),
            ("exact_stem", pred_by_stem.get(gt_stem_key, [])),
            ("relative_plus_suffix", rel_suffix_candidates),
            ("stem_plus_suffix", stem_suffix_candidates),
            ("normalized_relative", pred_by_norm_rel.get(norm_gt_rel_key, [])),
            ("normalized_stem", pred_by_norm_stem.get(norm_gt_stem_key, [])),
        ]
        pred_path = _pick_unique_match(candidate_groups, gt_rel, pred_dir, warnings)
        if pred_path is None:
            warn(f"Missing prediction for GT '{gt_rel}' in {pred_dir}", warnings)
            continue

        matched.append(
            MatchedPair(
                image_id=gt_source.image_id,
                gt_path=gt_source.gt_path,
                pred_path=pred_path,
                gt_rel=gt_rel,
                pred_rel=relative_stem(pred_path, pred_dir),
            )
        )

    return matched


def apply_label_offset(label_map: np.ndarray, offset: int, ignore_index: int) -> np.ndarray:
    if offset == 0:
        return label_map.astype(np.int64, copy=False)
    out = label_map.astype(np.int64, copy=True)
    valid = out != ignore_index
    out[valid] += offset
    return out


def collapse_label_map_ge(
    label_map: np.ndarray,
    collapse_ge_to: Optional[int],
    collapse_target: Optional[int],
    ignore_index: int,
) -> np.ndarray:
    if collapse_ge_to is None or collapse_target is None:
        return label_map
    out = label_map.astype(np.int64, copy=True)
    mask = (out >= collapse_ge_to) & (out != ignore_index)
    out[mask] = collapse_target
    return out


def load_label_map(
    path: Path,
    offset: int = 0,
    ignore_index: int = 255,
    collapse_ge_to: Optional[int] = None,
    collapse_target: Optional[int] = None,
) -> np.ndarray:
    label = _load_label_map(path)
    out = apply_label_offset(label, offset=offset, ignore_index=ignore_index)
    return collapse_label_map_ge(
        out,
        collapse_ge_to=collapse_ge_to,
        collapse_target=collapse_target,
        ignore_index=ignore_index,
    )


def load_similarity_matrix(path_or_dir: Path, num_classes: int, warnings: List[str]) -> np.ndarray:
    matrix_path = path_or_dir / "S.npy" if path_or_dir.is_dir() else path_or_dir
    if not matrix_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {matrix_path}")

    matrix = np.load(matrix_path).astype(np.float64)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(
            f"Similarity matrix shape mismatch: expected {(num_classes, num_classes)}, got {matrix.shape}"
        )

    if not np.allclose(matrix, matrix.T, atol=1e-6):
        warn("Similarity matrix is not symmetric.", warnings)
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        warn("Similarity matrix diagonal is not all ones.", warnings)
    if float(np.nanmin(matrix)) < 0.0 or float(np.nanmax(matrix)) > 1.0:
        warn(
            f"Similarity matrix has values outside [0, 1]: "
            f"[{float(np.nanmin(matrix)):.6f}, {float(np.nanmax(matrix)):.6f}]",
            warnings,
        )
    return matrix


def load_class_groups(path: Optional[Path]) -> Optional[Dict[str, List[int]]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("class_groups JSON must be a dict: {group_name: [class_ids...]}")

    out: Dict[str, List[int]] = {}
    for group_name, class_ids in data.items():
        if not isinstance(class_ids, list):
            raise ValueError(f"class_groups['{group_name}'] must be a list of class ids")
        out[str(group_name)] = [int(x) for x in class_ids]
    return out


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return value


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_for_json(data), f, ensure_ascii=False, indent=2)


def normalize_csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_csv(rows: Sequence[Dict[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(row.get(key, "")) for key in fieldnames})


def save_rgb_image(image: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def build_label_palette(size: int, ignore_index: int) -> np.ndarray:
    palette = np.zeros((size, 3), dtype=np.uint8)
    ids = np.arange(size, dtype=np.uint32)
    palette[:, 0] = ((ids * 37 + 59) % 256).astype(np.uint8)
    palette[:, 1] = ((ids * 67 + 17) % 256).astype(np.uint8)
    palette[:, 2] = ((ids * 97 + 101) % 256).astype(np.uint8)
    if size > 0:
        palette[0] = np.array([32, 32, 32], dtype=np.uint8)
    if 0 <= ignore_index < size:
        palette[ignore_index] = np.array([245, 245, 245], dtype=np.uint8)
    return palette


def colorize_label_map(label_map: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    if label_map.size == 0:
        return np.zeros(label_map.shape + (3,), dtype=np.uint8)

    max_label = int(np.max(label_map))
    palette_size = max(num_classes, ignore_index + 1, max_label + 1)
    palette = build_label_palette(palette_size, ignore_index)

    rgb = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    valid = (label_map >= 0) & (label_map < palette_size)
    rgb[valid] = palette[label_map[valid]]
    rgb[~valid] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def build_agreement_map(gt: np.ndarray, pred: np.ndarray, ignore_index: int) -> np.ndarray:
    agreement = np.full(gt.shape + (3,), 235, dtype=np.uint8)
    ignore_mask = gt == ignore_index
    valid_mask = ~ignore_mask
    match_mask = valid_mask & (gt == pred)
    mismatch_mask = valid_mask & (gt != pred)
    pred_ignore_mask = valid_mask & (pred == ignore_index)

    agreement[ignore_mask] = np.array([230, 230, 230], dtype=np.uint8)
    agreement[match_mask] = np.array([76, 175, 80], dtype=np.uint8)
    agreement[mismatch_mask] = np.array([220, 76, 70], dtype=np.uint8)
    agreement[pred_ignore_mask] = np.array([142, 68, 173], dtype=np.uint8)
    return agreement


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def find_existing_visual(path_candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in path_candidates:
        if candidate.exists():
            return candidate
    return None


def candidate_visual_dirs(primary_path: Path, extra_dirs: Sequence[Optional[Path]]) -> List[Path]:
    ordered_dirs = [primary_path.parent]
    ordered_dirs.extend(path for path in extra_dirs if path is not None)
    unique_dirs: List[Path] = []
    seen = set()
    for folder in ordered_dirs:
        resolved = folder.resolve()
        if resolved in seen or not folder.exists():
            continue
        seen.add(resolved)
        unique_dirs.append(folder)
    return unique_dirs


def resolve_case_visual_paths(
    image_id: str,
    role: str,
    primary_path: Path,
    extra_dirs: Sequence[Optional[Path]],
) -> Dict[str, Optional[Path]]:
    image_name = Path(image_id).name
    search_dirs = candidate_visual_dirs(primary_path, extra_dirs)
    if role == "gt":
        labeled_names = [f"{image_name}_gt_labeled.png", f"{image_name}_labeled.png"]
        overlay_names = [f"{image_name}_gt_overlay.png", f"{image_name}_overlay.png"]
        summary_names = [f"{image_name}_gt_summary.json"]
    else:
        labeled_names = [f"{image_name}_pred_labeled.png", f"{image_name}_labeled.png"]
        overlay_names = [f"{image_name}_pred_overlay.png", f"{image_name}_overlay.png"]
        summary_names = []

    labeled = find_existing_visual([folder / name for folder in search_dirs for name in labeled_names])
    overlay = find_existing_visual([folder / name for folder in search_dirs for name in overlay_names])
    summary = find_existing_visual([folder / name for folder in search_dirs for name in summary_names])
    return {"labeled": labeled, "overlay": overlay, "summary": summary}


def copy_visual_asset(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None:
        return None
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(src)


def summarize_class_counts(label_map: np.ndarray, ignore_index: int, top_k: int = 8) -> List[Dict[str, int]]:
    valid = label_map[label_map != ignore_index]
    if valid.size == 0:
        return []

    class_ids, counts = np.unique(valid, return_counts=True)
    order = np.argsort(-counts)
    rows: List[Dict[str, int]] = []
    for idx in order[:top_k]:
        rows.append({"class_id": int(class_ids[idx]), "pixel_count": int(counts[idx])})
    return rows


def resolve_num_workers(requested_workers: int) -> int:
    if requested_workers > 0:
        return requested_workers
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def init_image_worker(
    similarity_matrix: np.ndarray,
    num_classes: int,
    ignore_index: int,
    gt_offset: int,
    pred_offset: int,
    pred_collapse_ge_to: Optional[int],
    pred_collapse_target: Optional[int],
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
    use_dense_confusions: bool,
) -> None:
    global IMAGE_WORKER_CONTEXT
    IMAGE_WORKER_CONTEXT = {
        "similarity_matrix": np.asarray(similarity_matrix, dtype=np.float64),
        "num_classes": int(num_classes),
        "ignore_index": int(ignore_index),
        "gt_offset": int(gt_offset),
        "pred_offset": int(pred_offset),
        "pred_collapse_ge_to": None if pred_collapse_ge_to is None else int(pred_collapse_ge_to),
        "pred_collapse_target": None if pred_collapse_target is None else int(pred_collapse_target),
        "max_radius": int(max_radius),
        "boundary_width": int(boundary_width),
        "kernel_type": str(kernel_type),
        "use_dense_confusions": bool(use_dense_confusions),
    }


def evaluate_image_pair(
    pair: MatchedPair,
    similarity_matrix: np.ndarray,
    num_classes: int,
    ignore_index: int,
    gt_offset: int,
    pred_offset: int,
    pred_collapse_ge_to: Optional[int],
    pred_collapse_target: Optional[int],
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
    use_dense_confusions: bool,
) -> Dict[str, Any]:
    gt = load_label_map(pair.gt_path, offset=gt_offset, ignore_index=ignore_index)
    pred = load_label_map(
        pair.pred_path,
        offset=pred_offset,
        ignore_index=ignore_index,
        collapse_ge_to=pred_collapse_ge_to,
        collapse_target=pred_collapse_target,
    )

    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    confusion = compute_confusion_matrix(gt, pred, num_classes, ignore_index)
    miou_result = compute_miou_from_confusion(confusion)
    hier_result = compute_hier_soft_iou_from_confusion(confusion, similarity_matrix)
    struct_result = compute_multiclass_struct_score(
        gt=gt,
        pred=pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_radius=max_radius,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )

    conf_indices, conf_counts = confusion_to_sparse(confusion)
    result: Dict[str, Any] = {
        "image_id": pair.image_id,
        "miou": float(miou_result["miou"]),
        "hieriou": float(hier_result["hier_miou"]),
        "struct": float(struct_result["image_score"]),
        "persist": float(struct_result["image_persist_mean"]),
        "edge": float(struct_result["image_edge_mean"]),
        "num_valid_classes": int(len(struct_result["valid_classes"])),
        "conf_indices": conf_indices,
        "conf_counts": conf_counts,
        "struct_result": struct_result,
    }
    if use_dense_confusions:
        result["confusion_flat"] = confusion.reshape(-1).astype(np.uint32)
    return result


def evaluate_image_pair_worker(pair: MatchedPair) -> Dict[str, Any]:
    ctx = IMAGE_WORKER_CONTEXT
    try:
        payload = evaluate_image_pair(
            pair=pair,
            similarity_matrix=ctx["similarity_matrix"],
            num_classes=ctx["num_classes"],
            ignore_index=ctx["ignore_index"],
            gt_offset=ctx["gt_offset"],
            pred_offset=ctx["pred_offset"],
            pred_collapse_ge_to=ctx["pred_collapse_ge_to"],
            pred_collapse_target=ctx["pred_collapse_target"],
            max_radius=ctx["max_radius"],
            boundary_width=ctx["boundary_width"],
            kernel_type=ctx["kernel_type"],
            use_dense_confusions=ctx["use_dense_confusions"],
        )
        payload["ok"] = True
        return payload
    except Exception as exc:  # pragma: no cover - worker safety path
        return {
            "ok": False,
            "image_id": pair.image_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def confusion_to_sparse(confusion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = confusion.reshape(-1)
    indices = np.nonzero(flat)[0].astype(np.int64)
    counts = flat[indices].astype(np.int64)
    return indices, counts


def sparse_to_dense_confusion(
    indices: np.ndarray,
    counts: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    confusion = np.zeros((num_classes * num_classes,), dtype=np.int64)
    confusion[indices] = counts
    return confusion.reshape(num_classes, num_classes)


def compute_dataset_miou(confusion: np.ndarray) -> Dict[str, Any]:
    """
    Match Detectron2 SemSegEvaluator's mIoU semantics.

    For the paper-facing semantic segmentation numbers, the mean should be
    taken over classes that appear in GT, rather than over every class whose
    union is non-zero. This matters especially for long-tail datasets such as
    Pascal Context 459, where predicted-only classes can otherwise depress the
    score substantially relative to the official baseline numbers.
    """
    tp = np.diag(confusion).astype(np.float64)
    pos_gt = confusion.sum(axis=1).astype(np.float64)
    pos_pred = confusion.sum(axis=0).astype(np.float64)
    union = pos_gt + pos_pred - tp

    valid_classes = pos_gt > 0
    per_class_iou = np.full(confusion.shape[0], np.nan, dtype=np.float64)
    nonzero_union = valid_classes & (union > 0)
    per_class_iou[nonzero_union] = tp[nonzero_union] / union[nonzero_union]

    miou = float(np.nanmean(per_class_iou[valid_classes])) if np.any(valid_classes) else 0.0
    fp = pos_pred - tp
    fn = pos_gt - tp
    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "valid_classes": valid_classes,
        "num_valid_classes": int(valid_classes.sum()),
        "num_union_valid_classes": int((union > 0).sum()),
    }


def compute_dataset_hieriou(confusion: np.ndarray, similarity_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Match the dataset-level averaging semantics used for ``compute_dataset_miou``.

    ``compute_hier_soft_iou_from_confusion`` averages over every class whose soft
    denominator is non-zero. For long-tail datasets such as PC-459, that can pull
    in many prediction-only classes and make dataset-level Open-mIoU look
    artificially worse than the official mIoU, even when every GT-present class
    improves under the soft similarity matrix.

    For paper-facing dataset summaries we therefore average Open-mIoU over the
    same class set as mIoU: classes that appear in GT at least once.
    """
    hier_result = compute_hier_soft_iou_from_confusion(confusion, similarity_matrix)
    pos_gt = confusion.sum(axis=1).astype(np.float64)
    valid_classes = pos_gt > 0

    per_class_iou = np.asarray(hier_result["per_class_iou"], dtype=np.float64)
    hier_miou = float(np.nanmean(per_class_iou[valid_classes])) if np.any(valid_classes) else 0.0

    out = dict(hier_result)
    out["hier_miou"] = hier_miou
    out["valid_classes"] = valid_classes
    out["num_valid_classes"] = int(valid_classes.sum())
    out["soft_valid_classes"] = np.asarray(hier_result["valid_classes"], dtype=bool)
    out["num_soft_valid_classes"] = int(np.asarray(hier_result["valid_classes"], dtype=bool).sum())
    return out


def compute_dataset_structscore(
    per_image_struct_results: Sequence[Dict[str, Any]],
    num_classes: int,
) -> Dict[str, Dict[str, Any]]:
    struct_results = list(per_image_struct_results)
    image_mean = _aggregate_struct_dataset_results(struct_results, num_classes, "image_mean")
    class_mean = _aggregate_struct_dataset_results(struct_results, num_classes, "class_mean")
    if struct_results:
        max_radius = np.asarray(struct_results[0]["per_class_persist_inter"]).shape[1] - 1
        persist_inter_sum = np.zeros((num_classes, max_radius + 1), dtype=np.uint64)
        persist_union_sum = np.zeros((num_classes, max_radius + 1), dtype=np.uint64)
        edge_inter_sum = np.zeros((num_classes,), dtype=np.uint64)
        edge_union_sum = np.zeros((num_classes,), dtype=np.uint64)
        for result in struct_results:
            persist_inter_sum += np.asarray(result["per_class_persist_inter"], dtype=np.uint64)
            persist_union_sum += np.asarray(result["per_class_persist_union"], dtype=np.uint64)
            edge_inter_sum += np.asarray(result["per_class_edge_inter"], dtype=np.uint64)
            edge_union_sum += np.asarray(result["per_class_edge_union"], dtype=np.uint64)
        miou_style = _aggregate_struct_counts(persist_inter_sum, persist_union_sum, edge_inter_sum, edge_union_sum)
    else:
        miou_style = _aggregate_struct_counts(
            np.zeros((num_classes, 1), dtype=np.uint64),
            np.zeros((num_classes, 1), dtype=np.uint64),
            np.zeros((num_classes,), dtype=np.uint64),
            np.zeros((num_classes,), dtype=np.uint64),
        )
    return {
        "image_mean": image_mean,
        "class_mean": class_mean,
        "miou_style": miou_style,
    }


def bootstrap_metric(
    metric_name: str,
    image_entries: Sequence[Dict[str, Any]],
    num_classes: int,
    similarity_matrix: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    use_dense_confusions: bool = True,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    if not image_entries:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_std": float("nan"),
        }

    rng = np.random.default_rng(seed)
    n_images = len(image_entries)
    alpha = 1.0 - confidence_level

    if metric_name == "miou":
        observed = float(
            compute_dataset_miou(
                sum_dense_confusions(image_entries, num_classes, use_dense_confusions=use_dense_confusions)
            )["miou"]
        )
    elif metric_name == "hieriou":
        observed = float(
            compute_dataset_hieriou(
                sum_dense_confusions(image_entries, num_classes, use_dense_confusions=use_dense_confusions),
                similarity_matrix,
            )["hier_miou"]
        )
    elif metric_name == "struct":
        observed = float(
            compute_dataset_structscore(
                [entry["struct_result"] for entry in image_entries],
                num_classes=num_classes,
            )["miou_style"]["score"]
        )
    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    if n_images == 1:
        return {
            "mean": observed,
            "ci_lower": observed,
            "ci_upper": observed,
            "bootstrap_std": 0.0,
        }

    bootstrap_values = np.zeros((n_bootstrap,), dtype=np.float64)

    dense_stack: Optional[np.ndarray] = None
    if use_dense_confusions and metric_name in {"miou", "hieriou"}:
        dense_stack = np.stack([entry["confusion_flat"] for entry in image_entries], axis=0)

    persist_inter_stack = None
    persist_union_stack = None
    edge_inter_stack = None
    edge_union_stack = None
    if metric_name == "struct":
        persist_inter_stack = np.stack(
            [np.asarray(entry["struct_result"]["per_class_persist_inter"], dtype=np.uint32) for entry in image_entries],
            axis=0,
        )
        persist_union_stack = np.stack(
            [np.asarray(entry["struct_result"]["per_class_persist_union"], dtype=np.uint32) for entry in image_entries],
            axis=0,
        )
        edge_inter_stack = np.stack(
            [np.asarray(entry["struct_result"]["per_class_edge_inter"], dtype=np.uint32) for entry in image_entries],
            axis=0,
        )
        edge_union_stack = np.stack(
            [np.asarray(entry["struct_result"]["per_class_edge_union"], dtype=np.uint32) for entry in image_entries],
            axis=0,
        )

    progress_bar = tqdm(
        range(n_bootstrap),
        total=n_bootstrap,
        desc=progress_desc or f"bootstrap:{metric_name}",
        unit="iter",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for i in progress_bar:
        sampled_indices = rng.integers(0, n_images, size=n_images)
        if metric_name == "struct":
            assert persist_inter_stack is not None
            assert persist_union_stack is not None
            assert edge_inter_stack is not None
            assert edge_union_stack is not None
            sample_weights = np.bincount(sampled_indices, minlength=n_images).astype(np.uint64, copy=False)
            persist_inter_sum = np.tensordot(sample_weights, persist_inter_stack, axes=(0, 0))
            persist_union_sum = np.tensordot(sample_weights, persist_union_stack, axes=(0, 0))
            edge_inter_sum = np.tensordot(sample_weights, edge_inter_stack, axes=(0, 0))
            edge_union_sum = np.tensordot(sample_weights, edge_union_stack, axes=(0, 0))
            bootstrap_values[i] = float(
                _aggregate_struct_counts(
                    np.asarray(persist_inter_sum, dtype=np.uint64),
                    np.asarray(persist_union_sum, dtype=np.uint64),
                    np.asarray(edge_inter_sum, dtype=np.uint64),
                    np.asarray(edge_union_sum, dtype=np.uint64),
                )["score"]
            )
            continue

        if dense_stack is not None:
            sample_weights = np.bincount(sampled_indices, minlength=n_images).astype(np.int64, copy=False)
            sampled_confusion = (sample_weights @ dense_stack).reshape(num_classes, num_classes)
        else:
            sampled_confusion = np.zeros((num_classes * num_classes,), dtype=np.int64)
            for idx in sampled_indices:
                conf_idx = image_entries[idx]["conf_indices"]
                conf_counts = image_entries[idx]["conf_counts"]
                sampled_confusion[conf_idx] += conf_counts
            sampled_confusion = sampled_confusion.reshape(num_classes, num_classes)

        if metric_name == "miou":
            bootstrap_values[i] = float(compute_dataset_miou(sampled_confusion)["miou"])
        else:
            bootstrap_values[i] = float(compute_dataset_hieriou(sampled_confusion, similarity_matrix)["hier_miou"])

    progress_bar.close()

    return {
        "mean": observed,
        "ci_lower": float(np.percentile(bootstrap_values, 100.0 * alpha / 2.0)),
        "ci_upper": float(np.percentile(bootstrap_values, 100.0 * (1.0 - alpha / 2.0))),
        "bootstrap_std": float(np.std(bootstrap_values)),
    }


def sum_dense_confusions(
    image_entries: Sequence[Dict[str, Any]],
    num_classes: int,
    use_dense_confusions: bool = True,
) -> np.ndarray:
    if use_dense_confusions:
        dense_stack = np.stack([entry["confusion_flat"] for entry in image_entries], axis=0)
        return dense_stack.sum(axis=0, dtype=np.int64).reshape(num_classes, num_classes)

    total = np.zeros((num_classes * num_classes,), dtype=np.int64)
    for entry in image_entries:
        total[entry["conf_indices"]] += entry["conf_counts"]
    return total.reshape(num_classes, num_classes)


def evaluate_single_prediction_dir(
    gt_dir: Path,
    pred_dir: Path,
    pred_name: str,
    num_classes: int,
    ignore_index: int,
    similarity_matrix: np.ndarray,
    bootstrap_iters: int,
    bootstrap_seed: int,
    gt_offset: int,
    pred_offset: int,
    pred_collapse_ge_to: Optional[int],
    pred_collapse_target: Optional[int],
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
    max_images: int,
    num_workers: int,
    show_progress: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Evaluating {pred_name}")
    print("=" * 80)

    pairs = match_gt_pred_files(gt_dir, pred_dir, warnings)
    if not pairs:
        warn(f"No matched image pairs for {pred_name}", warnings)
        return {
            "pred_name": pred_name,
            "pred_dir": pred_dir,
            "matched_pairs": [],
            "image_entries": [],
            "image_rows": [],
            "per_class_rows": [],
            "overall_row": None,
            "bootstrap": {},
            "warnings": [],
        }

    if max_images > 0 and len(pairs) > max_images:
        print(f"[{pred_name}] limiting evaluation to first {max_images}/{len(pairs)} matched images")
        pairs = pairs[:max_images]

    effective_workers = resolve_num_workers(num_workers)
    if effective_workers > len(pairs):
        effective_workers = len(pairs)
    if effective_workers < 1:
        effective_workers = 1

    use_dense_confusions = (len(pairs) * num_classes * num_classes) <= 80_000_000
    if use_dense_confusions:
        print(f"[{pred_name}] using dense per-image confusion cache for bootstrap.")
    else:
        print(f"[{pred_name}] using sparse per-image confusion cache for bootstrap.")
    if effective_workers > 1:
        print(f"[{pred_name}] evaluating image pairs with {effective_workers} worker processes.")
    else:
        print(f"[{pred_name}] evaluating image pairs in the main process.")

    image_entries: List[Dict[str, Any]] = []
    per_image_struct_results: List[Dict[str, Any]] = []
    per_image_rows: List[Dict[str, Any]] = []
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    model_warnings: List[str] = []
    image_progress_desc = f"{pred_name} images"

    def process_pair_result(pair: MatchedPair, payload: Dict[str, Any]) -> None:
        if not payload.get("ok", True):
            message = (
                f"{pred_name}: failed on '{pair.image_id}' with "
                f"{payload.get('error_type', 'Error')}: {payload.get('error_message', 'unknown error')}. Skipped."
            )
            warn(message, warnings)
            model_warnings.append(message)
            worker_trace = payload.get("traceback")
            if worker_trace:
                print(worker_trace, file=sys.stderr)
            return

        if use_dense_confusions:
            confusion = np.asarray(payload["confusion_flat"], dtype=np.int64).reshape(num_classes, num_classes)
        else:
            confusion = sparse_to_dense_confusion(
                np.asarray(payload["conf_indices"], dtype=np.int64),
                np.asarray(payload["conf_counts"], dtype=np.int64),
                num_classes,
            )

        total_confusion[...] += confusion

        entry = {
            "pred_name": pred_name,
            "image_id": pair.image_id,
            "pair": pair,
            "miou": float(payload["miou"]),
            "hieriou": float(payload["hieriou"]),
            "struct": float(payload["struct"]),
            "persist": float(payload["persist"]),
            "edge": float(payload["edge"]),
            "num_valid_classes": int(payload["num_valid_classes"]),
            "conf_indices": np.asarray(payload["conf_indices"], dtype=np.int64),
            "conf_counts": np.asarray(payload["conf_counts"], dtype=np.int64),
            "struct_result": payload["struct_result"],
        }
        if use_dense_confusions:
            entry["confusion_flat"] = np.asarray(payload["confusion_flat"], dtype=np.uint32)

        image_entries.append(entry)
        per_image_struct_results.append(payload["struct_result"])
        per_image_rows.append(
            {
                "pred_name": pred_name,
                "image_id": pair.image_id,
                "miou": float(payload["miou"]),
                "hieriou": float(payload["hieriou"]),
                "struct": float(payload["struct"]),
                "persist": float(payload["persist"]),
                "edge": float(payload["edge"]),
                "num_valid_classes": int(payload["num_valid_classes"]),
                "gt_rel": pair.gt_rel,
                "pred_rel": pair.pred_rel,
            }
        )

    if effective_workers == 1:
        pair_iterator = tqdm(
            pairs,
            total=len(pairs),
            desc=image_progress_desc,
            unit="img",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for idx, pair in enumerate(pair_iterator, start=1):
            if (not show_progress) and (idx == 1 or idx % 50 == 0 or idx == len(pairs)):
                print(f"[{pred_name}] processed {idx - 1}/{len(pairs)} images")
            try:
                payload = evaluate_image_pair(
                    pair=pair,
                    similarity_matrix=similarity_matrix,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    gt_offset=gt_offset,
                    pred_offset=pred_offset,
                    pred_collapse_ge_to=pred_collapse_ge_to,
                    pred_collapse_target=pred_collapse_target,
                    max_radius=max_radius,
                    boundary_width=boundary_width,
                    kernel_type=kernel_type,
                    use_dense_confusions=use_dense_confusions,
                )
                payload["ok"] = True
            except Exception as exc:  # pragma: no cover - safety path
                payload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            process_pair_result(pair, payload)
        pair_iterator.close()
    else:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=init_image_worker,
            initargs=(
                similarity_matrix,
                num_classes,
                ignore_index,
                gt_offset,
                pred_offset,
                pred_collapse_ge_to,
                pred_collapse_target,
                max_radius,
                boundary_width,
                kernel_type,
                use_dense_confusions,
            ),
        ) as executor:
            future_to_index = {
                executor.submit(evaluate_image_pair_worker, pair): idx for idx, pair in enumerate(pairs)
            }
            ordered_payloads: List[Optional[Dict[str, Any]]] = [None] * len(pairs)
            completion_bar = tqdm(
                total=len(pairs),
                desc=image_progress_desc,
                unit="img",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
            )
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                ordered_payloads[idx] = future.result()
                completion_bar.update(1)
            completion_bar.close()

        for idx, (pair, payload) in enumerate(zip(pairs, ordered_payloads), start=1):
            if payload is None:
                payload = {
                    "ok": False,
                    "error_type": "MissingResult",
                    "error_message": "worker did not return a payload",
                }
            if (not show_progress) and (idx == 1 or idx % 50 == 0 or idx == len(pairs)):
                print(f"[{pred_name}] processed {idx - 1}/{len(pairs)} images")
            process_pair_result(pair, payload)

    if not image_entries:
        warn(f"No valid image evaluations completed for {pred_name}", warnings)
        return {
            "pred_name": pred_name,
            "pred_dir": pred_dir,
            "matched_pairs": pairs,
            "image_entries": [],
            "image_rows": [],
            "per_class_rows": [],
            "overall_row": None,
            "bootstrap": {},
            "warnings": model_warnings,
        }

    dataset_miou = compute_dataset_miou(total_confusion)
    dataset_hieriou = compute_dataset_hieriou(total_confusion, similarity_matrix)
    dataset_struct = compute_dataset_structscore(per_image_struct_results, num_classes)

    bootstrap_miou = bootstrap_metric(
        "miou",
        image_entries,
        num_classes=num_classes,
        similarity_matrix=similarity_matrix,
        n_bootstrap=bootstrap_iters,
        seed=bootstrap_seed,
        use_dense_confusions=use_dense_confusions,
        show_progress=show_progress,
        progress_desc=f"{pred_name} bootstrap mIoU",
    )
    bootstrap_hieriou = bootstrap_metric(
        "hieriou",
        image_entries,
        num_classes=num_classes,
        similarity_matrix=similarity_matrix,
        n_bootstrap=bootstrap_iters,
        seed=bootstrap_seed + 1,
        use_dense_confusions=use_dense_confusions,
        show_progress=show_progress,
        progress_desc=f"{pred_name} bootstrap hierIoU",
    )
    bootstrap_struct = bootstrap_metric(
        "struct",
        image_entries,
        num_classes=num_classes,
        similarity_matrix=similarity_matrix,
        n_bootstrap=bootstrap_iters,
        seed=bootstrap_seed + 2,
        use_dense_confusions=use_dense_confusions,
        show_progress=show_progress,
        progress_desc=f"{pred_name} bootstrap Struct",
    )

    overall_row = {
        "pred_name": pred_name,
        "pred_dir": str(pred_dir),
        "num_matched_images": len(pairs),
        "num_evaluated_images": len(image_entries),
        "num_skipped_images": len(pairs) - len(image_entries),
        "miou_mean": float(dataset_miou["miou"]),
        "miou_ci_low": bootstrap_miou["ci_lower"],
        "miou_ci_high": bootstrap_miou["ci_upper"],
        "hieriou_mean": float(dataset_hieriou["hier_miou"]),
        "hieriou_ci_low": bootstrap_hieriou["ci_lower"],
        "hieriou_ci_high": bootstrap_hieriou["ci_upper"],
        "struct_mean": float(dataset_struct["miou_style"]["score"]),
        "struct_ci_low": bootstrap_struct["ci_lower"],
        "struct_ci_high": bootstrap_struct["ci_upper"],
        "persist_mean": float(dataset_struct["miou_style"]["persist_score"]),
        "edge_mean": float(dataset_struct["miou_style"]["edge_score"]),
        "struct_image_mean": float(dataset_struct["image_mean"]["score"]),
        "persist_image_mean": float(dataset_struct["image_mean"]["persist_score"]),
        "edge_image_mean": float(dataset_struct["image_mean"]["edge_score"]),
        "struct_class_mean": float(dataset_struct["class_mean"]["score"]),
        "persist_class_mean": float(dataset_struct["class_mean"]["persist_score"]),
        "edge_class_mean": float(dataset_struct["class_mean"]["edge_score"]),
    }

    per_class_rows = aggregate_per_class_results(
        pred_name=pred_name,
        miou_result=dataset_miou,
        hieriou_result=dataset_hieriou,
        struct_result=dataset_struct["miou_style"],
    )

    return {
        "pred_name": pred_name,
        "pred_dir": pred_dir,
        "matched_pairs": pairs,
        "image_entries": image_entries,
        "image_rows": per_image_rows,
        "per_class_rows": per_class_rows,
        "overall_row": overall_row,
        "bootstrap": {
            "miou": bootstrap_miou,
            "hieriou": bootstrap_hieriou,
            "struct": bootstrap_struct,
        },
        "dataset_miou": dataset_miou,
        "dataset_hieriou": dataset_hieriou,
        "dataset_struct": dataset_struct,
        "warnings": model_warnings,
    }


def aggregate_per_class_results(
    pred_name: str,
    miou_result: Dict[str, Any],
    hieriou_result: Dict[str, Any],
    struct_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    iou = np.asarray(miou_result["per_class_iou"], dtype=np.float64)
    hieriou = np.asarray(hieriou_result["per_class_iou"], dtype=np.float64)
    persist = np.asarray(struct_result["per_class_persist_mean"], dtype=np.float64)
    edge = np.asarray(struct_result["per_class_edge_mean"], dtype=np.float64)
    struct = np.asarray(struct_result["per_class_mean"], dtype=np.float64)

    rows: List[Dict[str, Any]] = []
    for class_id in range(len(iou)):
        rows.append(
            {
                "pred_name": pred_name,
                "class_id": class_id,
                "iou": float(iou[class_id]) if np.isfinite(iou[class_id]) else float("nan"),
                "hieriou": float(hieriou[class_id]) if np.isfinite(hieriou[class_id]) else float("nan"),
                "persist": float(persist[class_id]) if np.isfinite(persist[class_id]) else float("nan"),
                "edge": float(edge[class_id]) if np.isfinite(edge[class_id]) else float("nan"),
                "struct": float(struct[class_id]) if np.isfinite(struct[class_id]) else float("nan"),
            }
        )
    return rows


def _assign_ranks(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    ordered = sorted(rows, key=lambda x: (-to_float_or_nan(x[key]), str(x["pred_name"])))
    return {row["pred_name"]: rank for rank, row in enumerate(ordered, start=1)}


def make_ranking_table(overall_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not overall_rows:
        return []

    miou_ranks = _assign_ranks(overall_rows, "miou_mean")
    hier_ranks = _assign_ranks(overall_rows, "hieriou_mean")
    struct_ranks = _assign_ranks(overall_rows, "struct_mean")

    ranking_rows = []
    for row in overall_rows:
        pred_name = row["pred_name"]
        rank_row = {
            "pred_name": pred_name,
            "rank_miou": miou_ranks[pred_name],
            "rank_hieriou": hier_ranks[pred_name],
            "rank_struct": struct_ranks[pred_name],
            "miou_mean": row["miou_mean"],
            "hieriou_mean": row["hieriou_mean"],
            "struct_mean": row["struct_mean"],
            "rank_gap_struct_vs_miou": struct_ranks[pred_name] - miou_ranks[pred_name],
            "rank_gap_struct_vs_hieriou": struct_ranks[pred_name] - hier_ranks[pred_name],
            "rank_gap_hieriou_vs_miou": hier_ranks[pred_name] - miou_ranks[pred_name],
        }
        ranking_rows.append(rank_row)

    return sorted(ranking_rows, key=lambda x: x["rank_struct"])


def build_pairwise_comparison_table(overall_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    if len(overall_rows) < 2:
        return rows

    miou_gaps = []
    struct_gaps = []
    for i in range(len(overall_rows)):
        for j in range(i + 1, len(overall_rows)):
            miou_gaps.append(abs(overall_rows[i]["miou_mean"] - overall_rows[j]["miou_mean"]))
            struct_gaps.append(abs(overall_rows[i]["struct_mean"] - overall_rows[j]["struct_mean"]))

    close_miou_threshold = float(np.percentile(miou_gaps, 25)) if miou_gaps else 0.0
    far_struct_threshold = float(np.percentile(struct_gaps, 75)) if struct_gaps else 0.0

    for i in range(len(overall_rows)):
        for j in range(i + 1, len(overall_rows)):
            a = overall_rows[i]
            b = overall_rows[j]
            miou_gap = float(a["miou_mean"] - b["miou_mean"])
            hieriou_gap = float(a["hieriou_mean"] - b["hieriou_mean"])
            struct_gap = float(a["struct_mean"] - b["struct_mean"])
            rows.append(
                {
                    "pred_a": a["pred_name"],
                    "pred_b": b["pred_name"],
                    "miou_gap": miou_gap,
                    "hieriou_gap": hieriou_gap,
                    "struct_gap": struct_gap,
                    "abs_miou_gap": abs(miou_gap),
                    "abs_hieriou_gap": abs(hieriou_gap),
                    "abs_struct_gap": abs(struct_gap),
                    "close_miou_far_struct": bool(
                        abs(miou_gap) <= close_miou_threshold and abs(struct_gap) >= far_struct_threshold
                    ),
                    "higher_hieriou_lower_struct_conflict": bool(
                        (hieriou_gap > 0.0 and struct_gap < 0.0)
                        or (hieriou_gap < 0.0 and struct_gap > 0.0)
                    ),
                }
            )

    return sorted(rows, key=lambda x: (-x["abs_struct_gap"], x["abs_miou_gap"]))


def plot_directory_level_scatter(overall_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not overall_rows:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    x = [row["hieriou_mean"] for row in overall_rows]
    y = [row["struct_mean"] for row in overall_rows]
    ax.scatter(x, y)

    for row in overall_rows:
        ax.annotate(row["pred_name"], (row["hieriou_mean"], row["struct_mean"]), fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("hierIoU")
    ax.set_ylabel("StructScore")
    ax.set_title("Directory-level hierIoU vs StructScore")
    ax.grid(True, alpha=0.3)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_image_level_scatter(image_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not image_rows:
        return

    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    pred_names = sorted({row["pred_name"] for row in image_rows})
    for pred_name in pred_names:
        subset = [row for row in image_rows if row["pred_name"] == pred_name]
        ax.scatter(
            [row["hieriou"] for row in subset],
            [row["struct"] for row in subset],
            s=18,
            alpha=0.7,
            label=pred_name,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("hierIoU")
    ax.set_ylabel("StructScore")
    ax.set_title("Image-level hierIoU vs StructScore")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "image_level_scatter_all.png", dpi=150)
    plt.close(fig)

    for pred_name in pred_names:
        subset = [row for row in image_rows if row["pred_name"] == pred_name]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            [row["hieriou"] for row in subset],
            [row["struct"] for row in subset],
            s=18,
            alpha=0.75,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("hierIoU")
        ax.set_ylabel("StructScore")
        ax.set_title(f"Image-level hierIoU vs StructScore: {pred_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"image_level_scatter_{safe_slug(pred_name)}.png", dpi=150)
        plt.close(fig)


def plot_per_class_scatter(per_class_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    if not per_class_rows:
        return

    ensure_dir(out_dir)
    pred_names = sorted({row["pred_name"] for row in per_class_rows})
    for pred_name in pred_names:
        subset = [
            row
            for row in per_class_rows
            if row["pred_name"] == pred_name and np.isfinite(row["hieriou"]) and np.isfinite(row["struct"])
        ]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            [row["hieriou"] for row in subset],
            [row["struct"] for row in subset],
            s=22,
            alpha=0.75,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Per-class hierIoU")
        ax.set_ylabel("Per-class StructScore")
        ax.set_title(f"Per-class hierIoU vs StructScore: {pred_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_scatter_{safe_slug(pred_name)}.png", dpi=150)
        plt.close(fig)


def plot_overall_metric_bars(overall_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not overall_rows:
        return

    pred_names = [row["pred_name"] for row in overall_rows]
    x = np.arange(len(pred_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(pred_names) * 1.5), 6))
    ax.bar(x - width, [row["miou_mean"] for row in overall_rows], width=width, label="mIoU")
    ax.bar(x, [row["hieriou_mean"] for row in overall_rows], width=width, label="hierIoU")
    ax.bar(x + width, [row["struct_mean"] for row in overall_rows], width=width, label="StructScore")
    ax.set_xticks(x)
    ax.set_xticklabels(pred_names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Prediction directory")
    ax.set_ylabel("Score")
    ax.set_title("Overall scores across prediction directories")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_comparison(ranking_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if not ranking_rows:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(ranking_rows) * 1.3), 6))
    x = np.arange(len(ranking_rows))
    ax.plot(x, [row["rank_miou"] for row in ranking_rows], marker="o", label="mIoU rank")
    ax.plot(x, [row["rank_hieriou"] for row in ranking_rows], marker="o", label="hierIoU rank")
    ax.plot(x, [row["rank_struct"] for row in ranking_rows], marker="o", label="StructScore rank")
    ax.set_xticks(x)
    ax.set_xticklabels([row["pred_name"] for row in ranking_rows], rotation=30, ha="right")
    ax.invert_yaxis()
    ax.set_xlabel("Prediction directory")
    ax.set_ylabel("Rank (1 is best)")
    ax.set_title("Rank comparison across metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def select_case_studies(
    image_entries: Sequence[Dict[str, Any]],
    case_quantile: float = 0.2,
    max_cases_per_bucket: int = 5,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
    if not image_entries:
        return {}, {}

    hier_values = np.asarray([entry["hieriou"] for entry in image_entries], dtype=np.float64)
    struct_values = np.asarray([entry["struct"] for entry in image_entries], dtype=np.float64)

    thresholds = {
        "hier_low": float(np.quantile(hier_values, case_quantile)),
        "hier_high": float(np.quantile(hier_values, 1.0 - case_quantile)),
        "struct_low": float(np.quantile(struct_values, case_quantile)),
        "struct_high": float(np.quantile(struct_values, 1.0 - case_quantile)),
    }

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "semantic_high_structure_high": [],
        "semantic_high_structure_low": [],
        "semantic_low_structure_high": [],
        "semantic_low_structure_low": [],
    }

    for entry in image_entries:
        high_hier = entry["hieriou"] >= thresholds["hier_high"]
        low_hier = entry["hieriou"] <= thresholds["hier_low"]
        high_struct = entry["struct"] >= thresholds["struct_high"]
        low_struct = entry["struct"] <= thresholds["struct_low"]

        if high_hier and high_struct:
            buckets["semantic_high_structure_high"].append(entry)
        if high_hier and low_struct:
            buckets["semantic_high_structure_low"].append(entry)
        if low_hier and high_struct:
            buckets["semantic_low_structure_high"].append(entry)
        if low_hier and low_struct:
            buckets["semantic_low_structure_low"].append(entry)

    def sort_entries(bucket_name: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if bucket_name == "semantic_high_structure_high":
            return sorted(entries, key=lambda x: (x["hieriou"] + x["struct"], x["miou"]), reverse=True)
        if bucket_name == "semantic_high_structure_low":
            return sorted(
                entries,
                key=lambda x: (x["hieriou"] - x["struct"], x["hieriou"], -x["struct"]),
                reverse=True,
            )
        if bucket_name == "semantic_low_structure_high":
            return sorted(
                entries,
                key=lambda x: (x["struct"] - x["hieriou"], x["struct"], -x["hieriou"]),
                reverse=True,
            )
        return sorted(entries, key=lambda x: (x["hieriou"] + x["struct"], x["miou"]))

    selected = {}
    for bucket_name, entries in buckets.items():
        selected[bucket_name] = sort_entries(bucket_name, entries)[:max_cases_per_bucket]

    return selected, thresholds


def save_raw_label_map(label_map: np.ndarray, out_prefix: Path, png_name_suffix: str = "") -> None:
    ensure_dir(out_prefix.parent)
    np.save(str(out_prefix.with_suffix(".npy")), label_map.astype(np.int32))
    if label_map.min() >= 0 and label_map.max() <= 65535:
        png_path = out_prefix.with_name(f"{out_prefix.name}{png_name_suffix}").with_suffix(".png")
        Image.fromarray(label_map.astype(np.uint16)).save(png_path)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_case_visualizations(
    pred_name: str,
    selected_cases: Dict[str, List[Dict[str, Any]]],
    thresholds: Dict[str, float],
    similarity_matrix: np.ndarray,
    num_classes: int,
    ignore_index: int,
    gt_offset: int,
    pred_offset: int,
    pred_collapse_ge_to: Optional[int],
    pred_collapse_target: Optional[int],
    case_gt_visual_dir: Optional[Path],
    out_dir: Path,
    max_case_masks: int = 3,
) -> List[Dict[str, Any]]:
    ensure_dir(out_dir)
    summary_rows: List[Dict[str, Any]] = []

    for bucket_name, entries in selected_cases.items():
        for rank, entry in enumerate(entries, start=1):
            pair: MatchedPair = entry["pair"]
            case_dir = ensure_dir(out_dir / safe_slug(pred_name) / bucket_name / f"{rank:02d}_{safe_slug(pair.image_id)}")

            gt = load_label_map(pair.gt_path, offset=gt_offset, ignore_index=ignore_index)
            pred = load_label_map(
                pair.pred_path,
                offset=pred_offset,
                ignore_index=ignore_index,
                collapse_ge_to=pred_collapse_ge_to,
                collapse_target=pred_collapse_target,
            )
            confusion = sparse_to_dense_confusion(entry["conf_indices"], entry["conf_counts"], num_classes)
            hier_result = compute_hier_soft_iou_from_confusion(confusion, similarity_matrix)
            struct_result = entry["struct_result"]
            gt_color = colorize_label_map(gt, num_classes=num_classes, ignore_index=ignore_index)
            pred_color = colorize_label_map(pred, num_classes=num_classes, ignore_index=ignore_index)
            agreement_map = build_agreement_map(gt, pred, ignore_index=ignore_index)

            gt_visuals = resolve_case_visual_paths(
                image_id=pair.image_id,
                role="gt",
                primary_path=pair.gt_path,
                extra_dirs=[case_gt_visual_dir, DEFAULT_CASE_GT_VIS_DIR],
            )
            pred_visuals = resolve_case_visual_paths(
                image_id=pair.image_id,
                role="pred",
                primary_path=pair.pred_path,
                extra_dirs=[],
            )

            gt_valid_pixels = int((gt != ignore_index).sum())
            gt_ignore_pixels = int((gt == ignore_index).sum())
            valid_diff_pixels = int(((gt != pred) & (gt != ignore_index)).sum())
            ignore_diff_pixels = int(((gt != pred) & (gt == ignore_index)).sum())
            gt_class_counts = summarize_class_counts(gt, ignore_index=ignore_index)
            pred_class_counts = summarize_class_counts(pred, ignore_index=ignore_index)

            save_raw_label_map(gt, case_dir / "gt_label", png_name_suffix="_raw")
            save_raw_label_map(pred, case_dir / "pred_label", png_name_suffix="_raw")
            save_rgb_image(gt_color, case_dir / "gt_label_color.png")
            save_rgb_image(pred_color, case_dir / "pred_label_color.png")
            save_rgb_image(agreement_map, case_dir / "agreement_map.png")

            gt_labeled_src = copy_visual_asset(gt_visuals["labeled"], case_dir / "gt_labeled.png")
            gt_overlay_src = copy_visual_asset(gt_visuals["overlay"], case_dir / "gt_overlay.png")
            pred_labeled_src = copy_visual_asset(pred_visuals["labeled"], case_dir / "pred_labeled.png")
            pred_overlay_src = copy_visual_asset(pred_visuals["overlay"], case_dir / "pred_overlay.png")

            gt_bottom = (
                load_rgb_image(case_dir / "gt_overlay.png")
                if (case_dir / "gt_overlay.png").exists()
                else load_rgb_image(case_dir / "gt_labeled.png")
                if (case_dir / "gt_labeled.png").exists()
                else gt_color
            )
            pred_bottom = (
                load_rgb_image(case_dir / "pred_overlay.png")
                if (case_dir / "pred_overlay.png").exists()
                else load_rgb_image(case_dir / "pred_labeled.png")
                if (case_dir / "pred_labeled.png").exists()
                else pred_color
            )

            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            axes[0, 0].imshow(gt_color, interpolation="nearest")
            axes[0, 0].set_title(f"GT Labels (consistent colors)\n{pair.image_id}")
            axes[0, 1].imshow(pred_color, interpolation="nearest")
            axes[0, 1].set_title(f"Pred Labels (consistent colors)\n{pred_name}")
            axes[0, 2].imshow(agreement_map, interpolation="nearest")
            axes[0, 2].set_title("Agreement on valid GT\nGreen=match Red=mismatch Gray=ignore")
            axes[1, 0].imshow(gt_bottom, interpolation="nearest")
            axes[1, 0].set_title("GT companion view")
            axes[1, 1].imshow(pred_bottom, interpolation="nearest")
            axes[1, 1].set_title("Pred companion view")
            axes[1, 2].axis("off")
            axes[1, 2].text(
                0.0,
                1.0,
                "\n".join(
                    [
                        f"pred_name: {pred_name}",
                        f"bucket: {bucket_name}",
                        f"mIoU: {entry['miou']:.4f}",
                        f"hierIoU: {entry['hieriou']:.4f}",
                        f"Struct: {entry['struct']:.4f}",
                        f"Persist: {entry['persist']:.4f}",
                        f"Edge: {entry['edge']:.4f}",
                        f"valid_pixels: {gt_valid_pixels}",
                        f"ignore_pixels: {gt_ignore_pixels}",
                        f"valid_diff_pixels: {valid_diff_pixels}",
                        f"ignore_diff_pixels: {ignore_diff_pixels}",
                        f"gt_top_classes: {gt_class_counts[:4]}",
                        f"pred_top_classes: {pred_class_counts[:4]}",
                    ]
                ),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            for ax in axes.flat:
                if ax.has_data():
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.suptitle(f"{pred_name} - {bucket_name}")
            fig.tight_layout()
            fig.savefig(case_dir / "case_visualization.png", dpi=150)
            plt.close(fig)

            per_class_hier = np.asarray(hier_result["per_class_iou"], dtype=np.float64)
            per_class_struct = np.asarray(struct_result["per_class_struct"], dtype=np.float64)
            valid = np.isfinite(per_class_hier) | np.isfinite(per_class_struct)
            candidate_classes = np.where(valid)[0]
            chosen_classes: List[int] = []
            if candidate_classes.size > 0:
                hier_fill = np.where(np.isfinite(per_class_hier), per_class_hier, -1.0)
                struct_fill = np.where(np.isfinite(per_class_struct), per_class_struct, -1.0)
                gaps = np.abs(hier_fill - struct_fill)
                ordered = candidate_classes[np.argsort(-gaps[candidate_classes])]
                for class_id in ordered[:max_case_masks]:
                    gt_mask = gt == class_id
                    pred_mask = pred == class_id
                    if not gt_mask.any() and not pred_mask.any():
                        continue
                    save_binary_mask(gt_mask, case_dir / f"class_{class_id}_gt_mask.png")
                    save_binary_mask(pred_mask, case_dir / f"class_{class_id}_pred_mask.png")
                    chosen_classes.append(int(class_id))

            case_json = {
                "pred_name": pred_name,
                "bucket": bucket_name,
                "rank_in_bucket": rank,
                "image_id": pair.image_id,
                "gt_path": pair.gt_path,
                "pred_path": pair.pred_path,
                "metrics": {
                    "miou": entry["miou"],
                    "hieriou": entry["hieriou"],
                    "struct": entry["struct"],
                    "persist": entry["persist"],
                    "edge": entry["edge"],
                },
                "thresholds": thresholds,
                "key_classes": chosen_classes,
                "pixel_stats": {
                    "valid_pixels": gt_valid_pixels,
                    "ignore_pixels": gt_ignore_pixels,
                    "valid_diff_pixels": valid_diff_pixels,
                    "ignore_diff_pixels": ignore_diff_pixels,
                },
                "gt_top_classes": gt_class_counts,
                "pred_top_classes": pred_class_counts,
                "visual_sources": {
                    "gt_labeled_source": gt_labeled_src,
                    "gt_overlay_source": gt_overlay_src,
                    "pred_labeled_source": pred_labeled_src,
                    "pred_overlay_source": pred_overlay_src,
                },
            }
            save_json(case_json, case_dir / "case_summary.json")

            summary_rows.append(
                {
                    "pred_name": pred_name,
                    "bucket": bucket_name,
                    "rank_in_bucket": rank,
                    "image_id": pair.image_id,
                    "miou": entry["miou"],
                    "hieriou": entry["hieriou"],
                    "struct": entry["struct"],
                    "persist": entry["persist"],
                    "edge": entry["edge"],
                    "case_dir": str(case_dir),
                }
            )

    return summary_rows


def evaluate_class_groups(
    pred_results: Sequence[Dict[str, Any]],
    class_groups: Dict[str, List[int]],
    num_classes: int,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not class_groups:
        return rows

    for result in pred_results:
        overall_row = result["overall_row"]
        if overall_row is None:
            continue
        per_class_rows = result["per_class_rows"]
        metric_by_class = {int(row["class_id"]): row for row in per_class_rows}

        for group_name, raw_class_ids in class_groups.items():
            valid_class_ids = [cid for cid in raw_class_ids if 0 <= cid < num_classes]
            dropped = [cid for cid in raw_class_ids if cid not in valid_class_ids]
            if dropped:
                warn(
                    f"{result['pred_name']}: class group '{group_name}' contains out-of-range ids {dropped}. "
                    f"They were ignored.",
                    warnings,
                )

            if not valid_class_ids:
                rows.append(
                    {
                        "pred_name": result["pred_name"],
                        "group_name": group_name,
                        "num_classes_in_group": 0,
                        "group_miou": float("nan"),
                        "group_hieriou": float("nan"),
                        "group_struct": float("nan"),
                        "group_persist": float("nan"),
                        "group_edge": float("nan"),
                    }
                )
                continue

            group_iou = np.asarray([metric_by_class[cid]["iou"] for cid in valid_class_ids], dtype=np.float64)
            group_hier = np.asarray([metric_by_class[cid]["hieriou"] for cid in valid_class_ids], dtype=np.float64)
            group_struct = np.asarray([metric_by_class[cid]["struct"] for cid in valid_class_ids], dtype=np.float64)
            group_persist = np.asarray([metric_by_class[cid]["persist"] for cid in valid_class_ids], dtype=np.float64)
            group_edge = np.asarray([metric_by_class[cid]["edge"] for cid in valid_class_ids], dtype=np.float64)

            rows.append(
                {
                    "pred_name": result["pred_name"],
                    "group_name": group_name,
                    "num_classes_in_group": len(valid_class_ids),
                    "group_miou": float(np.nanmean(group_iou)) if np.isfinite(group_iou).any() else float("nan"),
                    "group_hieriou": float(np.nanmean(group_hier)) if np.isfinite(group_hier).any() else float("nan"),
                    "group_struct": float(np.nanmean(group_struct)) if np.isfinite(group_struct).any() else float("nan"),
                    "group_persist": float(np.nanmean(group_persist)) if np.isfinite(group_persist).any() else float("nan"),
                    "group_edge": float(np.nanmean(group_edge)) if np.isfinite(group_edge).any() else float("nan"),
                }
            )

    return rows


def evaluate_all_prediction_dirs(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(args.output_dir))
    plots_dir = ensure_dir(output_dir / "plots")
    case_dir = ensure_dir(output_dir / "case_studies")
    logs_dir = ensure_dir(output_dir / "logs")

    warnings: List[str] = []
    similarity_matrix = load_similarity_matrix(Path(args.similarity_matrix), args.num_classes, warnings)
    class_groups = load_class_groups(args.class_groups)

    pred_dirs = [Path(p) for p in args.pred_dirs]
    pred_names = list(args.pred_names) if args.pred_names else []
    while len(pred_names) < len(pred_dirs):
        pred_names.append(pred_dirs[len(pred_names)].name)
    if len(pred_names) != len(pred_dirs):
        raise ValueError("pred_names must have the same length as pred_dirs")

    pred_results = []
    overall_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    bootstrap_summary: Dict[str, Any] = {}
    case_rows: List[Dict[str, Any]] = []

    pred_items = list(zip(pred_names, pred_dirs))
    pred_iterator = tqdm(
        pred_items,
        total=len(pred_items),
        desc="Prediction dirs",
        unit="dir",
        leave=True,
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    for pred_name, pred_dir in pred_iterator:
        result = evaluate_single_prediction_dir(
            gt_dir=Path(args.gt_dir),
            pred_dir=pred_dir,
            pred_name=pred_name,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            similarity_matrix=similarity_matrix,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_seed=args.seed,
            gt_offset=args.gt_offset,
            pred_offset=args.pred_offset,
            pred_collapse_ge_to=args.collapse_pred_ge_to,
            pred_collapse_target=args.collapse_pred_target,
            max_radius=args.max_radius,
            boundary_width=args.boundary_width,
            kernel_type=args.kernel_type,
            max_images=args.max_images,
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            warnings=warnings,
        )
        pred_results.append(result)
        if result["overall_row"] is None:
            continue

        overall_rows.append(result["overall_row"])
        per_class_rows.extend(result["per_class_rows"])
        image_rows.extend(result["image_rows"])
        bootstrap_summary[pred_name] = result["bootstrap"]

        selected_cases, thresholds = select_case_studies(
            result["image_entries"],
            case_quantile=args.case_quantile,
            max_cases_per_bucket=args.max_cases_per_bucket,
        )
        case_rows.extend(
            save_case_visualizations(
                pred_name=pred_name,
                selected_cases=selected_cases,
                thresholds=thresholds,
                similarity_matrix=similarity_matrix,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                gt_offset=args.gt_offset,
                pred_offset=args.pred_offset,
                pred_collapse_ge_to=args.collapse_pred_ge_to,
                pred_collapse_target=args.collapse_pred_target,
                case_gt_visual_dir=args.case_gt_visual_dir,
                out_dir=case_dir,
                max_case_masks=args.max_case_masks,
            )
        )
    pred_iterator.close()

    ranking_rows = make_ranking_table(overall_rows)
    pairwise_rows = build_pairwise_comparison_table(overall_rows)
    class_group_rows = evaluate_class_groups(pred_results, class_groups or {}, args.num_classes, warnings)

    overall_fieldnames = [
        "pred_name",
        "pred_dir",
        "num_matched_images",
        "num_evaluated_images",
        "num_skipped_images",
        "miou_mean",
        "miou_ci_low",
        "miou_ci_high",
        "hieriou_mean",
        "hieriou_ci_low",
        "hieriou_ci_high",
        "struct_mean",
        "struct_ci_low",
        "struct_ci_high",
        "persist_mean",
        "edge_mean",
        "struct_image_mean",
        "persist_image_mean",
        "edge_image_mean",
        "struct_class_mean",
        "persist_class_mean",
        "edge_class_mean",
    ]
    ranking_fieldnames = [
        "pred_name",
        "rank_miou",
        "rank_hieriou",
        "rank_struct",
        "miou_mean",
        "hieriou_mean",
        "struct_mean",
        "rank_gap_struct_vs_miou",
        "rank_gap_struct_vs_hieriou",
        "rank_gap_hieriou_vs_miou",
    ]
    per_class_fieldnames = ["pred_name", "class_id", "iou", "hieriou", "persist", "edge", "struct"]
    image_fieldnames = [
        "pred_name",
        "image_id",
        "miou",
        "hieriou",
        "struct",
        "persist",
        "edge",
        "num_valid_classes",
        "gt_rel",
        "pred_rel",
    ]
    pairwise_fieldnames = [
        "pred_a",
        "pred_b",
        "miou_gap",
        "hieriou_gap",
        "struct_gap",
        "abs_miou_gap",
        "abs_hieriou_gap",
        "abs_struct_gap",
        "close_miou_far_struct",
        "higher_hieriou_lower_struct_conflict",
    ]
    class_group_fieldnames = [
        "pred_name",
        "group_name",
        "num_classes_in_group",
        "group_miou",
        "group_hieriou",
        "group_struct",
        "group_persist",
        "group_edge",
    ]
    case_fieldnames = [
        "pred_name",
        "bucket",
        "rank_in_bucket",
        "image_id",
        "miou",
        "hieriou",
        "struct",
        "persist",
        "edge",
        "case_dir",
    ]

    save_csv(overall_rows, output_dir / "overall_metrics.csv", overall_fieldnames)
    save_csv(ranking_rows, output_dir / "ranking_table.csv", ranking_fieldnames)
    save_csv(per_class_rows, output_dir / "per_class_metrics.csv", per_class_fieldnames)
    save_csv(image_rows, output_dir / "image_level_metrics.csv", image_fieldnames)
    save_csv(pairwise_rows, output_dir / "pairwise_comparisons.csv", pairwise_fieldnames)
    save_csv(case_rows, output_dir / "case_studies.csv", case_fieldnames)
    if class_groups is not None:
        save_csv(class_group_rows, output_dir / "class_group_metrics.csv", class_group_fieldnames)

    plot_directory_level_scatter(overall_rows, plots_dir / "directory_level_scatter_hieriou_vs_struct.png")
    plot_image_level_scatter(image_rows, plots_dir)
    plot_per_class_scatter(per_class_rows, plots_dir)
    plot_overall_metric_bars(overall_rows, plots_dir / "overall_scores_bar.png")
    plot_rank_comparison(ranking_rows, plots_dir / "rank_comparison.png")

    summary = {
        "config": {
            "gt_dir": str(Path(args.gt_dir)),
            "pred_dirs": [str(p) for p in pred_dirs],
            "pred_names": pred_names,
            "num_classes": args.num_classes,
            "ignore_index": args.ignore_index,
            "similarity_matrix": str(Path(args.similarity_matrix)),
            "output_dir": str(output_dir),
            "bootstrap_iters": args.bootstrap_iters,
            "seed": args.seed,
            "class_groups": str(args.class_groups) if args.class_groups else None,
            "case_quantile": args.case_quantile,
            "max_cases_per_bucket": args.max_cases_per_bucket,
            "gt_offset": args.gt_offset,
            "pred_offset": args.pred_offset,
            "collapse_pred_ge_to": args.collapse_pred_ge_to,
            "collapse_pred_target": args.collapse_pred_target,
            "max_radius": args.max_radius,
            "boundary_width": args.boundary_width,
            "kernel_type": args.kernel_type,
            "max_images": args.max_images,
            "num_workers": args.num_workers,
            "no_progress": args.no_progress,
            "case_gt_visual_dir": str(args.case_gt_visual_dir) if args.case_gt_visual_dir else None,
        },
        "overall_metrics": overall_rows,
        "ranking_table": ranking_rows,
        "pairwise_comparisons": pairwise_rows,
        "bootstrap": bootstrap_summary,
        "class_group_metrics": class_group_rows,
        "case_studies": case_rows,
        "warnings": warnings,
    }

    save_json(summary, output_dir / "summary.json")
    save_json({"warnings": warnings}, logs_dir / "warnings.json")
    save_json(bootstrap_summary, logs_dir / "bootstrap_results.json")

    print(f"Saved outputs to {output_dir}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate real prediction directories with mIoU, hierIoU, and StructScore.")
    parser.add_argument("--gt_dir", type=Path, required=True, help="Ground-truth label directory")
    parser.add_argument("--pred_dirs", type=Path, nargs="+", required=True, help="Prediction directories")
    parser.add_argument("--pred_names", nargs="*", default=None, help="Readable names for prediction directories")
    parser.add_argument(
        "--similarity_matrix",
        type=Path,
        required=True,
        help="Similarity matrix path or directory containing S.npy",
    )
    parser.add_argument("--num_classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore label value")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--bootstrap_iters", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class_groups", type=Path, default=None, help="Optional class_groups.json")
    parser.add_argument("--case_quantile", type=float, default=0.2, help="Quantile for high/low case selection")
    parser.add_argument("--max_cases_per_bucket", type=int, default=5, help="Max cases per quadrant bucket")
    parser.add_argument("--max_case_masks", type=int, default=3, help="Max key class masks per case")
    parser.add_argument("--gt_offset", type=int, default=0, help="Optional GT class offset")
    parser.add_argument("--pred_offset", type=int, default=0, help="Optional prediction class offset")
    parser.add_argument(
        "--collapse_pred_ge_to",
        type=int,
        default=None,
        help="Optionally collapse predicted labels >= this threshold into --collapse_pred_target.",
    )
    parser.add_argument(
        "--collapse_pred_target",
        type=int,
        default=None,
        help="Target label used with --collapse_pred_ge_to for official evaluator-style remapping.",
    )
    parser.add_argument("--max_radius", type=int, default=3, help="StructScore persistence radius")
    parser.add_argument("--boundary_width", type=int, default=1, help="StructScore boundary width")
    parser.add_argument("--kernel_type", choices=["disk", "square"], default="disk", help="StructScore kernel type")
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on matched images per prediction dir for smoke tests")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Image-level worker processes. Use 1 for single-process, 0 for auto, or a positive integer.",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--case_gt_visual_dir",
        type=Path,
        default=DEFAULT_CASE_GT_VIS_DIR if DEFAULT_CASE_GT_VIS_DIR.exists() else None,
        help="Optional GT companion visualization directory used only for case-study export",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.num_classes <= 0:
        raise ValueError("--num_classes must be > 0")
    if args.bootstrap_iters <= 0:
        raise ValueError("--bootstrap_iters must be > 0")
    if not (0.0 < args.case_quantile < 0.5):
        raise ValueError("--case_quantile must be in (0, 0.5)")
    if args.max_images < 0:
        raise ValueError("--max_images must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if (args.collapse_pred_ge_to is None) != (args.collapse_pred_target is None):
        raise ValueError("--collapse_pred_ge_to and --collapse_pred_target must be provided together")

    evaluate_all_prediction_dirs(args)


if __name__ == "__main__":
    main()
