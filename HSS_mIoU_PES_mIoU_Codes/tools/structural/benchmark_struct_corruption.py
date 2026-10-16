import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))

from csp_perturbation_utils import _carve_holes, _cut_crack, _morph_mask, parse_name_list
from structural_metrics import (
    _list_label_files,
    _match_pred_gt_files,
    _make_structuring_element,
    binary_iou,
    compute_multiclass_struct_score,
    load_label_map,
    miou_from_confusion,
    update_confusion_matrix,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional UX improvement
    tqdm = None


DEFAULT_CORRUPTION_STRENGTHS: Dict[str, List[float]] = {
    "identity": [0],
    "boundary_shift": [1, 2, 3],
    "hole_insertion": [0.05, 0.10, 0.20],
    "component_break": [1, 2, 3],
    "thin_part_removal": [1, 2, 3],
    "blob_smoothing": [3, 5, 7],
    "instance_merge": [1, 2, 3],
}


def parse_number_list(text: str) -> List[float]:
    """Parse a comma-separated list of numeric values."""
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("At least one numeric value must be provided")
    return values


def parse_strengths_map(text: str) -> Dict[str, List[float]]:
    """Parse per-corruption strengths like `hole_insertion:0.05|0.1|0.2;boundary_shift:1|2|3`."""
    result: Dict[str, List[float]] = {}
    if not text.strip():
        return result

    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid strengths-map chunk: {chunk}")
        name, values_text = chunk.split(":", 1)
        values = [float(v.strip()) for v in values_text.split("|") if v.strip()]
        if not values:
            raise ValueError(f"No strengths found for corruption '{name}'")
        result[name.strip()] = values
    return result


def _to_serializable(value: Any) -> Any:
    """Convert numpy-heavy objects into JSON-safe Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _strength_to_str(value: float) -> str:
    """Format a numeric strength so it is safe in file names and CSV keys."""
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _build_strength_map(
    corruption_names: List[str],
    global_strengths: Optional[List[float]],
    override_map: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Resolve strengths for each requested corruption."""
    strength_map: Dict[str, List[float]] = {}
    for name in corruption_names:
        if global_strengths is not None:
            strength_map[name] = list(global_strengths)
        else:
            if name not in DEFAULT_CORRUPTION_STRENGTHS:
                raise ValueError(f"No default strengths configured for corruption '{name}'")
            strength_map[name] = list(DEFAULT_CORRUPTION_STRENGTHS[name])

    for name, values in override_map.items():
        if name not in corruption_names:
            continue
        strength_map[name] = values
    return strength_map


def _display_label_map(label_map: np.ndarray, ignore_index: int, num_classes: int) -> np.ndarray:
    """Prepare a label map for imshow while keeping ignore pixels visible."""
    display = label_map.copy()
    display[display == ignore_index] = num_classes
    return display


def _mask_bbox(mask: np.ndarray, margin: int = 6) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded bounding box `(y0, y1, x0, x1)` for a binary mask."""
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)
    y1 = min(mask.shape[0], int(y1) + margin)
    x1 = min(mask.shape[1], int(x1) + margin)
    return y0, y1, x0, x1


def _draw_bbox(ax: Any, bbox: Optional[Tuple[int, int, int, int]], color: str = "red") -> None:
    """Draw a rectangle on an axis if a valid bbox exists."""
    if bbox is None:
        return
    y0, y1, x0, x1 = bbox
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2.0)
    ax.add_patch(rect)


def _crop_array(array: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Crop a 2D or 3D array to a bounding box; return the original if bbox is missing."""
    if bbox is None:
        return array
    y0, y1, x0, x1 = bbox
    return array[y0:y1, x0:x1, ...] if array.ndim == 3 else array[y0:y1, x0:x1]


def _build_change_overlay(original_mask: np.ndarray, corrupted_mask: np.ndarray) -> np.ndarray:
    """Color-code removed and added pixels for easy visual inspection.

    Red   : removed target pixels
    Cyan  : added target pixels
    White : unchanged target pixels
    Black : background
    """
    original_mask = original_mask.astype(bool, copy=False)
    corrupted_mask = corrupted_mask.astype(bool, copy=False)
    removed = original_mask & ~corrupted_mask
    added = corrupted_mask & ~original_mask
    kept = original_mask & corrupted_mask

    overlay = np.zeros(original_mask.shape + (3,), dtype=np.float32)
    overlay[kept] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    overlay[removed] = np.array([1.0, 0.2, 0.2], dtype=np.float32)
    overlay[added] = np.array([0.1, 0.9, 1.0], dtype=np.float32)
    return overlay


def _nearest_non_target_label_map(gt: np.ndarray, target_class: int, ignore_index: int) -> np.ndarray:
    """Build a fallback label map using the nearest non-target GT label.

    This is used when a corruption removes target-class pixels. We need those
    pixels to become some plausible *other* label; otherwise erosion/hole/crack
    corruptions would be undone when the base prediction is GT itself.
    """
    valid_source = (gt != target_class) & (gt != ignore_index)
    if not np.any(valid_source):
        return np.zeros_like(gt, dtype=gt.dtype)

    _, indices = ndi.distance_transform_edt(~valid_source, return_indices=True)
    nearest_labels = gt[indices[0], indices[1]]
    return nearest_labels.astype(gt.dtype, copy=False)


def _rewrite_single_class_prediction(
    base_pred: np.ndarray,
    gt: np.ndarray,
    target_class: int,
    new_mask: np.ndarray,
    ignore_index: int,
) -> np.ndarray:
    """Write one corrupted class mask back into a multiclass prediction map.

    Removed pixels revert to the nearest *non-target* GT label so shrinkage-like
    corruptions really change the semantic map even when `pred` starts from GT.
    """
    out = base_pred.copy()
    valid = gt != ignore_index
    old_mask = (base_pred == target_class) & valid
    new_mask = new_mask.astype(bool, copy=False) & valid

    removed = old_mask & ~new_mask
    if np.any(removed):
        fallback_labels = _nearest_non_target_label_map(gt, target_class=target_class, ignore_index=ignore_index)
        out[removed] = fallback_labels[removed]
    out[new_mask] = target_class
    out[~valid] = ignore_index
    return out


def _get_target_classes(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int,
    min_class_pixels: int,
    max_classes_per_image: int,
) -> List[int]:
    """Select target classes to corrupt for one image."""
    valid = gt != ignore_index
    present = np.unique(pred[valid])
    present = present[(present >= 0) & (present < num_classes)]

    areas = []
    for class_id in present.tolist():
        area = int(((pred == class_id) & valid).sum())
        if area >= min_class_pixels:
            areas.append((class_id, area))
    areas.sort(key=lambda item: item[1], reverse=True)

    class_ids = [class_id for class_id, _ in areas]
    if max_classes_per_image > 0:
        class_ids = class_ids[:max_classes_per_image]
    return class_ids


def apply_boundary_shift(mask: np.ndarray, strength: float, kernel_type: str = "disk") -> np.ndarray:
    """Create a light inward or outward boundary offset.

    The benchmark objective here is to stress the edge term, so we use either
    erosion or dilation with the same radius as the corruption strength.
    """
    radius = max(1, int(round(strength)))
    eroded = _morph_mask(mask, radius=radius, mode="erode", kernel_type=kernel_type)
    dilated = _morph_mask(mask, radius=radius, mode="dilate", kernel_type=kernel_type)

    if eroded.any():
        return eroded
    return dilated


def apply_hole_insertion(mask: np.ndarray, strength: float, kernel_type: str = "disk", seed: int = 0) -> np.ndarray:
    """Insert interior holes to break region completeness."""
    rng = np.random.default_rng(seed)
    return _carve_holes(mask, ratio=float(strength), rng=rng, kernel_type=kernel_type)


def apply_component_break(mask: np.ndarray, strength: float, seed: int = 0) -> np.ndarray:
    """Cut a stripe through the object to create a break."""
    rng = np.random.default_rng(seed)
    return _cut_crack(mask, width=max(1, int(round(strength))), rng=rng)


def apply_thin_part_removal(mask: np.ndarray, strength: float, kernel_type: str = "disk") -> np.ndarray:
    """Remove thin protrusions using morphological opening.

    This corruption is intended to stress fine legs, poles, branches, and other
    narrow structures that disappear under light erosion/opening.
    """
    radius = max(1, int(round(strength)))
    structure = _make_structuring_element(radius, kernel_type=kernel_type)
    opened = ndi.binary_opening(mask.astype(bool, copy=False), structure=structure)
    return opened


def apply_blob_smoothing(mask: np.ndarray, strength: float, kernel_type: str = "disk") -> np.ndarray:
    """Smooth a mask with opening followed by closing."""
    kernel_size = max(1, int(round(strength)))
    radius = max(1, kernel_size // 2)
    structure = _make_structuring_element(radius, kernel_type=kernel_type)
    out = ndi.binary_opening(mask.astype(bool, copy=False), structure=structure)
    out = ndi.binary_closing(out, structure=structure)
    return out


def apply_instance_merge(mask: np.ndarray, strength: float, kernel_type: str = "disk") -> np.ndarray:
    """Bridge nearby same-class components by dilation.

    If multiple components are close, dilation makes them stick together,
    simulating a common semantic-segmentation failure where separate objects
    collapse into one blob.
    """
    radius = max(1, int(round(strength)))
    merged = _morph_mask(mask, radius=radius, mode="dilate", kernel_type=kernel_type)
    return merged


def compute_multiclass_iou(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """Compute standard multi-class IoU for one label-map pair."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    update_confusion_matrix(confusion, gt, pred, num_classes, ignore_index)
    return miou_from_confusion(confusion)


def _apply_corruption_to_prediction(
    gt: np.ndarray,
    base_pred: np.ndarray,
    target_class: int,
    corruption_type: str,
    strength: float,
    ignore_index: int,
    kernel_type: str,
    seed: int,
) -> Tuple[np.ndarray, bool]:
    """Apply one corruption to one target class and rewrite it into the multiclass prediction."""
    valid = gt != ignore_index
    target_mask = (base_pred == target_class) & valid
    if not target_mask.any():
        return base_pred.copy(), False

    if corruption_type == "identity":
        return base_pred.copy(), True
    if corruption_type == "boundary_shift":
        new_mask = apply_boundary_shift(target_mask, strength=strength, kernel_type=kernel_type)
    elif corruption_type == "hole_insertion":
        new_mask = apply_hole_insertion(target_mask, strength=strength, kernel_type=kernel_type, seed=seed)
    elif corruption_type == "component_break":
        new_mask = apply_component_break(target_mask, strength=strength, seed=seed)
    elif corruption_type == "thin_part_removal":
        new_mask = apply_thin_part_removal(target_mask, strength=strength, kernel_type=kernel_type)
    elif corruption_type == "blob_smoothing":
        new_mask = apply_blob_smoothing(target_mask, strength=strength, kernel_type=kernel_type)
    elif corruption_type == "instance_merge":
        new_mask = apply_instance_merge(target_mask, strength=strength, kernel_type=kernel_type)
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    new_mask = new_mask.astype(bool, copy=False) & valid
    if np.array_equal(new_mask, target_mask):
        return base_pred.copy(), False

    corrupted_pred = _rewrite_single_class_prediction(
        base_pred=base_pred,
        gt=gt,
        target_class=target_class,
        new_mask=new_mask,
        ignore_index=ignore_index,
    )
    return corrupted_pred, True


def run_single_corruption_benchmark(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_id: str,
    image_id: str,
    target_class: int,
    corruption_type: str,
    corruption_strength: float,
    num_classes: int,
    ignore_index: int = 255,
    max_radius: int = 3,
    boundary_width: int = 2,
    kernel_type: str = "disk",
    seed: int = 0,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]]:
    """Run one corruption case and return sample-level metrics plus per-class details."""
    corrupted_pred, applied = _apply_corruption_to_prediction(
        gt=gt,
        base_pred=pred,
        target_class=target_class,
        corruption_type=corruption_type,
        strength=corruption_strength,
        ignore_index=ignore_index,
        kernel_type=kernel_type,
        seed=seed,
    )
    if not applied:
        return None

    miou_res = compute_multiclass_iou(gt, corrupted_pred, num_classes=num_classes, ignore_index=ignore_index)
    struct_res = compute_multiclass_struct_score(
        gt=gt,
        pred=corrupted_pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_radius=max_radius,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )

    per_class_rows: List[Dict[str, Any]] = []
    for class_id in struct_res["valid_classes"]:
        per_class_rows.append(
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "target_class": int(target_class),
                "class_id": int(class_id),
                "is_target_class": int(class_id == target_class),
                "corruption_type": corruption_type,
                "corruption_strength": float(corruption_strength),
                "class_iou": float(miou_res["per_class_iou"][class_id]) if not np.isnan(miou_res["per_class_iou"][class_id]) else np.nan,
                "class_persist": float(struct_res["per_class_persist"][class_id]) if not np.isnan(struct_res["per_class_persist"][class_id]) else np.nan,
                "class_edge": float(struct_res["per_class_edge"][class_id]) if not np.isnan(struct_res["per_class_edge"][class_id]) else np.nan,
                "class_struct": float(struct_res["per_class_struct"][class_id]) if not np.isnan(struct_res["per_class_struct"][class_id]) else np.nan,
            }
        )

    target_iou = miou_res["per_class_iou"][target_class] if target_class < len(miou_res["per_class_iou"]) else np.nan
    target_persist = struct_res["per_class_persist"][target_class] if target_class < len(struct_res["per_class_persist"]) else np.nan
    target_edge = struct_res["per_class_edge"][target_class] if target_class < len(struct_res["per_class_edge"]) else np.nan
    target_struct = struct_res["per_class_struct"][target_class] if target_class < len(struct_res["per_class_struct"]) else np.nan

    sample_row = {
        "sample_id": sample_id,
        "image_id": image_id,
        "target_class": int(target_class),
        "corruption_type": corruption_type,
        "corruption_strength": float(corruption_strength),
        "miou": float(miou_res["miou"]),
        "image_struct": float(struct_res["image_score"]),
        "image_persist": float(struct_res["image_persist_mean"]),
        "image_edge": float(struct_res["image_edge_mean"]),
        "target_iou": float(target_iou) if not np.isnan(target_iou) else np.nan,
        "target_persist": float(target_persist) if not np.isnan(target_persist) else np.nan,
        "target_edge": float(target_edge) if not np.isnan(target_edge) else np.nan,
        "target_struct": float(target_struct) if not np.isnan(target_struct) else np.nan,
        "num_valid_classes": int(len(struct_res["valid_classes"])),
    }
    return sample_row, per_class_rows, corrupted_pred


def aggregate_benchmark_results(sample_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-sample rows by corruption type and strength."""
    if not sample_rows:
        return []

    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        groups[(row["corruption_type"], float(row["corruption_strength"]))].append(row)

    aggregated_rows: List[Dict[str, Any]] = []
    for (corruption_type, strength), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        def _mean_std(key: str) -> Tuple[float, float]:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            return float(np.nanmean(values)), float(np.nanstd(values))

        mean_miou, std_miou = _mean_std("miou")
        mean_struct, std_struct = _mean_std("image_struct")
        mean_persist, std_persist = _mean_std("image_persist")
        mean_edge, std_edge = _mean_std("image_edge")
        mean_target_struct, std_target_struct = _mean_std("target_struct")
        mean_target_persist, std_target_persist = _mean_std("target_persist")
        mean_target_edge, std_target_edge = _mean_std("target_edge")
        mean_target_iou, std_target_iou = _mean_std("target_iou")

        aggregated_rows.append(
            {
                "corruption_type": corruption_type,
                "corruption_strength": float(strength),
                "num_samples": len(rows),
                "mean_miou": mean_miou,
                "std_miou": std_miou,
                "mean_struct": mean_struct,
                "std_struct": std_struct,
                "mean_persist": mean_persist,
                "std_persist": std_persist,
                "mean_edge": mean_edge,
                "std_edge": std_edge,
                "mean_target_iou": mean_target_iou,
                "std_target_iou": std_target_iou,
                "mean_target_struct": mean_target_struct,
                "std_target_struct": std_target_struct,
                "mean_target_persist": mean_target_persist,
                "std_target_persist": std_target_persist,
                "mean_target_edge": mean_target_edge,
                "std_target_edge": std_target_edge,
            }
        )
    return aggregated_rows


def plot_metric_curves(aggregated_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Draw one metric curve figure per corruption type.

    Each figure highlights how mIoU / persist / edge / struct respond as the
    corruption strength increases. This directly addresses the benchmark goal:
    boundary_shift should mainly hurt edge, while hole/break should hurt persist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        groups[row["corruption_type"]].append(row)

    for corruption_type, rows in groups.items():
        rows = sorted(rows, key=lambda item: item["corruption_strength"])
        strengths = [row["corruption_strength"] for row in rows]

        plt.figure(figsize=(7, 5))
        plt.plot(strengths, [row["mean_miou"] for row in rows], marker="o", label="mIoU")
        plt.plot(strengths, [row["mean_persist"] for row in rows], marker="o", label="persist")
        plt.plot(strengths, [row["mean_edge"] for row in rows], marker="o", label="edge")
        plt.plot(strengths, [row["mean_struct"] for row in rows], marker="o", label="struct")
        plt.xlabel("Corruption Strength")
        plt.ylabel("Metric Value")
        plt.title(f"{corruption_type} (image-level metrics)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{corruption_type}_image_metrics.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(strengths, [row["mean_target_iou"] for row in rows], marker="o", label="target IoU")
        plt.plot(strengths, [row["mean_target_persist"] for row in rows], marker="o", label="target persist")
        plt.plot(strengths, [row["mean_target_edge"] for row in rows], marker="o", label="target edge")
        plt.plot(strengths, [row["mean_target_struct"] for row in rows], marker="o", label="target struct")
        plt.xlabel("Corruption Strength")
        plt.ylabel("Metric Value")
        plt.title(f"{corruption_type} (target-class metrics)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{corruption_type}_target_metrics.png", dpi=180)
        plt.close()


def save_visual_examples(
    gt: np.ndarray,
    original_pred: np.ndarray,
    corrupted_pred: np.ndarray,
    target_class: int,
    sample_id: str,
    corruption_type: str,
    corruption_strength: float,
    output_dir: Path,
    ignore_index: int,
    num_classes: int,
) -> None:
    """Save a marked visual panel with change maps and a zoomed-in corruption region."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_gt = gt == target_class
    target_pred = original_pred == target_class
    target_corrupted = corrupted_pred == target_class
    target_changed = np.logical_xor(target_pred, target_corrupted)
    semantic_changed = original_pred != corrupted_pred
    focus_mask = target_changed | semantic_changed
    focus_bbox = _mask_bbox(focus_mask, margin=10)
    change_overlay = _build_change_overlay(target_pred, target_corrupted)

    gt_disp = _display_label_map(gt, ignore_index=ignore_index, num_classes=num_classes)
    pred_disp = _display_label_map(original_pred, ignore_index=ignore_index, num_classes=num_classes)
    corrupted_disp = _display_label_map(corrupted_pred, ignore_index=ignore_index, num_classes=num_classes)
    gt_zoom = _crop_array(gt_disp, focus_bbox)
    pred_zoom = _crop_array(pred_disp, focus_bbox)
    corrupted_zoom = _crop_array(corrupted_disp, focus_bbox)
    overlay_zoom = _crop_array(change_overlay, focus_bbox)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(gt_disp, interpolation="nearest")
    axes[0, 0].set_title("GT")
    _draw_bbox(axes[0, 0], focus_bbox, color="yellow")
    axes[0, 1].imshow(pred_disp, interpolation="nearest")
    axes[0, 1].set_title("Original Pred")
    _draw_bbox(axes[0, 1], focus_bbox, color="yellow")
    axes[0, 2].imshow(corrupted_disp, interpolation="nearest")
    axes[0, 2].set_title("Corrupted Pred")
    _draw_bbox(axes[0, 2], focus_bbox, color="yellow")
    axes[0, 3].imshow(change_overlay, interpolation="nearest")
    axes[0, 3].set_title("Target Change Map")
    axes[0, 3].text(
        0.02,
        0.02,
        "white=kept\nred=removed\ncyan=added",
        transform=axes[0, 3].transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        verticalalignment="bottom",
    )
    axes[1, 0].imshow(target_gt, cmap="gray", interpolation="nearest")
    axes[1, 0].set_title(f"GT mask c={target_class}")
    axes[1, 1].imshow(target_pred, cmap="gray", interpolation="nearest")
    axes[1, 1].set_title("Original target mask")
    axes[1, 2].imshow(target_corrupted, cmap="gray", interpolation="nearest")
    axes[1, 2].set_title("Corrupted target mask")
    axes[1, 3].imshow(corrupted_zoom, interpolation="nearest")
    axes[1, 3].imshow(overlay_zoom, interpolation="nearest", alpha=0.65)
    axes[1, 3].set_title("Zoomed corruption region")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{sample_id} | {corruption_type} | strength={corruption_strength}")
    fig.tight_layout()

    out_name = f"{sample_id}.png"
    fig.savefig(output_dir / out_name, dpi=180)
    plt.close(fig)


def _build_pairs(gt_dir: str, pred_dir: Optional[str]) -> List[Tuple[str, str, str]]:
    """Match GT and prediction files, or clone GT as the base prediction when pred_dir is absent."""
    if pred_dir:
        pairs = _match_pred_gt_files(pred_dir, gt_dir, preview_pairs=0)
        return [(pp.name, str(gp), str(pp)) for pp, gp in pairs]

    gt_files = _list_label_files(gt_dir)
    return [(gp.stem, str(gp), str(gp)) for gp in gt_files]


def _make_progress(total: int, desc: str):
    """Create a progress iterator if tqdm is available, otherwise a lightweight fallback."""
    if tqdm is not None:
        return tqdm(total=total, desc=desc, dynamic_ncols=True)
    return None


def _progress_message(message: str, progress_bar: Any = None) -> None:
    """Emit a progress message without breaking tqdm output when available."""
    if progress_bar is not None and tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the structure corruption benchmark."""
    parser = argparse.ArgumentParser("Controlled structure corruption benchmark for CSP")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory of GT semantic label maps")
    parser.add_argument("--pred-dir", type=str, default="", help="Optional directory of prediction label maps")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignored GT label")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save benchmark outputs")
    parser.add_argument(
        "--corruptions",
        type=parse_name_list,
        default=list(DEFAULT_CORRUPTION_STRENGTHS.keys()),
        help="Comma-separated corruption names",
    )
    parser.add_argument(
        "--strengths",
        type=parse_number_list,
        default=None,
        help="Optional shared strength list applied to all corruptions, e.g. 1,2,3",
    )
    parser.add_argument(
        "--strengths-map",
        type=parse_strengths_map,
        default={},
        help="Optional per-corruption strengths, e.g. boundary_shift:1|2|3;hole_insertion:0.05|0.1|0.2",
    )
    parser.add_argument("--max-radius", type=int, default=3, help="CSP persistence radius")
    parser.add_argument("--boundary-width", type=int, default=2, help="CSP boundary width")
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"], help="Morphology kernel")
    parser.add_argument("--max-images", type=int, default=0, help="Optional image limit; 0 means all")
    parser.add_argument("--max-classes-per-image", type=int, default=1, help="How many target classes to corrupt per image; 0 means all")
    parser.add_argument("--min-class-pixels", type=int, default=64, help="Minimum target-class area to corrupt")
    parser.add_argument("--num-visual-examples", type=int, default=2, help="How many examples to save per corruption/strength")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic corruption choices")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the controlled structure corruption benchmark."""
    corruption_names = args.corruptions
    strength_map = _build_strength_map(corruption_names, args.strengths, args.strengths_map)

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    visuals_dir = output_dir / "visual_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _build_pairs(args.gt_dir, args.pred_dir if args.pred_dir else None)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    image_plans: List[Dict[str, Any]] = []
    total_tasks = 0
    strength_count = sum(len(strength_map[name]) for name in corruption_names)
    for image_id, gt_path, pred_path in pairs:
        gt = load_label_map(gt_path)
        pred = load_label_map(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {image_id}: gt={gt.shape}, pred={pred.shape}")

        target_classes = _get_target_classes(
            pred=pred,
            gt=gt,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            min_class_pixels=args.min_class_pixels,
            max_classes_per_image=args.max_classes_per_image,
        )
        image_plans.append(
            {
                "image_id": image_id,
                "gt_path": gt_path,
                "pred_path": pred_path,
                "target_classes": target_classes,
            }
        )
        total_tasks += len(target_classes) * strength_count

    sample_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    visual_counts: Dict[Tuple[str, float], int] = defaultdict(int)

    print(f"[Benchmark] images to process: {len(image_plans)}")
    print(f"[Benchmark] corruptions: {corruption_names}")
    print(f"[Benchmark] estimated corruption tasks: {total_tasks}")

    image_progress = _make_progress(len(image_plans), desc="Images")
    task_progress = _make_progress(total_tasks, desc="Corruptions") if total_tasks > 0 else None
    fallback_every = max(1, total_tasks // 20) if total_tasks > 0 else 1
    completed_tasks = 0

    try:
        for image_index, plan in enumerate(image_plans):
            image_id = plan["image_id"]
            gt_path = plan["gt_path"]
            pred_path = plan["pred_path"]
            target_classes = plan["target_classes"]

            gt = load_label_map(gt_path)
            pred = load_label_map(pred_path)

            if image_progress is not None:
                image_progress.set_postfix_str(f"{Path(image_id).stem} | classes={len(target_classes)}")
            elif target_classes:
                _progress_message(
                    f"[Benchmark] image {image_index + 1}/{len(image_plans)}: {Path(image_id).stem} "
                    f"(target classes={len(target_classes)})"
                )

            if not target_classes:
                if image_progress is not None:
                    image_progress.update(1)
                continue

            for target_class in target_classes:
                for corruption_type in corruption_names:
                    for strength in strength_map[corruption_type]:
                        sample_id = f"{Path(image_id).stem}_c{target_class}_{corruption_type}_s{_strength_to_str(strength)}"
                        seed = args.seed + image_index * 10007 + target_class * 97 + int(round(strength * 100))
                        res = run_single_corruption_benchmark(
                            gt=gt,
                            pred=pred,
                            sample_id=sample_id,
                            image_id=Path(image_id).stem,
                            target_class=target_class,
                            corruption_type=corruption_type,
                            corruption_strength=strength,
                            num_classes=args.num_classes,
                            ignore_index=args.ignore_index,
                            max_radius=args.max_radius,
                            boundary_width=args.boundary_width,
                            kernel_type=args.kernel_type,
                            seed=seed,
                        )
                        completed_tasks += 1
                        if task_progress is not None:
                            task_progress.update(1)
                            task_progress.set_postfix_str(
                                f"{Path(image_id).stem} | c={target_class} | {corruption_type}={strength}"
                            )
                        elif completed_tasks == 1 or completed_tasks % fallback_every == 0 or completed_tasks == total_tasks:
                            _progress_message(
                                f"[Benchmark] task {completed_tasks}/{total_tasks}: {Path(image_id).stem} "
                                f"c={target_class} {corruption_type} strength={strength}"
                            )

                        if res is None:
                            continue

                        sample_row, class_rows, corrupted_pred = res
                        sample_rows.append(sample_row)
                        per_class_rows.extend(class_rows)

                        key = (corruption_type, float(strength))
                        if visual_counts[key] < args.num_visual_examples:
                            save_visual_examples(
                                gt=gt,
                                original_pred=pred,
                                corrupted_pred=corrupted_pred,
                                target_class=target_class,
                                sample_id=sample_id,
                                corruption_type=corruption_type,
                                corruption_strength=strength,
                                output_dir=visuals_dir,
                                ignore_index=args.ignore_index,
                                num_classes=args.num_classes,
                            )
                            visual_counts[key] += 1

            if image_progress is not None:
                image_progress.update(1)
    finally:
        if image_progress is not None:
            image_progress.close()
        if task_progress is not None:
            task_progress.close()

    aggregated_rows = aggregate_benchmark_results(sample_rows)
    plot_metric_curves(aggregated_rows, plots_dir)

    sample_csv = output_dir / "sample_metrics.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()) if sample_rows else [])
        if sample_rows:
            writer.writeheader()
            writer.writerows(sample_rows)

    per_class_csv = output_dir / "per_class_metrics.csv"
    with per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()) if per_class_rows else [])
        if per_class_rows:
            writer.writeheader()
            writer.writerows(per_class_rows)

    aggregated_csv = output_dir / "aggregated_metrics.csv"
    with aggregated_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregated_rows[0].keys()) if aggregated_rows else [])
        if aggregated_rows:
            writer.writeheader()
            writer.writerows(aggregated_rows)

    summary = {
        "config": {
            "gt_dir": str(Path(args.gt_dir).resolve()),
            "pred_dir": str(Path(args.pred_dir).resolve()) if args.pred_dir else "",
            "num_classes": args.num_classes,
            "ignore_index": args.ignore_index,
            "corruptions": corruption_names,
            "strength_map": strength_map,
            "max_radius": args.max_radius,
            "boundary_width": args.boundary_width,
            "kernel_type": args.kernel_type,
            "max_images": args.max_images,
            "max_classes_per_image": args.max_classes_per_image,
            "min_class_pixels": args.min_class_pixels,
            "num_visual_examples": args.num_visual_examples,
            "seed": args.seed,
        },
        "num_sample_rows": len(sample_rows),
        "num_per_class_rows": len(per_class_rows),
        "aggregated_rows": _to_serializable(aggregated_rows),
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Benchmark] sample metrics CSV: {sample_csv}")
    print(f"[Benchmark] per-class metrics CSV: {per_class_csv}")
    print(f"[Benchmark] aggregated metrics CSV: {aggregated_csv}")
    print(f"[Benchmark] summary JSON: {summary_json}")
    print(f"[Benchmark] plots dir: {plots_dir}")
    print(f"[Benchmark] visual examples dir: {visuals_dir}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
