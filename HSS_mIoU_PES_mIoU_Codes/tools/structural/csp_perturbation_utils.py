from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))

from structural_metrics import _list_label_files, _make_structuring_element, boundary_band, load_label_map


DEFAULT_PERTURBATION_SPECS: Dict[str, Dict[str, object]] = {
    "shift": {
        "levels": [1, 2, 4],
        "family": "edge",
        "description": "Translate the class mask by a small number of pixels.",
    },
    "erode": {
        "levels": [1, 2, 3],
        "family": "mixed",
        "description": "Shrink the class mask with morphological erosion.",
    },
    "dilate": {
        "levels": [1, 2, 3],
        "family": "mixed",
        "description": "Expand the class mask with morphological dilation.",
    },
    "hole": {
        "levels": [0.02, 0.05, 0.10],
        "family": "persist",
        "description": "Remove interior regions to create holes.",
    },
    "crack": {
        "levels": [1, 2, 4],
        "family": "persist",
        "description": "Cut a stripe through the object to break connectivity.",
    },
    "boundary_noise": {
        "levels": [0.01, 0.03, 0.05],
        "family": "edge",
        "description": "Flip a small subset of boundary-band pixels.",
    },
}


def parse_name_list(text: str) -> List[str]:
    """Parse a comma-separated list of names."""
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one name must be provided")
    return names


def build_perturbation_specs(names: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
    """Select a subset of the built-in perturbation specifications."""
    if names is None:
        return dict(DEFAULT_PERTURBATION_SPECS)

    specs: Dict[str, Dict[str, object]] = {}
    for name in names:
        if name not in DEFAULT_PERTURBATION_SPECS:
            raise ValueError(
                f"Unknown perturbation '{name}'. Available: {sorted(DEFAULT_PERTURBATION_SPECS.keys())}"
            )
        specs[name] = dict(DEFAULT_PERTURBATION_SPECS[name])
    return specs


def _sample_with_replacement(coords: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample `n` rows from a coordinate array, allowing replacement if needed."""
    if coords.shape[0] == 0 or n <= 0:
        return np.empty((0, 2), dtype=np.int64)
    replace = coords.shape[0] < n
    indices = rng.choice(coords.shape[0], size=n, replace=replace)
    return coords[indices]


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a binary mask without wraparound."""
    h, w = mask.shape
    shifted = np.zeros_like(mask, dtype=bool)

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w

    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _morph_mask(mask: np.ndarray, radius: int, mode: str, kernel_type: str) -> np.ndarray:
    """Apply a simple dilation or erosion to a binary mask."""
    mask = mask.astype(bool, copy=False)
    if radius <= 0:
        return mask.copy()
    structure = _make_structuring_element(radius, kernel_type=kernel_type)
    if mode == "erode":
        return ndi.binary_erosion(mask, structure=structure, border_value=0)
    if mode == "dilate":
        return ndi.binary_dilation(mask, structure=structure, border_value=0)
    raise ValueError(f"Unsupported morph mode: {mode}")


def _carve_holes(mask: np.ndarray, ratio: float, rng: np.random.Generator, kernel_type: str) -> np.ndarray:
    """Remove interior pixels until the requested ratio is approximately reached."""
    mask = mask.astype(bool, copy=False)
    if ratio <= 0 or not mask.any():
        return mask.copy()

    target_remove = max(1, int(round(mask.sum(dtype=np.float64) * ratio)))
    inner = ndi.binary_erosion(mask, structure=_make_structuring_element(1, kernel_type=kernel_type), border_value=0)
    centers = np.argwhere(inner if inner.any() else mask)
    out = mask.copy()
    removed = 0
    attempts = 0

    while removed < target_remove and centers.shape[0] > 0 and attempts < 32:
        attempts += 1
        center = centers[rng.integers(0, centers.shape[0])]
        remaining = target_remove - removed
        hole_radius = max(1, int(np.sqrt(max(remaining, 1) / np.pi) / 1.5))
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        if kernel_type == "square":
            hole = (np.abs(yy - center[0]) <= hole_radius) & (np.abs(xx - center[1]) <= hole_radius)
        else:
            hole = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= hole_radius ** 2
        to_remove = hole & out
        removed_now = int(to_remove.sum())
        if removed_now == 0:
            continue
        out[to_remove] = False
        removed += removed_now
        inner = ndi.binary_erosion(out, structure=_make_structuring_element(1, kernel_type=kernel_type), border_value=0)
        centers = np.argwhere(inner if inner.any() else out)

    return out


def _cut_crack(mask: np.ndarray, width: int, rng: np.random.Generator) -> np.ndarray:
    """Cut a horizontal or vertical stripe through the object."""
    mask = mask.astype(bool, copy=False)
    if width <= 0 or not mask.any():
        return mask.copy()

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    out = mask.copy()

    for _ in range(8):
        cy, cx = coords[rng.integers(0, coords.shape[0])]
        orientation = "horizontal" if rng.random() < 0.5 else "vertical"
        stripe = np.zeros_like(mask, dtype=bool)
        if orientation == "horizontal":
            y0 = max(y_min, cy - width // 2)
            y1 = min(y_max + 1, y0 + width)
            stripe[y0:y1, x_min:x_max + 1] = True
        else:
            x0 = max(x_min, cx - width // 2)
            x1 = min(x_max + 1, x0 + width)
            stripe[y_min:y_max + 1, x0:x1] = True

        removed = stripe & out
        if removed.any():
            out[removed] = False
            break

    return out


def _boundary_noise(mask: np.ndarray, ratio: float, rng: np.random.Generator, kernel_type: str) -> np.ndarray:
    """Flip a subset of pixels in a thin boundary band."""
    mask = mask.astype(bool, copy=False)
    if ratio <= 0:
        return mask.copy()

    band = boundary_band(mask, delta=1, kernel_type=kernel_type)
    coords = np.argwhere(band)
    if coords.shape[0] == 0:
        return mask.copy()

    count = max(1, int(round(coords.shape[0] * ratio)))
    chosen = _sample_with_replacement(coords, rng, count)
    out = mask.copy()
    out[chosen[:, 0], chosen[:, 1]] = ~out[chosen[:, 0], chosen[:, 1]]
    return out


def apply_perturbation(
    mask: np.ndarray,
    perturbation_name: str,
    level_value: float,
    seed: int,
    kernel_type: str = "disk",
) -> np.ndarray:
    """Apply one named perturbation to a binary class mask."""
    mask = mask.astype(bool, copy=False)
    rng = np.random.default_rng(seed)

    if perturbation_name == "shift":
        direction_table = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dy_base, dx_base = direction_table[seed % len(direction_table)]
        return _shift_mask(mask, dy=int(level_value * dy_base), dx=int(level_value * dx_base))
    if perturbation_name == "erode":
        return _morph_mask(mask, radius=int(level_value), mode="erode", kernel_type=kernel_type)
    if perturbation_name == "dilate":
        return _morph_mask(mask, radius=int(level_value), mode="dilate", kernel_type=kernel_type)
    if perturbation_name == "hole":
        return _carve_holes(mask, ratio=float(level_value), rng=rng, kernel_type=kernel_type)
    if perturbation_name == "crack":
        return _cut_crack(mask, width=int(level_value), rng=rng)
    if perturbation_name == "boundary_noise":
        return _boundary_noise(mask, ratio=float(level_value), rng=rng, kernel_type=kernel_type)
    raise ValueError(f"Unknown perturbation '{perturbation_name}'")


def select_class_masks(
    gt_dir: str,
    num_classes: int,
    ignore_index: int = 255,
    max_images: Optional[int] = None,
    max_instances: Optional[int] = 200,
    max_classes_per_image: int = 2,
    min_mask_pixels: int = 64,
    class_ids: Optional[List[int]] = None,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Sample binary class masks from a directory of semantic label maps."""
    rng = np.random.default_rng(seed)
    gt_files = _list_label_files(gt_dir)
    if max_images is not None:
        gt_files = gt_files[:max_images]

    class_filter = None if class_ids is None else set(class_ids)
    records: List[Dict[str, object]] = []

    for image_index, gt_path in enumerate(gt_files):
        gt = load_label_map(Path(gt_path))
        valid = gt != ignore_index
        if not np.any(valid):
            continue

        present_classes = np.unique(gt[valid])
        present_classes = present_classes[(present_classes >= 0) & (present_classes < num_classes)]
        if class_filter is not None:
            present_classes = np.asarray([c for c in present_classes.tolist() if c in class_filter], dtype=np.int64)
        if present_classes.size == 0:
            continue

        present_classes = rng.permutation(present_classes)
        classes_taken = 0
        for class_id in present_classes.tolist():
            mask = (gt == class_id) & valid
            area = int(mask.sum())
            if area < min_mask_pixels:
                continue

            record = {
                "instance_id": len(records),
                "image_index": image_index,
                "image_path": str(Path(gt_path).resolve()),
                "class_id": int(class_id),
                "mask": mask.astype(bool, copy=False),
                "area": area,
                "shape": list(mask.shape),
            }
            records.append(record)
            classes_taken += 1

            if max_instances is not None and len(records) >= max_instances:
                return records
            if max_classes_per_image > 0 and classes_taken >= max_classes_per_image:
                break

    return records
