import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration
    njit = None
    NUMBA_AVAILABLE = False


def _make_structuring_element(radius: int, kernel_type: str = "disk") -> np.ndarray:
    """Create a 2D binary structuring element for morphology."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return np.ones((1, 1), dtype=bool)

    kernel_type = kernel_type.lower()
    if kernel_type == "square":
        size = 2 * radius + 1
        return np.ones((size, size), dtype=bool)
    if kernel_type == "disk":
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return (x * x + y * y) <= (radius * radius)
    raise ValueError(f"Unsupported kernel_type: {kernel_type}. Use 'disk' or 'square'.")


def _distance_transform(mask: np.ndarray, kernel_type: str = "disk") -> np.ndarray:
    """Compute a metric-consistent distance transform for a padded binary mask."""
    if kernel_type == "disk":
        return ndi.distance_transform_edt(mask)
    if kernel_type == "square":
        return ndi.distance_transform_cdt(mask, metric="chessboard").astype(np.float64)
    raise ValueError(f"Unsupported kernel_type: {kernel_type}. Use 'disk' or 'square'.")


def _crop_from_padded(arr: np.ndarray, pad: int) -> np.ndarray:
    """Crop a padded 2D array back to its original spatial extent."""
    if pad == 0:
        return arr
    return arr[pad:-pad, pad:-pad]


def _prepare_mask_distance_maps(mask: np.ndarray, pad: int, kernel_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distance-to-background and distance-to-foreground maps once per mask."""
    mask = np.ascontiguousarray(mask.astype(bool, copy=False))
    padded = np.pad(mask, pad, mode="constant", constant_values=False) if pad > 0 else mask
    dist_to_bg = _crop_from_padded(_distance_transform(padded, kernel_type=kernel_type), pad)
    dist_to_fg = _crop_from_padded(_distance_transform(~padded, kernel_type=kernel_type), pad)
    return dist_to_bg, dist_to_fg


if NUMBA_AVAILABLE:

    @njit(cache=False)
    def _persistence_score_from_distances_numba(
        gt_dist: np.ndarray,
        pred_dist: np.ndarray,
        max_radius: int,
    ) -> float:
        diff_inter = np.zeros(max_radius + 2, dtype=np.int64)
        diff_union = np.zeros(max_radius + 2, dtype=np.int64)

        gt_flat = gt_dist.ravel()
        pred_flat = pred_dist.ravel()

        for i in range(gt_flat.size):
            dg = gt_flat[i]
            dp = pred_flat[i]

            inter_limit = dg if dg < dp else dp
            union_limit = dg if dg > dp else dp

            inter_t = int(np.ceil(inter_limit)) - 1
            union_t = int(np.ceil(union_limit)) - 1

            if inter_t > max_radius:
                inter_t = max_radius
            if union_t > max_radius:
                union_t = max_radius

            if inter_t >= 0:
                diff_inter[0] += 1
                diff_inter[inter_t + 1] -= 1

            if union_t >= 0:
                diff_union[0] += 1
                diff_union[union_t + 1] -= 1

        running_inter = 0
        running_union = 0
        score_sum = 0.0
        valid_count = 0
        for r in range(max_radius + 1):
            running_inter += diff_inter[r]
            running_union += diff_union[r]
            if running_union > 0:
                score_sum += running_inter / running_union
                valid_count += 1

        if valid_count == 0:
            return 0.0
        return score_sum / valid_count


def _persistence_score_from_distances(gt_dist: np.ndarray, pred_dist: np.ndarray, max_radius: int) -> float:
    """Compute persistence directly from distance maps without per-radius masks."""
    if NUMBA_AVAILABLE:
        return float(_persistence_score_from_distances_numba(gt_dist, pred_dist, max_radius))

    diff_inter = np.zeros(max_radius + 2, dtype=np.int64)
    diff_union = np.zeros(max_radius + 2, dtype=np.int64)
    gt_flat = gt_dist.ravel()
    pred_flat = pred_dist.ravel()

    for dg, dp in zip(gt_flat, pred_flat):
        inter_t = int(np.ceil(min(dg, dp))) - 1
        union_t = int(np.ceil(max(dg, dp))) - 1

        if inter_t >= 0:
            inter_t = min(inter_t, max_radius)
            diff_inter[0] += 1
            diff_inter[inter_t + 1] -= 1

        if union_t >= 0:
            union_t = min(union_t, max_radius)
            diff_union[0] += 1
            diff_union[union_t + 1] -= 1

    inter = np.cumsum(diff_inter[:-1])
    union = np.cumsum(diff_union[:-1])
    valid = union > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(inter[valid] / union[valid]))


def _compute_class_score_details(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
) -> Dict[str, Union[float, int, bool, np.ndarray]]:
    """Compute class-wise scores and the exact counts needed for dataset aggregation."""
    pad = max(max_radius, boundary_width)
    gt_dist_bg, gt_dist_fg = _prepare_mask_distance_maps(gt_mask, pad=pad, kernel_type=kernel_type)
    pred_dist_bg, pred_dist_fg = _prepare_mask_distance_maps(pred_mask, pad=pad, kernel_type=kernel_type)

    persist_inter = np.zeros((max_radius + 1,), dtype=np.uint32)
    persist_union = np.zeros((max_radius + 1,), dtype=np.uint32)
    for radius in range(max_radius + 1):
        gt_eroded = gt_dist_bg > radius
        pred_eroded = pred_dist_bg > radius
        persist_inter[radius] = np.logical_and(gt_eroded, pred_eroded).sum(dtype=np.uint32)
        persist_union[radius] = np.logical_or(gt_eroded, pred_eroded).sum(dtype=np.uint32)

    persist_valid = persist_union > 0
    if np.any(persist_valid):
        s_persist = float(np.mean(persist_inter[persist_valid] / persist_union[persist_valid]))
    else:
        s_persist = float("nan")

    if boundary_width <= 0:
        edge_inter = 0
        edge_union = 0
        s_edge = float("nan")
    else:
        gt_band = np.logical_xor(gt_dist_fg <= boundary_width, gt_dist_bg > boundary_width)
        pred_band = np.logical_xor(pred_dist_fg <= boundary_width, pred_dist_bg > boundary_width)
        edge_inter = int(np.logical_and(gt_band, pred_band).sum(dtype=np.uint32))
        edge_union = int(np.logical_or(gt_band, pred_band).sum(dtype=np.uint32))
        if edge_union > 0:
            s_edge = float(edge_inter / edge_union)
        else:
            s_edge = float("nan")

    struct_valid = bool(np.isfinite(s_persist) and np.isfinite(s_edge))
    if struct_valid:
        s_struct = float(np.clip(np.sqrt(max(s_persist, 0.0) * max(s_edge, 0.0)), 0.0, 1.0))
    else:
        s_struct = float("nan")

    return {
        "persist_score": float(np.clip(s_persist, 0.0, 1.0)) if np.isfinite(s_persist) else float("nan"),
        "edge_score": float(np.clip(s_edge, 0.0, 1.0)) if np.isfinite(s_edge) else float("nan"),
        "struct_score": s_struct,
        "persist_inter": persist_inter,
        "persist_union": persist_union,
        "edge_inter": edge_inter,
        "edge_union": edge_union,
        "persist_valid": persist_valid,
        "persist_class_valid": bool(np.any(persist_valid)),
        "edge_class_valid": bool(edge_union > 0),
        "struct_valid": struct_valid,
    }


def _compute_class_score_components(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
) -> Tuple[float, float, float]:
    """Compute persistence, edge, and structural score with shared distance maps."""
    pad = max(max_radius, boundary_width)
    gt_dist_bg, gt_dist_fg = _prepare_mask_distance_maps(gt_mask, pad=pad, kernel_type=kernel_type)
    pred_dist_bg, pred_dist_fg = _prepare_mask_distance_maps(pred_mask, pad=pad, kernel_type=kernel_type)

    s_persist = _persistence_score_from_distances(gt_dist_bg, pred_dist_bg, max_radius=max_radius)

    if boundary_width <= 0:
        s_edge = 0.0
    else:
        gt_band = np.logical_xor(gt_dist_fg <= boundary_width, gt_dist_bg > boundary_width)
        pred_band = np.logical_xor(pred_dist_fg <= boundary_width, pred_dist_bg > boundary_width)
        gt_nonempty = bool(gt_band.any())
        pred_nonempty = bool(pred_band.any())
        if (not gt_nonempty) and (not pred_nonempty):
            s_edge = 0.0
        else:
            s_edge = binary_iou(gt_band, pred_band)

    s_struct = float(np.clip(np.sqrt(max(s_persist, 0.0) * max(s_edge, 0.0)), 0.0, 1.0))
    return float(np.clip(s_persist, 0.0, 1.0)), float(np.clip(s_edge, 0.0, 1.0)), s_struct


def _erode_from_distance(mask: np.ndarray, radius: int, kernel_type: str = "disk") -> np.ndarray:
    """Exact erosion via distance transform, matching border_value=0 semantics."""
    mask = mask.astype(bool, copy=False)
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dist_to_bg = _distance_transform(padded, kernel_type=kernel_type)
    return dist_to_bg[radius:-radius, radius:-radius] > radius


def _dilate_from_distance(mask: np.ndarray, radius: int, kernel_type: str = "disk") -> np.ndarray:
    """Exact dilation via distance transform, matching border_value=0 semantics."""
    mask = mask.astype(bool, copy=False)
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return mask.copy()

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dist_to_fg = _distance_transform(~padded, kernel_type=kernel_type)
    return dist_to_fg[radius:-radius, radius:-radius] <= radius


def binary_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute IoU between two binary masks.

    Args:
        a: Binary-like array of shape (H, W).
        b: Binary-like array of shape (H, W).
        eps: Small constant to avoid division instability.

    Returns:
        IoU in [0, 1]. If both masks are empty, returns 1.0.
    """
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    return float(inter / (union + eps))


def erode_mask(mask: np.ndarray, radius: int, kernel_type: str = "disk") -> np.ndarray:
    """Erode a binary mask with a disk or square structuring element.

    Args:
        mask: Binary-like array of shape (H, W).
        radius: Erosion radius. `0` returns a copy of the input.
        kernel_type: Either `"disk"` or `"square"`.

    Returns:
        Eroded boolean mask of shape (H, W).
    """
    return _erode_from_distance(mask, radius, kernel_type=kernel_type)


def _dilate_mask(mask: np.ndarray, radius: int, kernel_type: str = "disk") -> np.ndarray:
    """Dilate a binary mask with a disk or square structuring element."""
    return _dilate_from_distance(mask, radius, kernel_type=kernel_type)


def boundary_band(mask: np.ndarray, delta: int, kernel_type: str = "disk") -> np.ndarray:
    """Build the boundary band B_delta(M) = Dilate(M, delta) XOR Erode(M, delta).

    Args:
        mask: Binary-like array of shape (H, W).
        delta: Boundary width. `0` returns an all-false mask.
        kernel_type: Either `"disk"` or `"square"`.

    Returns:
        Boolean boundary-band mask of shape (H, W).
    """
    if delta < 0:
        raise ValueError("delta must be >= 0")
    mask = mask.astype(bool, copy=False)
    d = _dilate_from_distance(mask, radius=delta, kernel_type=kernel_type)
    e = _erode_from_distance(mask, radius=delta, kernel_type=kernel_type)
    return np.logical_xor(d, e)


def compute_class_persistence_score(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    max_radius: int = 3,
    kernel_type: str = "disk",
) -> float:
    """Compute the class-wise persistence term.

    The score follows the definition:
        S_persist(c) = mean_r IoU(Erode(G_c, r), Erode(P_c, r))

    where:
    - r ranges from 0 to `max_radius`
    - scales where both eroded masks are empty are skipped
    - if all scales are skipped, the score is defined as 0

    Args:
        gt_mask: Binary GT mask for one class.
        pred_mask: Binary prediction mask for one class.
        max_radius: Maximum erosion radius.
        kernel_type: Either `"disk"` or `"square"`.

    Returns:
        Persistence score in [0, 1].
    """
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")

    s_persist, _, _ = _compute_class_score_components(
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        max_radius=max_radius,
        boundary_width=0,
        kernel_type=kernel_type,
    )
    return s_persist


def compute_class_edge_score(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    boundary_width: int = 1,
    kernel_type: str = "disk",
) -> float:
    """Compute the class-wise edge agreement term.

    The score follows the definition:
        S_edge(c) = IoU(B_delta(G_c), B_delta(P_c))

    If both boundary bands are empty, the score is defined as 0.

    Args:
        gt_mask: Binary GT mask for one class.
        pred_mask: Binary prediction mask for one class.
        boundary_width: Boundary band width `delta`.
        kernel_type: Either `"disk"` or `"square"`.

    Returns:
        Edge score in [0, 1].
    """
    if boundary_width < 0:
        raise ValueError("boundary_width must be >= 0")

    _, s_edge, _ = _compute_class_score_components(
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        max_radius=0,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )
    return s_edge


def compute_class_struct_score(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    max_radius: int = 3,
    boundary_width: int = 1,
    kernel_type: str = "disk",
) -> float:
    """Compute the class-wise structural score.

    The score is defined as:
        S_struct(c) = sqrt(S_persist(c) * S_edge(c))

    Args:
        gt_mask: Binary GT mask for one class.
        pred_mask: Binary prediction mask for one class.
        max_radius: Maximum erosion radius for persistence.
        boundary_width: Boundary width for edge agreement.
        kernel_type: Either `"disk"` or `"square"`.

    Returns:
        Structural score in [0, 1].
    """
    _, _, s_struct = _compute_class_score_components(
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        max_radius=max_radius,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )
    return s_struct


def compute_multiclass_struct_score(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    max_radius: int = 3,
    boundary_width: int = 1,
    kernel_type: str = "disk",
) -> Dict[str, Union[float, np.ndarray, List[int]]]:
    """Compute CSP for one multi-class segmentation pair.

    Evaluation is class-wise. For each class `c`, this function constructs
    binary masks:
        G_c = (gt == c)
        P_c = (pred == c)

    Pixels with `gt == ignore_index` are removed from both masks.
    A class contributes to the final structural score only when both of the
    following are valid on this image:
    - at least one erosion scale has non-zero persistence union
    - the boundary-band union is non-zero

    Args:
        gt: Integer GT label map of shape (H, W).
        pred: Integer prediction label map of shape (H, W).
        num_classes: Number of semantic classes.
        ignore_index: Ignored GT label value.
        max_radius: Maximum erosion radius for persistence.
        boundary_width: Boundary width for edge agreement.
        kernel_type: Either `"disk"` or `"square"`.

    Returns:
        A dict containing:
        - `image_score`: image-level CSP
        - `image_persist_mean`: image-level mean persistence over persistence-valid classes
        - `image_edge_mean`: image-level mean edge score over edge-valid classes
        - `per_class_struct`: per-class structural score, NaN for invalid classes
        - `per_class_persist`: per-class persistence score, NaN for invalid classes
        - `per_class_edge`: per-class edge score, NaN for invalid classes
        - `per_class_valid`: per-class structural validity mask
        - `per_class_persist_valid`: per-class persistence validity mask
        - `per_class_edge_valid`: per-class edge validity mask
        - `per_class_persist_inter`: per-class per-radius persistence intersections
        - `per_class_persist_union`: per-class per-radius persistence unions
        - `per_class_edge_inter`: per-class boundary intersections
        - `per_class_edge_union`: per-class boundary unions
        - `valid_classes`: list of structure-valid class ids
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    if num_classes <= 0:
        raise ValueError("num_classes must be > 0")

    gt = np.asarray(gt)
    pred = np.asarray(pred)

    valid_pixels = (gt != ignore_index)
    gt_present = np.unique(gt[valid_pixels])
    pred_present = np.unique(pred[valid_pixels])
    active_classes = np.union1d(gt_present, pred_present)
    active_classes = active_classes[(active_classes >= 0) & (active_classes < num_classes)].astype(np.int64)

    per_class_struct = np.full((num_classes,), np.nan, dtype=np.float64)
    per_class_persist = np.full((num_classes,), np.nan, dtype=np.float64)
    per_class_edge = np.full((num_classes,), np.nan, dtype=np.float64)
    per_class_valid = np.zeros((num_classes,), dtype=bool)
    per_class_persist_valid = np.zeros((num_classes,), dtype=bool)
    per_class_edge_valid = np.zeros((num_classes,), dtype=bool)
    per_class_persist_inter = np.zeros((num_classes, max_radius + 1), dtype=np.uint32)
    per_class_persist_union = np.zeros((num_classes, max_radius + 1), dtype=np.uint32)
    per_class_edge_inter = np.zeros((num_classes,), dtype=np.uint32)
    per_class_edge_union = np.zeros((num_classes,), dtype=np.uint32)
    valid_classes: List[int] = []

    for c in active_classes.tolist():
        g_c = (gt == c) & valid_pixels
        p_c = (pred == c) & valid_pixels

        class_valid = bool(g_c.any() or p_c.any())
        if not class_valid:
            continue

        details = _compute_class_score_details(
            gt_mask=g_c,
            pred_mask=p_c,
            max_radius=max_radius,
            boundary_width=boundary_width,
            kernel_type=kernel_type,
        )

        per_class_persist_inter[c] = details["persist_inter"]
        per_class_persist_union[c] = details["persist_union"]
        per_class_edge_inter[c] = details["edge_inter"]
        per_class_edge_union[c] = details["edge_union"]
        per_class_persist_valid[c] = details["persist_class_valid"]
        per_class_edge_valid[c] = details["edge_class_valid"]

        if per_class_persist_valid[c]:
            per_class_persist[c] = float(details["persist_score"])
        if per_class_edge_valid[c]:
            per_class_edge[c] = float(details["edge_score"])
        if details["struct_valid"]:
            per_class_valid[c] = True
            valid_classes.append(c)
            per_class_struct[c] = float(details["struct_score"])

    persist_finite = np.isfinite(per_class_persist)
    edge_finite = np.isfinite(per_class_edge)

    if len(valid_classes) == 0:
        image_score = 0.0
    else:
        image_score = float(np.nanmean(per_class_struct[per_class_valid]))

    if np.any(persist_finite):
        image_persist_mean = float(np.nanmean(per_class_persist[persist_finite]))
    else:
        image_persist_mean = 0.0

    if np.any(edge_finite):
        image_edge_mean = float(np.nanmean(per_class_edge[edge_finite]))
    else:
        image_edge_mean = 0.0

    return {
        "image_score": float(np.clip(image_score, 0.0, 1.0)),
        "image_persist_mean": float(np.clip(image_persist_mean, 0.0, 1.0)),
        "image_edge_mean": float(np.clip(image_edge_mean, 0.0, 1.0)),
        "per_class_struct": per_class_struct,
        "per_class_persist": per_class_persist,
        "per_class_edge": per_class_edge,
        "per_class_valid": per_class_valid,
        "per_class_persist_valid": per_class_persist_valid,
        "per_class_edge_valid": per_class_edge_valid,
        "per_class_persist_inter": per_class_persist_inter,
        "per_class_persist_union": per_class_persist_union,
        "per_class_edge_inter": per_class_edge_inter,
        "per_class_edge_union": per_class_edge_union,
        "valid_classes": valid_classes,
    }


def _aggregate_struct_counts(
    persist_inter_sum: np.ndarray,
    persist_union_sum: np.ndarray,
    edge_inter_sum: np.ndarray,
    edge_union_sum: np.ndarray,
) -> Dict[str, Union[float, np.ndarray]]:
    """Aggregate exact class-wise intersections/unions into mIoU-style PES metrics."""
    num_classes = persist_inter_sum.shape[0]

    per_class_persist = np.full((num_classes,), np.nan, dtype=np.float64)
    per_class_edge = np.full((num_classes,), np.nan, dtype=np.float64)
    per_class_struct = np.full((num_classes,), np.nan, dtype=np.float64)

    per_class_persist_valid = np.any(persist_union_sum > 0, axis=1)
    per_class_edge_valid = edge_union_sum > 0
    per_class_valid = per_class_persist_valid & per_class_edge_valid

    for class_id in range(num_classes):
        valid_r = persist_union_sum[class_id] > 0
        if np.any(valid_r):
            per_class_persist[class_id] = np.mean(
                persist_inter_sum[class_id, valid_r] / persist_union_sum[class_id, valid_r]
            )
        if per_class_edge_valid[class_id]:
            per_class_edge[class_id] = edge_inter_sum[class_id] / edge_union_sum[class_id]
        if per_class_valid[class_id]:
            per_class_struct[class_id] = np.sqrt(
                max(per_class_persist[class_id], 0.0) * max(per_class_edge[class_id], 0.0)
            )

    persist_score = float(np.nanmean(per_class_persist[per_class_persist_valid])) if np.any(per_class_persist_valid) else 0.0
    edge_score = float(np.nanmean(per_class_edge[per_class_edge_valid])) if np.any(per_class_edge_valid) else 0.0
    score = float(np.nanmean(per_class_struct[per_class_valid])) if np.any(per_class_valid) else 0.0

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "persist_score": float(np.clip(persist_score, 0.0, 1.0)),
        "edge_score": float(np.clip(edge_score, 0.0, 1.0)),
        "reduction": "miou_style",
        "per_class_mean": per_class_struct,
        "per_class_persist_mean": per_class_persist,
        "per_class_edge_mean": per_class_edge,
        "per_class_valid": per_class_valid,
        "per_class_persist_valid": per_class_persist_valid,
        "per_class_edge_valid": per_class_edge_valid,
    }


def _aggregate_struct_dataset_results(
    per_image_results: List[Dict[str, Union[float, np.ndarray, List[int]]]],
    num_classes: int,
    reduction: str,
) -> Dict[str, Union[float, np.ndarray]]:
    """Aggregate per-image structural results into image_mean or class_mean metrics."""
    reduction = reduction.lower()
    if reduction not in ("image_mean", "class_mean"):
        raise ValueError("reduction must be 'image_mean' or 'class_mean'")

    n = len(per_image_results)
    image_scores = np.zeros((n,), dtype=np.float64)
    image_persist_scores = np.zeros((n,), dtype=np.float64)
    image_edge_scores = np.zeros((n,), dtype=np.float64)
    all_per_class = np.full((n, num_classes), np.nan, dtype=np.float64)
    all_per_class_persist = np.full((n, num_classes), np.nan, dtype=np.float64)
    all_per_class_edge = np.full((n, num_classes), np.nan, dtype=np.float64)
    all_per_class_valid = np.zeros((n, num_classes), dtype=bool)
    all_per_class_persist_valid = np.zeros((n, num_classes), dtype=bool)
    all_per_class_edge_valid = np.zeros((n, num_classes), dtype=bool)

    for i, res in enumerate(per_image_results):
        image_scores[i] = res["image_score"]
        image_persist_scores[i] = res["image_persist_mean"]
        image_edge_scores[i] = res["image_edge_mean"]
        all_per_class[i] = res["per_class_struct"]
        all_per_class_persist[i] = res["per_class_persist"]
        all_per_class_edge[i] = res["per_class_edge"]
        all_per_class_valid[i] = res["per_class_valid"]
        all_per_class_persist_valid[i] = res["per_class_persist_valid"]
        all_per_class_edge_valid[i] = res["per_class_edge_valid"]

    if reduction == "image_mean":
        score = float(np.mean(image_scores))
        persist_score = float(np.mean(image_persist_scores))
        edge_score = float(np.mean(image_edge_scores))
        per_class_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        per_class_persist_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        per_class_edge_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        per_class_valid = np.zeros((num_classes,), dtype=bool)
        per_class_persist_valid = np.zeros((num_classes,), dtype=bool)
        per_class_edge_valid = np.zeros((num_classes,), dtype=bool)
    else:
        per_class_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        per_class_persist_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        per_class_edge_mean = np.full((num_classes,), np.nan, dtype=np.float64)
        per_class_valid = np.zeros((num_classes,), dtype=bool)
        per_class_persist_valid = np.zeros((num_classes,), dtype=bool)
        per_class_edge_valid = np.zeros((num_classes,), dtype=bool)
        for c in range(num_classes):
            valid_idx = all_per_class_valid[:, c]
            persist_valid_idx = all_per_class_persist_valid[:, c]
            edge_valid_idx = all_per_class_edge_valid[:, c]
            if np.any(valid_idx):
                per_class_mean[c] = np.mean(all_per_class[valid_idx, c])
                per_class_valid[c] = True
            if np.any(persist_valid_idx):
                per_class_persist_mean[c] = np.mean(all_per_class_persist[persist_valid_idx, c])
                per_class_persist_valid[c] = True
            if np.any(edge_valid_idx):
                per_class_edge_mean[c] = np.mean(all_per_class_edge[edge_valid_idx, c])
                per_class_edge_valid[c] = True

        score = float(np.nanmean(per_class_mean[per_class_valid])) if np.any(per_class_valid) else 0.0
        persist_score = float(np.nanmean(per_class_persist_mean[per_class_persist_valid])) if np.any(per_class_persist_valid) else 0.0
        edge_score = float(np.nanmean(per_class_edge_mean[per_class_edge_valid])) if np.any(per_class_edge_valid) else 0.0

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "persist_score": float(np.clip(persist_score, 0.0, 1.0)),
        "edge_score": float(np.clip(edge_score, 0.0, 1.0)),
        "reduction": reduction,
        "image_scores": image_scores,
        "image_persist_scores": image_persist_scores,
        "image_edge_scores": image_edge_scores,
        "per_class_mean": per_class_mean,
        "per_class_persist_mean": per_class_persist_mean,
        "per_class_edge_mean": per_class_edge_mean,
        "per_class_valid": per_class_valid,
        "per_class_persist_valid": per_class_persist_valid,
        "per_class_edge_valid": per_class_edge_valid,
    }


def evaluate_dataset_struct_score(
    gt_list: List[np.ndarray],
    pred_list: List[np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
    max_radius: int = 3,
    boundary_width: int = 1,
    kernel_type: str = "disk",
    reduction: str = "image_mean",
) -> Dict[str, Union[float, np.ndarray]]:
    """Evaluate CSP on a dataset of label-map pairs.

    Supports three dataset reductions:
    - `image_mean`: average image-level scores over all images
    - `class_mean`: average per-class structural scores over images first,
      then average over valid classes
    - `miou_style`: accumulate class-wise intersections/unions over the dataset
      first, then apply the PES-mIoU class macro-average

    Args:
        gt_list: List of GT label maps.
        pred_list: List of prediction label maps.
        num_classes: Number of semantic classes.
        ignore_index: Ignored GT label value.
        max_radius: Maximum erosion radius for persistence.
        boundary_width: Boundary width for edge agreement.
        kernel_type: Either `"disk"` or `"square"`.
        reduction: One of `"image_mean"`, `"class_mean"`, or `"miou_style"`.

    Returns:
        A dict containing:
        - `score`: final dataset-level CSP score
        - `persist_score`: final dataset-level persistence score
        - `edge_score`: final dataset-level edge score
        - `reduction`: reduction mode
        - `image_scores`: per-image CSP scores
        - `image_persist_scores`: per-image persistence means
        - `image_edge_scores`: per-image edge means
        - `per_class_mean`: per-class CSP means for `class_mean`,
          otherwise all-NaN
        - `per_class_persist_mean`: per-class persistence means
        - `per_class_edge_mean`: per-class edge means
    """
    if len(gt_list) != len(pred_list):
        raise ValueError("gt_list and pred_list must have the same length")
    if len(gt_list) == 0:
        raise ValueError("Empty dataset: gt_list/pred_list must be non-empty")

    reduction = reduction.lower()
    if reduction not in ("image_mean", "class_mean", "miou_style"):
        raise ValueError("reduction must be 'image_mean', 'class_mean', or 'miou_style'")

    n = len(gt_list)
    image_scores = np.zeros((n,), dtype=np.float64)
    image_persist_scores = np.zeros((n,), dtype=np.float64)
    image_edge_scores = np.zeros((n,), dtype=np.float64)
    per_image_results: List[Dict[str, Union[float, np.ndarray, List[int]]]] = []

    for i, (gt, pred) in enumerate(zip(gt_list, pred_list)):
        res = compute_multiclass_struct_score(
            gt,
            pred,
            num_classes=num_classes,
            ignore_index=ignore_index,
            max_radius=max_radius,
            boundary_width=boundary_width,
            kernel_type=kernel_type,
        )
        per_image_results.append(res)
        image_scores[i] = res["image_score"]
        image_persist_scores[i] = res["image_persist_mean"]
        image_edge_scores[i] = res["image_edge_mean"]

    if reduction in ("image_mean", "class_mean"):
        aggregated = _aggregate_struct_dataset_results(per_image_results, num_classes, reduction)
    else:
        persist_inter_sum = np.zeros((num_classes, max_radius + 1), dtype=np.uint64)
        persist_union_sum = np.zeros((num_classes, max_radius + 1), dtype=np.uint64)
        edge_inter_sum = np.zeros((num_classes,), dtype=np.uint64)
        edge_union_sum = np.zeros((num_classes,), dtype=np.uint64)
        for res in per_image_results:
            persist_inter_sum += np.asarray(res["per_class_persist_inter"], dtype=np.uint64)
            persist_union_sum += np.asarray(res["per_class_persist_union"], dtype=np.uint64)
            edge_inter_sum += np.asarray(res["per_class_edge_inter"], dtype=np.uint64)
            edge_union_sum += np.asarray(res["per_class_edge_union"], dtype=np.uint64)
        aggregated = _aggregate_struct_counts(persist_inter_sum, persist_union_sum, edge_inter_sum, edge_union_sum)
        aggregated["image_scores"] = image_scores
        aggregated["image_persist_scores"] = image_persist_scores
        aggregated["image_edge_scores"] = image_edge_scores

    return aggregated


def _evaluate_pair_from_paths(
    pred_path: Union[str, Path],
    gt_path: Union[str, Path],
    num_classes: int,
    ignore_index: int,
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
) -> Tuple[Dict[str, Union[float, np.ndarray, List[int]]], np.ndarray]:
    """Worker helper for directory-based parallel evaluation."""
    pred = load_label_map(pred_path)
    gt = load_label_map(gt_path)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt_path} gt={gt.shape}, pred={pred.shape}")

    struct_res = compute_multiclass_struct_score(
        gt=gt,
        pred=pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_radius=max_radius,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    update_confusion_matrix(confusion, gt, pred, num_classes, ignore_index)
    return struct_res, confusion


def load_label_map(label_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load a single-channel integer label map from path or ndarray."""
    if isinstance(label_input, np.ndarray):
        arr = label_input
    else:
        p = Path(label_input)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        arr = np.load(str(p)) if p.suffix.lower() == ".npy" else np.array(Image.open(str(p)))

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Label must be single-channel index map, got {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Label must be HxW, got {arr.shape}")

    return arr.astype(np.int64, copy=False)


def update_confusion_matrix(
    confusion: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> None:
    """Accumulate a standard semantic-segmentation confusion matrix."""
    valid = (gt != ignore_index)
    valid &= (gt >= 0) & (gt < num_classes)
    valid &= (pred >= 0) & (pred < num_classes)
    if not np.any(valid):
        return

    inds = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    confusion += np.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def miou_from_confusion(confusion: np.ndarray) -> Dict[str, Union[float, np.ndarray, List[int]]]:
    """Compute mIoU and per-class IoU from a confusion matrix."""
    tp = np.diag(confusion).astype(np.float64)
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - tp

    per_class_iou = np.full_like(tp, np.nan, dtype=np.float64)
    valid = union > 0
    per_class_iou[valid] = tp[valid] / np.maximum(union[valid], 1e-12)

    miou = float(np.nanmean(per_class_iou)) if np.any(valid) else 0.0
    return {
        "miou": float(np.clip(miou, 0.0, 1.0)),
        "per_class_iou": per_class_iou,
        "valid_classes": np.where(valid)[0].tolist(),
    }


def compute_pair_metrics(
    gt_input: Union[str, Path, np.ndarray],
    pred_input: Union[str, Path, np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
    max_radius: int = 3,
    boundary_width: int = 1,
    kernel_type: str = "disk",
) -> Dict[str, Union[float, np.ndarray, List[int]]]:
    """Compute CSP and mIoU for one GT/pred pair."""
    gt = load_label_map(gt_input)
    pred = load_label_map(pred_input)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt_input}={gt.shape}, {pred_input}={pred.shape}")

    struct_res = compute_multiclass_struct_score(
        gt=gt,
        pred=pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_radius=max_radius,
        boundary_width=boundary_width,
        kernel_type=kernel_type,
    )

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    update_confusion_matrix(conf, gt, pred, num_classes, ignore_index)
    miou_res = miou_from_confusion(conf)

    out = dict(struct_res)
    out["miou"] = miou_res["miou"]
    out["per_class_iou"] = miou_res["per_class_iou"]
    out["miou_valid_classes"] = miou_res["valid_classes"]
    return out


def _list_label_files(folder: Union[str, Path]) -> List[Path]:
    """List supported label-map files in a directory."""
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _match_pred_gt_files(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    preview_pairs: int = 10,
) -> List[Tuple[Path, Path]]:
    """Match prediction files to GT files by `*_pred_mask.png` naming."""
    pred_files = _list_label_files(pred_dir)
    gt_files = _list_label_files(gt_dir)

    # 仅用 *_pred_mask.png 作为预测图
    pred_candidates = [p for p in pred_files if p.name.endswith("_pred_mask.png")]
    if len(pred_candidates) == 0:
        raise RuntimeError(f"No *_pred_mask.png found in pred_dir={pred_dir}")

    gt_by_stem = {p.stem: p for p in gt_files}

    pairs: List[Tuple[Path, Path]] = []
    used_gt = set()

    for pp in pred_candidates:
        key = pp.stem[:-len("_pred_mask")] if pp.stem.endswith("_pred_mask") else pp.stem
        gp = gt_by_stem.get(key, None)
        if gp is not None and gp not in used_gt:
            pairs.append((pp, gp))
            used_gt.add(gp)

    if not pairs:
        raise RuntimeError(
            f"No matched files between pred_dir={pred_dir} and gt_dir={gt_dir} "
            f"using *_pred_mask.png -> GT stem mapping."
        )

    print(f"[Match] using *_pred_mask.png, paired files: {len(pairs)}")
    if preview_pairs and len(pairs) > 0:
        print("[Match] preview pairs (pred -> gt):")
        for pp, gp in pairs[:preview_pairs]:
            print(f"  {pp.name} -> {gp.name}")

    return pairs


def evaluate_dataset_from_dirs(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    num_classes: int,
    ignore_index: int = 255,
    max_radius: int = 3,
    boundary_width: int = 1,
    kernel_type: str = "disk",
    preview_pairs: int = 10,
    num_workers: int = 1,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Evaluate CSP and mIoU from prediction/GT directories."""
    pairs = _match_pred_gt_files(pred_dir, gt_dir, preview_pairs=preview_pairs)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_results: List[Dict[str, Union[float, np.ndarray, List[int]]]] = []

    if num_workers <= 1:
        for pred_path, gt_path in pairs:
            struct_res, pair_confusion = _evaluate_pair_from_paths(
                pred_path=pred_path,
                gt_path=gt_path,
                num_classes=num_classes,
                ignore_index=ignore_index,
                max_radius=max_radius,
                boundary_width=boundary_width,
                kernel_type=kernel_type,
            )
            per_image_results.append(struct_res)
            confusion += pair_confusion
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_pair_from_paths,
                    pred_path,
                    gt_path,
                    num_classes,
                    ignore_index,
                    max_radius,
                    boundary_width,
                    kernel_type,
                )
                for pred_path, gt_path in pairs
            ]
            for future in futures:
                struct_res, pair_confusion = future.result()
                per_image_results.append(struct_res)
                confusion += pair_confusion

    csp_image_mean = _aggregate_struct_dataset_results(per_image_results, num_classes, "image_mean")
    csp_class_mean = _aggregate_struct_dataset_results(per_image_results, num_classes, "class_mean")
    max_radius_local = per_image_results[0]["per_class_persist_inter"].shape[1] - 1 if per_image_results else max_radius
    persist_inter_sum = np.zeros((num_classes, max_radius_local + 1), dtype=np.uint64)
    persist_union_sum = np.zeros((num_classes, max_radius_local + 1), dtype=np.uint64)
    edge_inter_sum = np.zeros((num_classes,), dtype=np.uint64)
    edge_union_sum = np.zeros((num_classes,), dtype=np.uint64)
    for res in per_image_results:
        persist_inter_sum += np.asarray(res["per_class_persist_inter"], dtype=np.uint64)
        persist_union_sum += np.asarray(res["per_class_persist_union"], dtype=np.uint64)
        edge_inter_sum += np.asarray(res["per_class_edge_inter"], dtype=np.uint64)
        edge_union_sum += np.asarray(res["per_class_edge_union"], dtype=np.uint64)
    csp_miou_style = _aggregate_struct_counts(persist_inter_sum, persist_union_sum, edge_inter_sum, edge_union_sum)
    miou_res = miou_from_confusion(confusion)

    return {
        "num_images": len(pairs),
        "csp_image_mean": csp_image_mean["score"],
        "persist_image_mean": csp_image_mean["persist_score"],
        "edge_image_mean": csp_image_mean["edge_score"],
        "csp_class_mean": csp_class_mean["score"],
        "persist_class_mean": csp_class_mean["persist_score"],
        "edge_class_mean": csp_class_mean["edge_score"],
        "csp_image_scores": csp_image_mean["image_scores"],
        "persist_image_scores": csp_image_mean["image_persist_scores"],
        "edge_image_scores": csp_image_mean["image_edge_scores"],
        "csp_per_class_struct_mean": csp_class_mean["per_class_mean"],
        "csp_per_class_persist_mean": csp_class_mean["per_class_persist_mean"],
        "csp_per_class_edge_mean": csp_class_mean["per_class_edge_mean"],
        "csp_miou_style": csp_miou_style["score"],
        "persist_miou_style": csp_miou_style["persist_score"],
        "edge_miou_style": csp_miou_style["edge_score"],
        "csp_per_class_struct_miou_style": csp_miou_style["per_class_mean"],
        "csp_per_class_persist_miou_style": csp_miou_style["per_class_persist_mean"],
        "csp_per_class_edge_miou_style": csp_miou_style["per_class_edge_mean"],
        "miou": miou_res["miou"],
        "per_class_iou": miou_res["per_class_iou"],
        "miou_valid_classes": miou_res["valid_classes"],
    }


def _make_demo_label_maps() -> Tuple[np.ndarray, np.ndarray]:
    """Create a tiny multi-class example for quick manual inspection."""
    gt = np.zeros((8, 8), dtype=np.int64)
    pred = np.zeros((8, 8), dtype=np.int64)

    gt[1:4, 1:4] = 1
    gt[4:7, 4:7] = 2

    pred[1:4, 2:5] = 1
    pred[4:7, 4:7] = 2
    pred[5, 5] = 0  # carve a tiny hole inside class 2

    return gt, pred


def run_csp_minimal_example() -> None:
    """Run a minimal multi-class CSP example and print the results."""
    gt, pred = _make_demo_label_maps()
    res = compute_multiclass_struct_score(
        gt=gt,
        pred=pred,
        num_classes=3,
        ignore_index=255,
        max_radius=2,
        boundary_width=1,
        kernel_type="disk",
    )
    dataset_image_mean = evaluate_dataset_struct_score(
        [gt],
        [pred],
        num_classes=3,
        reduction="image_mean",
    )
    dataset_class_mean = evaluate_dataset_struct_score(
        [gt],
        [pred],
        num_classes=3,
        reduction="class_mean",
    )
    dataset_miou_style = evaluate_dataset_struct_score(
        [gt],
        [pred],
        num_classes=3,
        reduction="miou_style",
    )

    print("[Example] GT label map:")
    print(gt)
    print("[Example] Pred label map:")
    print(pred)
    print(f"[Example] Image CSP: {res['image_score']:.6f}")
    print(f"[Example] Image Persistence Mean: {res['image_persist_mean']:.6f}")
    print(f"[Example] Image Edge Mean: {res['image_edge_mean']:.6f}")
    print(f"[Example] Per-class CSP: {res['per_class_struct']}")
    print(f"[Example] Per-class Persistence: {res['per_class_persist']}")
    print(f"[Example] Per-class Edge: {res['per_class_edge']}")
    print(f"[Example] Valid classes: {res['valid_classes']}")
    print(f"[Example] Dataset CSP(image_mean): {dataset_image_mean['score']:.6f}")
    print(f"[Example] Dataset Persistence(image_mean): {dataset_image_mean['persist_score']:.6f}")
    print(f"[Example] Dataset Edge(image_mean): {dataset_image_mean['edge_score']:.6f}")
    print(f"[Example] Dataset CSP(class_mean): {dataset_class_mean['score']:.6f}")
    print(f"[Example] Dataset Persistence(class_mean): {dataset_class_mean['persist_score']:.6f}")
    print(f"[Example] Dataset Edge(class_mean): {dataset_class_mean['edge_score']:.6f}")
    print(f"[Example] Dataset CSP(miou_style): {dataset_miou_style['score']:.6f}")
    print(f"[Example] Dataset Persistence(miou_style): {dataset_miou_style['persist_score']:.6f}")
    print(f"[Example] Dataset Edge(miou_style): {dataset_miou_style['edge_score']:.6f}")


def run_csp_basic_tests() -> None:
    """Run a few lightweight assertions for the baseline CSP implementation."""
    gt = np.zeros((9, 9), dtype=np.int64)
    gt[2:7, 2:7] = 1

    pred_same = gt.copy()
    same_struct = compute_class_struct_score(gt == 1, pred_same == 1, max_radius=2, boundary_width=1)
    assert same_struct > 0.999, f"Expected identical masks to score near 1, got {same_struct}"

    pred_wrong = np.zeros_like(gt)
    pred_wrong[2:7, 2:7] = 2
    wrong_res = compute_multiclass_struct_score(gt, pred_wrong, num_classes=3)
    assert wrong_res["per_class_struct"][1] < 0.05, (
        f"Expected completely wrong class assignment to have low class-1 struct, got {wrong_res['per_class_struct'][1]}"
    )

    pred_missing = np.zeros_like(gt)
    missing_struct = compute_class_struct_score(gt == 1, pred_missing == 1, max_radius=2, boundary_width=1)
    assert missing_struct < 0.05, f"Expected missing class to score low, got {missing_struct}"

    pred_shift = np.zeros_like(gt)
    pred_shift[2:7, 3:8] = 1
    edge_same = compute_class_edge_score(gt == 1, pred_same == 1, boundary_width=1)
    edge_shift = compute_class_edge_score(gt == 1, pred_shift == 1, boundary_width=1)
    assert edge_shift < edge_same, f"Expected boundary shift to reduce edge score, got {edge_shift} vs {edge_same}"

    pred_hole = gt.copy()
    pred_hole[4, 4] = 0
    persist_same = compute_class_persistence_score(gt == 1, pred_same == 1, max_radius=2)
    persist_hole = compute_class_persistence_score(gt == 1, pred_hole == 1, max_radius=2)
    assert persist_hole < persist_same, (
        f"Expected interior hole to reduce persistence, got {persist_hole} vs {persist_same}"
    )

    print("[Tests] All CSP baseline assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CSP + mIoU evaluator")
    parser.add_argument("--mode", type=str, choices=["single", "dataset", "example", "test"], required=True)

    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--max-radius", type=int, default=3)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"])

    parser.add_argument("--pred", type=str, default=None, help="single mode: pred path")
    parser.add_argument("--gt", type=str, default=None, help="single mode: gt path")

    parser.add_argument("--pred-dir", type=str, default=None, help="dataset mode: pred dir")
    parser.add_argument("--gt-dir", type=str, default=None, help="dataset mode: gt dir")
    parser.add_argument("--preview-pairs", type=int, default=10, help="dataset mode: print first N pred->gt pairs")
    parser.add_argument("--num-workers", type=int, default=1, help="dataset mode: number of parallel workers")

    args = parser.parse_args()

    if args.mode == "example":
        run_csp_minimal_example()

    elif args.mode == "test":
        run_csp_basic_tests()

    elif args.mode == "single":
        if args.pred is None or args.gt is None:
            raise ValueError("single mode requires --pred and --gt")

        out = compute_pair_metrics(
            gt_input=args.gt,
            pred_input=args.pred,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            max_radius=args.max_radius,
            boundary_width=args.boundary_width,
            kernel_type=args.kernel_type,
        )
        print(f"[Single] CSP(image_score): {out['image_score']:.6f}")
        print(f"[Single] Persistence(image_mean): {out['image_persist_mean']:.6f}")
        print(f"[Single] Edge(image_mean): {out['image_edge_mean']:.6f}")
        print(f"[Single] mIoU: {out['miou']:.6f}")
        print(f"[Single] CSP valid classes: {out['valid_classes']}")
        print(f"[Single] Per-class CSP: {out['per_class_struct']}")
        print(f"[Single] Per-class Persistence: {out['per_class_persist']}")
        print(f"[Single] Per-class Edge: {out['per_class_edge']}")
        print(f"[Single] mIoU valid classes: {out['miou_valid_classes']}")

    else:
        if args.pred_dir is None or args.gt_dir is None:
            raise ValueError("dataset mode requires --pred-dir and --gt-dir")

        out = evaluate_dataset_from_dirs(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            max_radius=args.max_radius,
            boundary_width=args.boundary_width,
            kernel_type=args.kernel_type,
            preview_pairs=args.preview_pairs,
            num_workers=args.num_workers,
        )
        print(f"[Dataset] num_images: {out['num_images']}")
        print(f"[Dataset] CSP(image_mean): {out['csp_image_mean']:.6f}")
        print(f"[Dataset] Persistence(image_mean): {out['persist_image_mean']:.6f}")
        print(f"[Dataset] Edge(image_mean): {out['edge_image_mean']:.6f}")
        print(f"[Dataset] CSP(class_mean): {out['csp_class_mean']:.6f}")
        print(f"[Dataset] Persistence(class_mean): {out['persist_class_mean']:.6f}")
        print(f"[Dataset] Edge(class_mean): {out['edge_class_mean']:.6f}")
        print(f"[Dataset] CSP(miou_style): {out['csp_miou_style']:.6f}")
        print(f"[Dataset] Persistence(miou_style): {out['persist_miou_style']:.6f}")
        print(f"[Dataset] Edge(miou_style): {out['edge_miou_style']:.6f}")
        print(f"[Dataset] mIoU: {out['miou']:.6f}")
