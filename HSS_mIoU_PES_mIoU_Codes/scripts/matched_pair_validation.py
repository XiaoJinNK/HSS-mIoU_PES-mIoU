#!/usr/bin/env python3
"""
Objective matched-pair validation on real predictions.

This experiment is designed to validate the semantic and structural metrics
without subjective questionnaires:

1. Semantic matched-pair validation
   Mine image/model pairs where mIoU and StructScore are close, but semantic
   metrics disagree. Validate the preferred prediction using independent
   hierarchy-based proxies:
   - coarse-label mIoU curve / AUC across WordNet depths
   - mean error hop distance on the WordNet tree

2. Structural matched-pair validation
   Mine image/model pairs where mIoU and semantic score are close, but
   StructScore differs. Validate the preferred prediction using classic
   structure-oriented proxies:
   - boundary F1 with tolerance
   - component-count consistency

3. Overall multi-metric ranking
   Report image-level mean + bootstrap CI for mIoU, three semantic metrics,
   and structural metrics on the same real-prediction model set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
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
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import binomtest
from skimage.morphology import disk
from skimage.segmentation import find_boundaries

SCRIPTS_DIR = ROOT / "scripts"
METRICS_DIR = ROOT / "metrics"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(METRICS_DIR))

from evaluate_predictions import (  # type: ignore
    MatchedPair,
    colorize_label_map,
    compute_hier_soft_iou_from_confusion,
    compute_miou_from_confusion,
    ensure_dir,
    evaluate_image_pair,
    evaluate_image_pair_worker,
    init_image_worker,
    load_label_map,
    load_similarity_matrix,
    match_gt_pred_files,
    resolve_num_workers,
    safe_slug,
    save_csv,
    save_json,
    sparse_to_dense_confusion,
)

from coarsen import build_coarse_mapping, coarsen_mask, load_class_paths  # type: ignore  # noqa: E402


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
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

    def tqdm(iterable: Optional[Iterable[Any]] = None, **kwargs: Any) -> _DummyTqdm:
        return _DummyTqdm(iterable=iterable, **kwargs)


@dataclass
class SemanticProxy:
    coarse_auc: float
    coarse_gain_auc: float
    mean_hop: float
    far_error_ratio: float
    depth_scores: Dict[int, float]


@dataclass
class StructuralProxy:
    boundary_f1: float
    component_score: float
    classic_struct: float


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan")}
    if arr.size == 1:
        x = float(arr[0])
        return {"mean": x, "ci_low": x, "ci_high": x, "std": 0.0}

    rng = np.random.default_rng(seed)
    n = arr.size
    draws = np.empty((n_bootstrap,), dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = float(arr[idx].mean())

    alpha = 1.0 - confidence_level
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(draws, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
        "std": float(draws.std()),
    }


def bootstrap_rate_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan")}
    if arr.size == 1:
        x = float(arr[0])
        return {"mean": x, "ci_low": x, "ci_high": x, "std": 0.0}

    rng = np.random.default_rng(seed)
    n = arr.size
    draws = np.empty((n_bootstrap,), dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = float(arr[idx].mean())

    alpha = 1.0 - confidence_level
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(draws, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
        "std": float(draws.std()),
    }


def shared_prefix_length(path_a: Sequence[str], path_b: Sequence[str]) -> int:
    shared = 0
    for a, b in zip(path_a, path_b):
        if a != b:
            break
        shared += 1
    return shared


def build_hop_distance_matrix(class_paths: Dict[int, List[str]], num_classes: int) -> np.ndarray:
    dist = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        path_i = class_paths[i]
        for j in range(i + 1, num_classes):
            path_j = class_paths[j]
            shared = shared_prefix_length(path_i, path_j)
            edge_dist = (len(path_i) - shared) + (len(path_j) - shared)
            dist[i, j] = edge_dist
            dist[j, i] = edge_dist
    return dist


def compute_far_distance_threshold(hop_dist: np.ndarray) -> float:
    off_diag = hop_dist[~np.eye(hop_dist.shape[0], dtype=bool)]
    off_diag = off_diag[np.isfinite(off_diag)]
    if off_diag.size == 0:
        return 0.0
    return float(np.percentile(off_diag, 75))


def build_depth_mappings(
    class_paths: Dict[int, List[str]],
    depths: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    mappings: Dict[int, Dict[str, Any]] = {}
    for depth in depths:
        fine_to_coarse, _, coarse_names = build_coarse_mapping(class_paths, depth)
        mappings[int(depth)] = {
            "fine_to_coarse": fine_to_coarse,
            "num_coarse_classes": len(coarse_names),
            "coarse_names": coarse_names,
        }
    return mappings


def _boundary_mask(label_map: np.ndarray, ignore_index: int) -> np.ndarray:
    arr = label_map.astype(np.int64, copy=True)
    arr[arr == ignore_index] = -1
    boundary = find_boundaries(arr, mode="thick")
    boundary &= (arr != -1)
    return boundary


def boundary_f1_score(gt: np.ndarray, pred: np.ndarray, ignore_index: int, tolerance: int) -> float:
    gt_b = _boundary_mask(gt, ignore_index)
    pred_b = _boundary_mask(pred, ignore_index)
    if not gt_b.any() and not pred_b.any():
        return 1.0
    if tolerance > 0:
        footprint = disk(tolerance)
        gt_d = ndi.binary_dilation(gt_b, structure=footprint)
        pred_d = ndi.binary_dilation(pred_b, structure=footprint)
    else:
        gt_d = gt_b
        pred_d = pred_b

    precision = float((pred_b & gt_d).sum() / max(1, int(pred_b.sum())))
    recall = float((gt_b & pred_d).sum() / max(1, int(gt_b.sum())))
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def component_consistency_score(gt: np.ndarray, pred: np.ndarray, ignore_index: int, num_classes: int) -> float:
    valid_gt = gt != ignore_index
    total_weight = 0.0
    accum = 0.0
    for class_id in range(num_classes):
        gt_mask = (gt == class_id) & valid_gt
        pred_mask = pred == class_id
        if not gt_mask.any() and not pred_mask.any():
            continue
        n_gt = int(ndi.label(gt_mask)[1]) if gt_mask.any() else 0
        n_pred = int(ndi.label(pred_mask)[1]) if pred_mask.any() else 0
        denom = max(1, n_gt, n_pred)
        score = 1.0 - (abs(n_pred - n_gt) / denom)
        weight = float(max(int(gt_mask.sum()), int(pred_mask.sum())))
        accum += score * weight
        total_weight += weight
    if total_weight <= 0.0:
        return float("nan")
    return float(accum / total_weight)


def compute_structural_proxy(
    gt: np.ndarray,
    pred: np.ndarray,
    ignore_index: int,
    num_classes: int,
    boundary_tolerance: int,
) -> StructuralProxy:
    boundary_f1 = boundary_f1_score(gt, pred, ignore_index=ignore_index, tolerance=boundary_tolerance)
    component_score = component_consistency_score(gt, pred, ignore_index=ignore_index, num_classes=num_classes)
    if not np.isfinite(component_score):
        component_score = 0.0
    classic_struct = float(math.sqrt(max(0.0, boundary_f1) * max(0.0, component_score)))
    return StructuralProxy(
        boundary_f1=float(boundary_f1),
        component_score=float(component_score),
        classic_struct=classic_struct,
    )


def compute_semantic_proxy(
    gt: np.ndarray,
    pred: np.ndarray,
    ignore_index: int,
    hop_dist: np.ndarray,
    far_distance_threshold: float,
    depth_mappings: Dict[int, Dict[str, Any]],
) -> SemanticProxy:
    valid = gt != ignore_index
    err = valid & (gt != pred)
    if err.any():
        distances = hop_dist[gt[err], pred[err]]
        mean_hop = float(np.mean(distances))
        far_error_ratio = float(np.mean(distances >= far_distance_threshold))
    else:
        mean_hop = 0.0
        far_error_ratio = 0.0

    depth_scores: Dict[int, float] = {}
    for depth, meta in depth_mappings.items():
        gt_coarse = coarsen_mask(gt, meta["fine_to_coarse"], ignore_label=ignore_index)
        pred_coarse = coarsen_mask(pred, meta["fine_to_coarse"], ignore_label=ignore_index)
        confusion = np.zeros((meta["num_coarse_classes"], meta["num_coarse_classes"]), dtype=np.int64)
        valid_coarse = gt_coarse != ignore_index
        gt_flat = gt_coarse[valid_coarse].reshape(-1)
        pred_flat = pred_coarse[valid_coarse].reshape(-1)
        idx = meta["num_coarse_classes"] * gt_flat + pred_flat
        bincount = np.bincount(idx, minlength=meta["num_coarse_classes"] ** 2)
        confusion[...] = bincount.reshape(meta["num_coarse_classes"], meta["num_coarse_classes"])
        miou = compute_miou_from_confusion(confusion)["miou"]
        depth_scores[int(depth)] = float(miou)

    score_values = [depth_scores[d] for d in sorted(depth_scores)]
    coarse_auc = float(np.mean(score_values)) if score_values else float("nan")
    base_miou = compute_miou_from_confusion(
        np.bincount(
            (gt[valid] * hop_dist.shape[0] + pred[valid]).reshape(-1),
            minlength=hop_dist.shape[0] ** 2,
        ).reshape(hop_dist.shape[0], hop_dist.shape[0])
    )["miou"]
    coarse_gain_auc = float(np.mean([score - base_miou for score in score_values])) if score_values else float("nan")
    return SemanticProxy(
        coarse_auc=coarse_auc,
        coarse_gain_auc=coarse_gain_auc,
        mean_hop=mean_hop,
        far_error_ratio=far_error_ratio,
        depth_scores=depth_scores,
    )


def plot_metric_ranking(overall_df: pd.DataFrame, metrics: Sequence[str], out_dir: Path) -> None:
    ensure_dir(out_dir)
    model_names = overall_df["pred_name"].tolist()
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 5))
        values = overall_df[metric].to_numpy(dtype=np.float64)
        ci_low = overall_df[f"{metric}_ci_low"].to_numpy(dtype=np.float64)
        ci_high = overall_df[f"{metric}_ci_high"].to_numpy(dtype=np.float64)
        x = np.arange(len(model_names))
        yerr = np.vstack([values - ci_low, ci_high - values])
        ax.bar(x, values, yerr=yerr, capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} (image-level mean with 95% CI)")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"ranking_{safe_slug(metric)}.png", dpi=160)
        plt.close(fig)


def plot_pair_gap_scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_path: Path) -> None:
    if df.empty:
        return
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df[xcol], df[ycol], s=12, alpha=0.35)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_semantic_alignment(summary_df: pd.DataFrame, proxy_key: str, out_path: Path) -> None:
    if summary_df.empty:
        return
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = summary_df.loc[summary_df["proxy"] == proxy_key].copy()
    if subset.empty:
        plt.close(fig)
        return
    x = np.arange(len(subset))
    values = subset["agreement_mean"].to_numpy(dtype=np.float64)
    ci_low = subset["agreement_ci_low"].to_numpy(dtype=np.float64)
    ci_high = subset["agreement_ci_high"].to_numpy(dtype=np.float64)
    yerr = np.vstack([values - ci_low, ci_high - values])
    ax.bar(x, values, yerr=yerr, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["metric"].tolist())
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement Rate")
    ax.set_title(f"Semantic Alignment ({proxy_key})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_structural_alignment(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(summary_df))
    values = summary_df["agreement_mean"].to_numpy(dtype=np.float64)
    ci_low = summary_df["agreement_ci_low"].to_numpy(dtype=np.float64)
    ci_high = summary_df["agreement_ci_high"].to_numpy(dtype=np.float64)
    yerr = np.vstack([values - ci_low, ci_high - values])
    ax.bar(x, values, yerr=yerr, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["proxy"].tolist())
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement Rate")
    ax.set_title("StructScore Alignment")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def copy_or_make_case_image(
    image_path: Path,
    label_map: np.ndarray,
    num_classes: int,
    ignore_index: int,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    if image_path.exists():
        Image.open(image_path).convert("RGB").save(out_path)
    else:
        Image.fromarray(colorize_label_map(label_map, num_classes=num_classes, ignore_index=ignore_index)).save(out_path)


def export_pair_cases(
    candidates: pd.DataFrame,
    out_dir: Path,
    gt_summary_dir: Path,
    num_classes: int,
    ignore_index: int,
    max_cases: int,
    metric_col: str,
    title_tag: str,
) -> List[Dict[str, Any]]:
    if candidates.empty:
        return []
    ensure_dir(out_dir)
    exported: List[Dict[str, Any]] = []
    top = candidates.sort_values(metric_col, ascending=False).head(max_cases)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        case_dir = ensure_dir(out_dir / f"{rank:02d}_{safe_slug(str(row['image_id']))}")
        gt_summary = gt_summary_dir / f"{row['image_id']}_gt_summary.json"
        gt_meta = json.load(open(gt_summary, "r", encoding="utf-8")) if gt_summary.exists() else {}
        gt_path = Path(row["gt_path"])
        pred_pref_path = Path(row["preferred_pred_path"])
        pred_other_path = Path(row["other_pred_path"])

        gt = load_label_map(gt_path, offset=0, ignore_index=ignore_index)
        pref = load_label_map(pred_pref_path, offset=0, ignore_index=ignore_index)
        other = load_label_map(pred_other_path, offset=0, ignore_index=ignore_index)

        copy_or_make_case_image(
            Path(str(gt_meta.get("image_path", ""))),
            gt,
            num_classes=num_classes,
            ignore_index=ignore_index,
            out_path=case_dir / "input_image.png",
        )
        Image.fromarray(colorize_label_map(gt, num_classes=num_classes, ignore_index=ignore_index)).save(case_dir / "gt_color.png")
        Image.fromarray(colorize_label_map(pref, num_classes=num_classes, ignore_index=ignore_index)).save(case_dir / "preferred_color.png")
        Image.fromarray(colorize_label_map(other, num_classes=num_classes, ignore_index=ignore_index)).save(case_dir / "other_color.png")

        summary = {
            "title_tag": title_tag,
            "image_id": row["image_id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "preferred_model": row["preferred_model"],
            "other_model": row["other_model"],
            "metric_gap": float(row[metric_col]),
            "metrics": {
                "miou_a": float(row["miou_a"]),
                "miou_b": float(row["miou_b"]),
                "hier_hybrid_a": float(row["hier_hybrid_a"]),
                "hier_hybrid_b": float(row["hier_hybrid_b"]),
                "struct_a": float(row["struct_a"]),
                "struct_b": float(row["struct_b"]),
            },
        }
        save_json(summary, case_dir / "summary.json")
        exported.append(
            {
                "title_tag": title_tag,
                "rank": rank,
                "image_id": row["image_id"],
                "preferred_model": row["preferred_model"],
                "other_model": row["other_model"],
                "case_dir": str(case_dir),
            }
        )
    return exported


class ProxyCache:
    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        hop_dist: np.ndarray,
        far_distance_threshold: float,
        depth_mappings: Dict[int, Dict[str, Any]],
        boundary_tolerance: int,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.hop_dist = hop_dist
        self.far_distance_threshold = far_distance_threshold
        self.depth_mappings = depth_mappings
        self.boundary_tolerance = boundary_tolerance
        self.semantic_cache: Dict[Tuple[str, str], SemanticProxy] = {}
        self.struct_cache: Dict[Tuple[str, str], StructuralProxy] = {}

    def get_semantic(self, pred_name: str, image_id: str, gt_path: Path, pred_path: Path) -> SemanticProxy:
        key = (pred_name, image_id)
        if key not in self.semantic_cache:
            gt = load_label_map(gt_path, offset=0, ignore_index=self.ignore_index)
            pred = load_label_map(pred_path, offset=0, ignore_index=self.ignore_index)
            self.semantic_cache[key] = compute_semantic_proxy(
                gt=gt,
                pred=pred,
                ignore_index=self.ignore_index,
                hop_dist=self.hop_dist,
                far_distance_threshold=self.far_distance_threshold,
                depth_mappings=self.depth_mappings,
            )
        return self.semantic_cache[key]

    def get_structural(self, pred_name: str, image_id: str, gt_path: Path, pred_path: Path) -> StructuralProxy:
        key = (pred_name, image_id)
        if key not in self.struct_cache:
            gt = load_label_map(gt_path, offset=0, ignore_index=self.ignore_index)
            pred = load_label_map(pred_path, offset=0, ignore_index=self.ignore_index)
            self.struct_cache[key] = compute_structural_proxy(
                gt=gt,
                pred=pred,
                ignore_index=self.ignore_index,
                num_classes=self.num_classes,
                boundary_tolerance=self.boundary_tolerance,
            )
        return self.struct_cache[key]


def evaluate_prediction_dir(
    gt_dir: Path,
    pred_dir: Path,
    pred_name: str,
    hybrid_matrix: np.ndarray,
    path_matrix: np.ndarray,
    wup_matrix: np.ndarray,
    num_classes: int,
    ignore_index: int,
    gt_offset: int,
    pred_offset: int,
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
    max_images: int,
    num_workers: int,
    no_progress: bool,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    pairs = match_gt_pred_files(gt_dir, pred_dir, warnings)
    if max_images > 0:
        pairs = pairs[:max_images]
    if not pairs:
        return []

    use_dense_confusions = False
    results: List[Dict[str, Any]] = []
    effective_workers = resolve_num_workers(num_workers)
    if effective_workers > len(pairs):
        effective_workers = len(pairs)
    if effective_workers < 1:
        effective_workers = 1

    def process_payload(pair: MatchedPair, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not payload.get("ok", True):
            warnings.append(
                f"{pred_name}:{pair.image_id}:{payload.get('error_type', 'Error')}:{payload.get('error_message', '')}"
            )
            return None
        confusion = sparse_to_dense_confusion(
            np.asarray(payload["conf_indices"], dtype=np.int64),
            np.asarray(payload["conf_counts"], dtype=np.int64),
            num_classes,
        )
        path_score = compute_hier_soft_iou_from_confusion(confusion, path_matrix)["hier_miou"]
        wup_score = compute_hier_soft_iou_from_confusion(confusion, wup_matrix)["hier_miou"]
        return {
            "pred_name": pred_name,
            "image_id": pair.image_id,
            "gt_path": str(pair.gt_path),
            "pred_path": str(pair.pred_path),
            "gt_rel": pair.gt_rel,
            "pred_rel": pair.pred_rel,
            "miou": float(payload["miou"]),
            "hier_hybrid": float(payload["hieriou"]),
            "hier_path": float(path_score),
            "hier_wup": float(wup_score),
            "struct": float(payload["struct"]),
            "persist": float(payload["persist"]),
            "edge": float(payload["edge"]),
            "num_valid_classes": int(payload["num_valid_classes"]),
            "conf_indices": np.asarray(payload["conf_indices"], dtype=np.int64),
            "conf_counts": np.asarray(payload["conf_counts"], dtype=np.int64),
        }

    if effective_workers == 1:
        iterator = tqdm(
            pairs,
            total=len(pairs),
            desc=f"{pred_name} images",
            unit="img",
            disable=no_progress,
            dynamic_ncols=True,
        )
        for pair in iterator:
            payload = evaluate_image_pair(
                pair=pair,
                similarity_matrix=hybrid_matrix,
                num_classes=num_classes,
                ignore_index=ignore_index,
                gt_offset=gt_offset,
                pred_offset=pred_offset,
                pred_collapse_ge_to=None,
                pred_collapse_target=None,
                max_radius=max_radius,
                boundary_width=boundary_width,
                kernel_type=kernel_type,
                use_dense_confusions=use_dense_confusions,
            )
            payload["ok"] = True
            row = process_payload(pair, payload)
            if row is not None:
                results.append(row)
        iterator.close()
    else:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=init_image_worker,
            initargs=(
                hybrid_matrix,
                num_classes,
                ignore_index,
                gt_offset,
                pred_offset,
                None,
                None,
                max_radius,
                boundary_width,
                kernel_type,
                use_dense_confusions,
            ),
        ) as executor:
            future_to_pair = {executor.submit(evaluate_image_pair_worker, pair): pair for pair in pairs}
            progress = tqdm(total=len(pairs), desc=f"{pred_name} images", unit="img", disable=no_progress, dynamic_ncols=True)
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                payload = future.result()
                row = process_payload(pair, payload)
                if row is not None:
                    results.append(row)
                progress.update(1)
            progress.close()
        results.sort(key=lambda row: row["image_id"])
    return results


def build_overall_summary(image_df: pd.DataFrame, metrics: Sequence[str], n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pred_name, group in image_df.groupby("pred_name", sort=True):
        row: Dict[str, Any] = {"pred_name": pred_name, "num_images": int(len(group))}
        for idx, metric in enumerate(metrics):
            stats = bootstrap_mean_ci(group[metric].to_numpy(dtype=np.float64), n_bootstrap=n_bootstrap, seed=seed + idx)
            row[metric] = stats["mean"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    overall_df = pd.DataFrame(rows)
    for metric in metrics:
        overall_df[f"rank_{metric}"] = overall_df[metric].rank(method="min", ascending=False).astype(int)
    return overall_df.sort_values("pred_name").reset_index(drop=True)


def build_pair_dataframe(image_df: pd.DataFrame) -> pd.DataFrame:
    pair_rows: List[pd.DataFrame] = []
    metric_cols = ["miou", "hier_hybrid", "hier_path", "hier_wup", "struct", "persist", "edge"]
    for model_a, model_b in combinations(sorted(image_df["pred_name"].unique().tolist()), 2):
        a_df = image_df.loc[image_df["pred_name"] == model_a, ["image_id", "gt_path", "pred_path"] + metric_cols].copy()
        b_df = image_df.loc[image_df["pred_name"] == model_b, ["image_id", "gt_path", "pred_path"] + metric_cols].copy()
        a_df = a_df.rename(columns={col: f"{col}_a" for col in a_df.columns if col != "image_id"})
        b_df = b_df.rename(columns={col: f"{col}_b" for col in b_df.columns if col != "image_id"})
        merged = a_df.merge(b_df, on="image_id", how="inner")
        if merged.empty:
            continue
        merged["model_a"] = model_a
        merged["model_b"] = model_b
        merged["gt_path"] = merged["gt_path_a"]
        for metric in metric_cols:
            merged[f"delta_{metric}"] = merged[f"{metric}_b"] - merged[f"{metric}_a"]
            merged[f"abs_delta_{metric}"] = merged[f"delta_{metric}"].abs()
        merged["max_abs_sem_gap"] = merged[["abs_delta_hier_hybrid", "abs_delta_hier_path", "abs_delta_hier_wup"]].max(axis=1)
        pair_rows.append(merged)
    if not pair_rows:
        return pd.DataFrame()
    return pd.concat(pair_rows, ignore_index=True)


def pick_preferred_side(row: pd.Series, delta_col: str) -> Dict[str, Any]:
    delta = float(row[delta_col])
    if delta >= 0.0:
        return {
            "preferred_model": row["model_b"],
            "other_model": row["model_a"],
            "preferred_pred_path": row["pred_path_b"],
            "other_pred_path": row["pred_path_a"],
            "preferred_suffix": "b",
            "other_suffix": "a",
            "preference_delta": delta,
        }
    return {
        "preferred_model": row["model_a"],
        "other_model": row["model_b"],
        "preferred_pred_path": row["pred_path_a"],
        "other_pred_path": row["pred_path_b"],
        "preferred_suffix": "a",
        "other_suffix": "b",
        "preference_delta": -delta,
    }


def build_semantic_pool(pair_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, float]]:
    close_miou_thr = float(pair_df["abs_delta_miou"].quantile(args.semantic_close_miou_quantile))
    close_struct_thr = float(pair_df["abs_delta_struct"].quantile(args.semantic_close_struct_quantile))
    sem_gap_thr = float(pair_df["max_abs_sem_gap"].quantile(args.semantic_gap_quantile))
    pool = pair_df.loc[
        (pair_df["abs_delta_miou"] <= close_miou_thr)
        & (pair_df["abs_delta_struct"] <= close_struct_thr)
        & (pair_df["max_abs_sem_gap"] >= sem_gap_thr)
    ].copy()
    thresholds = {
        "close_miou_thr": close_miou_thr,
        "close_struct_thr": close_struct_thr,
        "semantic_gap_thr": sem_gap_thr,
    }
    return pool.reset_index(drop=True), thresholds


def build_structural_pool(pair_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, float]]:
    close_miou_thr = float(pair_df["abs_delta_miou"].quantile(args.struct_close_miou_quantile))
    close_hybrid_thr = float(pair_df["abs_delta_hier_hybrid"].quantile(args.struct_close_sem_quantile))
    struct_gap_thr = float(pair_df["abs_delta_struct"].quantile(args.struct_gap_quantile))
    pool = pair_df.loc[
        (pair_df["abs_delta_miou"] <= close_miou_thr)
        & (pair_df["abs_delta_hier_hybrid"] <= close_hybrid_thr)
        & (pair_df["abs_delta_struct"] >= struct_gap_thr)
    ].copy()
    thresholds = {
        "close_miou_thr": close_miou_thr,
        "close_hybrid_thr": close_hybrid_thr,
        "struct_gap_thr": struct_gap_thr,
    }
    return pool.reset_index(drop=True), thresholds


def evaluate_semantic_alignment(
    semantic_pool: pd.DataFrame,
    proxy_cache: ProxyCache,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_map = {
        "Path": "delta_hier_path",
        "WuP": "delta_hier_wup",
        "Ours": "delta_hier_hybrid",
    }
    summary_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    if semantic_pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    for metric_name, delta_col in metric_map.items():
        pref_gap_thr = float(semantic_pool[delta_col].abs().quantile(args.semantic_metric_gap_quantile))
        eligible = semantic_pool.loc[semantic_pool[delta_col].abs() >= pref_gap_thr].copy()
        if eligible.empty:
            continue
        eligible = eligible.sort_values(delta_col, key=lambda s: s.abs(), ascending=False)
        eligible = eligible.drop_duplicates(subset=["image_id"], keep="first").reset_index(drop=True)

        coarse_wins: List[float] = []
        hop_wins: List[float] = []
        far_wins: List[float] = []
        coarse_margins: List[float] = []
        hop_margins: List[float] = []

        for _, row in eligible.iterrows():
            side = pick_preferred_side(row, delta_col=delta_col)
            pref_sem = proxy_cache.get_semantic(
                pred_name=str(side["preferred_model"]),
                image_id=str(row["image_id"]),
                gt_path=Path(row["gt_path"]),
                pred_path=Path(side["preferred_pred_path"]),
            )
            other_sem = proxy_cache.get_semantic(
                pred_name=str(side["other_model"]),
                image_id=str(row["image_id"]),
                gt_path=Path(row["gt_path"]),
                pred_path=Path(side["other_pred_path"]),
            )

            coarse_margin = pref_sem.coarse_auc - other_sem.coarse_auc
            hop_margin = other_sem.mean_hop - pref_sem.mean_hop
            far_margin = other_sem.far_error_ratio - pref_sem.far_error_ratio
            coarse_win = 0.5 if abs(coarse_margin) <= args.proxy_tie_epsilon else float(coarse_margin > 0.0)
            hop_win = 0.5 if abs(hop_margin) <= args.proxy_tie_epsilon else float(hop_margin > 0.0)
            far_win = 0.5 if abs(far_margin) <= args.proxy_tie_epsilon else float(far_margin > 0.0)

            coarse_wins.append(coarse_win)
            hop_wins.append(hop_win)
            far_wins.append(far_win)
            coarse_margins.append(coarse_margin)
            hop_margins.append(hop_margin)

            candidate_rows.append(
                {
                    "metric": metric_name,
                    "image_id": row["image_id"],
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "preferred_model": side["preferred_model"],
                    "other_model": side["other_model"],
                    "preferred_pred_path": side["preferred_pred_path"],
                    "other_pred_path": side["other_pred_path"],
                    "gt_path": row["gt_path"],
                    "miou_a": row["miou_a"],
                    "miou_b": row["miou_b"],
                    "hier_hybrid_a": row["hier_hybrid_a"],
                    "hier_hybrid_b": row["hier_hybrid_b"],
                    "struct_a": row["struct_a"],
                    "struct_b": row["struct_b"],
                    "metric_gap": side["preference_delta"],
                    "coarse_auc_margin": coarse_margin,
                    "hop_margin": hop_margin,
                    "far_error_margin": far_margin,
                    "coarse_auc_pref": pref_sem.coarse_auc,
                    "coarse_auc_other": other_sem.coarse_auc,
                    "mean_hop_pref": pref_sem.mean_hop,
                    "mean_hop_other": other_sem.mean_hop,
                    "far_error_pref": pref_sem.far_error_ratio,
                    "far_error_other": other_sem.far_error_ratio,
                }
            )

        for proxy_name, values, margins in [
            ("coarse_auc", coarse_wins, coarse_margins),
            ("mean_hop", hop_wins, hop_margins),
            ("far_error_ratio", far_wins, []),
        ]:
            rate_stats = bootstrap_rate_ci(values, n_bootstrap=args.bootstrap_iters, seed=args.seed)
            if proxy_name == "coarse_auc":
                tie_free = [v for v in values if v != 0.5]
            elif proxy_name == "mean_hop":
                tie_free = [v for v in values if v != 0.5]
            else:
                tie_free = [v for v in values if v != 0.5]

            successes = sum(1 for v in tie_free if v > 0.5)
            trials = len(tie_free)
            pvalue = float(binomtest(successes, trials, 0.5).pvalue) if trials > 0 else float("nan")
            summary_rows.append(
                {
                    "metric": metric_name,
                    "proxy": proxy_name,
                    "num_pairs": int(len(values)),
                    "num_tie_free_pairs": int(trials),
                    "preference_gap_threshold": pref_gap_thr,
                    "agreement_mean": rate_stats["mean"],
                    "agreement_ci_low": rate_stats["ci_low"],
                    "agreement_ci_high": rate_stats["ci_high"],
                    "agreement_std": rate_stats["std"],
                    "mean_proxy_margin": float(np.mean(margins)) if margins else float("nan"),
                    "binom_pvalue": pvalue,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(candidate_rows)


def evaluate_structural_alignment(
    structural_pool: pd.DataFrame,
    proxy_cache: ProxyCache,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    if structural_pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    struct_gap_thr = float(structural_pool["abs_delta_struct"].quantile(args.struct_metric_gap_quantile))
    eligible = structural_pool.loc[structural_pool["abs_delta_struct"] >= struct_gap_thr].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = eligible.sort_values("abs_delta_struct", ascending=False)
    eligible = eligible.drop_duplicates(subset=["image_id"], keep="first").reset_index(drop=True)

    boundary_wins: List[float] = []
    component_wins: List[float] = []
    classic_wins: List[float] = []
    boundary_margins: List[float] = []
    component_margins: List[float] = []
    classic_margins: List[float] = []

    for _, row in eligible.iterrows():
        side = pick_preferred_side(row, delta_col="delta_struct")
        pref_proxy = proxy_cache.get_structural(
            pred_name=str(side["preferred_model"]),
            image_id=str(row["image_id"]),
            gt_path=Path(row["gt_path"]),
            pred_path=Path(side["preferred_pred_path"]),
        )
        other_proxy = proxy_cache.get_structural(
            pred_name=str(side["other_model"]),
            image_id=str(row["image_id"]),
            gt_path=Path(row["gt_path"]),
            pred_path=Path(side["other_pred_path"]),
        )

        boundary_margin = pref_proxy.boundary_f1 - other_proxy.boundary_f1
        component_margin = pref_proxy.component_score - other_proxy.component_score
        classic_margin = pref_proxy.classic_struct - other_proxy.classic_struct

        boundary_win = 0.5 if abs(boundary_margin) <= args.proxy_tie_epsilon else float(boundary_margin > 0.0)
        component_win = 0.5 if abs(component_margin) <= args.proxy_tie_epsilon else float(component_margin > 0.0)
        classic_win = 0.5 if abs(classic_margin) <= args.proxy_tie_epsilon else float(classic_margin > 0.0)

        boundary_wins.append(boundary_win)
        component_wins.append(component_win)
        classic_wins.append(classic_win)
        boundary_margins.append(boundary_margin)
        component_margins.append(component_margin)
        classic_margins.append(classic_margin)

        candidate_rows.append(
            {
                "metric": "StructScore",
                "image_id": row["image_id"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "preferred_model": side["preferred_model"],
                "other_model": side["other_model"],
                "preferred_pred_path": side["preferred_pred_path"],
                "other_pred_path": side["other_pred_path"],
                "gt_path": row["gt_path"],
                "miou_a": row["miou_a"],
                "miou_b": row["miou_b"],
                "hier_hybrid_a": row["hier_hybrid_a"],
                "hier_hybrid_b": row["hier_hybrid_b"],
                "struct_a": row["struct_a"],
                "struct_b": row["struct_b"],
                "struct_gap": side["preference_delta"],
                "boundary_margin": boundary_margin,
                "component_margin": component_margin,
                "classic_margin": classic_margin,
                "boundary_pref": pref_proxy.boundary_f1,
                "boundary_other": other_proxy.boundary_f1,
                "component_pref": pref_proxy.component_score,
                "component_other": other_proxy.component_score,
                "classic_pref": pref_proxy.classic_struct,
                "classic_other": other_proxy.classic_struct,
            }
        )

    for proxy_name, values, margins in [
        ("boundary_f1", boundary_wins, boundary_margins),
        ("component_score", component_wins, component_margins),
        ("classic_struct", classic_wins, classic_margins),
    ]:
        rate_stats = bootstrap_rate_ci(values, n_bootstrap=args.bootstrap_iters, seed=args.seed)
        tie_free = [v for v in values if v != 0.5]
        successes = sum(1 for v in tie_free if v > 0.5)
        trials = len(tie_free)
        pvalue = float(binomtest(successes, trials, 0.5).pvalue) if trials > 0 else float("nan")
        summary_rows.append(
            {
                "metric": "StructScore",
                "proxy": proxy_name,
                "num_pairs": int(len(values)),
                "num_tie_free_pairs": int(trials),
                "preference_gap_threshold": struct_gap_thr,
                "agreement_mean": rate_stats["mean"],
                "agreement_ci_low": rate_stats["ci_low"],
                "agreement_ci_high": rate_stats["ci_high"],
                "agreement_std": rate_stats["std"],
                "mean_proxy_margin": float(np.mean(margins)),
                "binom_pvalue": pvalue,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(candidate_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Objective matched-pair validation on real predictions.")
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--gt_summary_dir", type=Path, default=ROOT / "data" / "gt_visual_summaries")
    parser.add_argument("--pred_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pred_names", nargs="*", default=None)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--hybrid_matrix", type=Path, required=True)
    parser.add_argument("--path_matrix", type=Path, required=True)
    parser.add_argument("--wup_matrix", type=Path, required=True)
    parser.add_argument("--class_path_mapping", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gt_offset", type=int, default=0)
    parser.add_argument("--pred_offset", type=int, default=0)
    parser.add_argument("--max_radius", type=int, default=3)
    parser.add_argument("--boundary_width", type=int, default=1)
    parser.add_argument("--kernel_type", choices=["disk", "square"], default="disk")
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--semantic_close_miou_quantile", type=float, default=0.25)
    parser.add_argument("--semantic_close_struct_quantile", type=float, default=0.30)
    parser.add_argument("--semantic_gap_quantile", type=float, default=0.80)
    parser.add_argument("--semantic_metric_gap_quantile", type=float, default=0.60)
    parser.add_argument("--struct_close_miou_quantile", type=float, default=0.25)
    parser.add_argument("--struct_close_sem_quantile", type=float, default=0.30)
    parser.add_argument("--struct_gap_quantile", type=float, default=0.80)
    parser.add_argument("--struct_metric_gap_quantile", type=float, default=0.60)
    parser.add_argument("--proxy_tie_epsilon", type=float, default=1e-4)
    parser.add_argument("--max_case_exports", type=int, default=6)
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    plots_dir = ensure_dir(output_dir / "plots")
    logs_dir = ensure_dir(output_dir / "logs")
    cases_dir = ensure_dir(output_dir / "cases")

    pred_names = list(args.pred_names) if args.pred_names else []
    while len(pred_names) < len(args.pred_dirs):
        pred_names.append(args.pred_dirs[len(pred_names)].name)
    if len(pred_names) != len(args.pred_dirs):
        raise ValueError("--pred_names must match --pred_dirs")

    warnings: List[str] = []
    hybrid_matrix = load_similarity_matrix(args.hybrid_matrix, args.num_classes, warnings)
    path_matrix = load_similarity_matrix(args.path_matrix, args.num_classes, warnings)
    wup_matrix = load_similarity_matrix(args.wup_matrix, args.num_classes, warnings)
    class_paths = load_class_paths(args.class_path_mapping)
    hop_dist = build_hop_distance_matrix(class_paths, args.num_classes)
    far_distance_threshold = compute_far_distance_threshold(hop_dist)
    depth_mappings = build_depth_mappings(class_paths, args.depths)

    print("=" * 80)
    print("Objective Matched-Pair Validation")
    print("=" * 80)
    print(f"predictions: {len(args.pred_dirs)}")
    print(f"depths: {args.depths}")
    print(f"far distance threshold: {far_distance_threshold:.4f}")

    all_rows: List[Dict[str, Any]] = []
    for pred_name, pred_dir in zip(pred_names, args.pred_dirs):
        print("-" * 80)
        print(f"evaluating {pred_name}")
        rows = evaluate_prediction_dir(
            gt_dir=args.gt_dir,
            pred_dir=pred_dir,
            pred_name=pred_name,
            hybrid_matrix=hybrid_matrix,
            path_matrix=path_matrix,
            wup_matrix=wup_matrix,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            gt_offset=args.gt_offset,
            pred_offset=args.pred_offset,
            max_radius=args.max_radius,
            boundary_width=args.boundary_width,
            kernel_type=args.kernel_type,
            max_images=args.max_images,
            num_workers=args.num_workers,
            no_progress=args.no_progress,
            warnings=warnings,
        )
        all_rows.extend(rows)
        print(f"{pred_name}: {len(rows)} images")

    if not all_rows:
        raise RuntimeError("No evaluation rows were produced.")

    image_df = pd.DataFrame(all_rows)
    image_df = image_df.sort_values(["pred_name", "image_id"]).reset_index(drop=True)
    image_out = image_df.drop(columns=["conf_indices", "conf_counts"]).copy()
    image_out.to_csv(output_dir / "image_level_metrics_multiscore.csv", index=False)

    overall_metrics = ["miou", "hier_path", "hier_wup", "hier_hybrid", "struct", "persist", "edge"]
    overall_df = build_overall_summary(image_df, overall_metrics, n_bootstrap=args.bootstrap_iters, seed=args.seed)
    overall_df.to_csv(output_dir / "overall_ranking_with_ci.csv", index=False)
    plot_metric_ranking(overall_df, overall_metrics[:5], plots_dir / "ranking")

    pair_df = build_pair_dataframe(image_df.drop(columns=["conf_indices", "conf_counts"]))
    pair_df.to_csv(output_dir / "pair_level_deltas.csv", index=False)
    plot_pair_gap_scatter(
        pair_df,
        xcol="delta_hier_hybrid",
        ycol="delta_struct",
        title="Pair-level hybrid semantic vs structural gap",
        out_path=plots_dir / "pair_scatter_hybrid_vs_struct.png",
    )

    semantic_pool, semantic_thresholds = build_semantic_pool(pair_df, args)
    structural_pool, structural_thresholds = build_structural_pool(pair_df, args)
    semantic_pool.to_csv(output_dir / "semantic_pool.csv", index=False)
    structural_pool.to_csv(output_dir / "structural_pool.csv", index=False)

    proxy_cache = ProxyCache(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        hop_dist=hop_dist,
        far_distance_threshold=far_distance_threshold,
        depth_mappings=depth_mappings,
        boundary_tolerance=args.boundary_tolerance,
    )

    semantic_summary_df, semantic_candidates_df = evaluate_semantic_alignment(semantic_pool, proxy_cache, args)
    structural_summary_df, structural_candidates_df = evaluate_structural_alignment(structural_pool, proxy_cache, args)

    semantic_summary_df.to_csv(output_dir / "semantic_alignment_summary.csv", index=False)
    structural_summary_df.to_csv(output_dir / "structural_alignment_summary.csv", index=False)
    semantic_candidates_df.to_csv(output_dir / "semantic_alignment_candidates.csv", index=False)
    structural_candidates_df.to_csv(output_dir / "structural_alignment_candidates.csv", index=False)

    plot_semantic_alignment(semantic_summary_df, "coarse_auc", plots_dir / "semantic_alignment_coarse_auc.png")
    plot_semantic_alignment(semantic_summary_df, "mean_hop", plots_dir / "semantic_alignment_mean_hop.png")
    plot_structural_alignment(structural_summary_df, plots_dir / "structural_alignment.png")

    case_rows = []
    if not semantic_candidates_df.empty:
        hybrid_cases = semantic_candidates_df.loc[semantic_candidates_df["metric"] == "Ours"].copy()
        case_rows.extend(
            export_pair_cases(
                candidates=hybrid_cases,
                out_dir=cases_dir / "semantic_ours",
                gt_summary_dir=args.gt_summary_dir,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                max_cases=args.max_case_exports,
                metric_col="metric_gap",
                title_tag="semantic_ours",
            )
        )
    if not structural_candidates_df.empty:
        case_rows.extend(
            export_pair_cases(
                candidates=structural_candidates_df,
                out_dir=cases_dir / "structural",
                gt_summary_dir=args.gt_summary_dir,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                max_cases=args.max_case_exports,
                metric_col="struct_gap",
                title_tag="structural",
            )
        )
    if case_rows:
        save_csv(case_rows, output_dir / "case_exports.csv", ["title_tag", "rank", "image_id", "preferred_model", "other_model", "case_dir"])

    summary = {
        "config": {
            "gt_dir": str(args.gt_dir),
            "gt_summary_dir": str(args.gt_summary_dir),
            "pred_dirs": [str(p) for p in args.pred_dirs],
            "pred_names": pred_names,
            "num_classes": args.num_classes,
            "ignore_index": args.ignore_index,
            "max_radius": args.max_radius,
            "boundary_width": args.boundary_width,
            "kernel_type": args.kernel_type,
            "depths": args.depths,
            "seed": args.seed,
            "bootstrap_iters": args.bootstrap_iters,
            "boundary_tolerance": args.boundary_tolerance,
            "semantic_thresholds": semantic_thresholds,
            "structural_thresholds": structural_thresholds,
        },
        "counts": {
            "num_image_rows": int(len(image_df)),
            "num_pair_rows": int(len(pair_df)),
            "num_semantic_pool": int(len(semantic_pool)),
            "num_structural_pool": int(len(structural_pool)),
            "num_semantic_candidates": int(len(semantic_candidates_df)),
            "num_structural_candidates": int(len(structural_candidates_df)),
        },
        "semantic_alignment": semantic_summary_df.to_dict(orient="records"),
        "structural_alignment": structural_summary_df.to_dict(orient="records"),
        "warnings": warnings,
    }
    save_json(summary, output_dir / "summary.json")
    save_json({"warnings": warnings}, logs_dir / "warnings.json")

    md_lines = [
        "# Objective Matched-Pair Validation",
        "",
        "## Counts",
        f"- image rows: {len(image_df)}",
        f"- pair rows: {len(pair_df)}",
        f"- semantic pool: {len(semantic_pool)}",
        f"- structural pool: {len(structural_pool)}",
        "",
        "## Semantic Thresholds",
        f"- close mIoU threshold: {semantic_thresholds['close_miou_thr']:.6f}",
        f"- close StructScore threshold: {semantic_thresholds['close_struct_thr']:.6f}",
        f"- semantic gap threshold: {semantic_thresholds['semantic_gap_thr']:.6f}",
        "",
        "## Structural Thresholds",
        f"- close mIoU threshold: {structural_thresholds['close_miou_thr']:.6f}",
        f"- close hybrid semantic threshold: {structural_thresholds['close_hybrid_thr']:.6f}",
        f"- structural gap threshold: {structural_thresholds['struct_gap_thr']:.6f}",
        "",
        "## Top Semantic Alignment Rows",
    ]
    if not semantic_summary_df.empty:
        for _, row in semantic_summary_df.sort_values(["proxy", "agreement_mean"], ascending=[True, False]).iterrows():
            md_lines.append(
                f"- {row['metric']} / {row['proxy']}: agreement={row['agreement_mean']:.4f} "
                f"[{row['agreement_ci_low']:.4f}, {row['agreement_ci_high']:.4f}], "
                f"n={int(row['num_pairs'])}, p={row['binom_pvalue']:.4g}"
            )
    else:
        md_lines.append("- no semantic alignment rows")

    md_lines.extend(["", "## Top Structural Alignment Rows"])
    if not structural_summary_df.empty:
        for _, row in structural_summary_df.sort_values("agreement_mean", ascending=False).iterrows():
            md_lines.append(
                f"- {row['proxy']}: agreement={row['agreement_mean']:.4f} "
                f"[{row['agreement_ci_low']:.4f}, {row['agreement_ci_high']:.4f}], "
                f"n={int(row['num_pairs'])}, p={row['binom_pvalue']:.4g}"
            )
    else:
        md_lines.append("- no structural alignment rows")

    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
