import argparse
import csv
import itertools
import json
import math
import os
import tempfile
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))

from calibrate_csp_params import build_perturbation_dataset, evaluate_parameter_pair
from csp_perturbation_utils import build_perturbation_specs, parse_name_list
from structural_metrics import (
    _aggregate_struct_counts,
    _aggregate_struct_dataset_results,
    _evaluate_pair_from_paths,
    _match_pred_gt_files,
    evaluate_dataset_from_dirs,
    miou_from_confusion,
)


def parse_int_list(text: str) -> List[int]:
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


def _safe_name(path: str) -> str:
    return Path(path).name.replace(" ", "_")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _center_distance(radius: float, boundary_width: float, center_radius: float, center_bw: float) -> float:
    return abs(radius - center_radius) + abs(boundary_width - center_bw)


def _rank_methods(scores: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {name: idx + 1 for idx, (name, _) in enumerate(ordered)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Select a non-degenerate StructScore operating point with controlled calibration and real-prediction sanity checks"
    )
    parser.add_argument("--gt-dir", type=str, required=True, help="GT label directory for calibration and real sweeps")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignored label value")
    parser.add_argument(
        "--radii",
        type=parse_int_list,
        default=[1, 2, 3, 4, 5],
        help="Non-degenerate candidate radii, e.g. 1,2,3,4,5",
    )
    parser.add_argument(
        "--degenerate-radii",
        type=parse_int_list,
        default=[0],
        help="Degenerate ablation radii kept for reference but excluded from final selection",
    )
    parser.add_argument(
        "--boundary-widths",
        type=parse_int_list,
        default=[1, 2, 3],
        help="Candidate boundary widths, e.g. 1,2,3",
    )
    parser.add_argument("--kernel-type", type=str, default="disk", choices=["disk", "square"], help="Morphology kernel")
    parser.add_argument("--perturbations", type=parse_name_list, default=None, help="Optional comma-separated perturbation names")
    parser.add_argument("--pred-dirs", nargs="*", default=None, help="Optional real-prediction directories for sanity checks")
    parser.add_argument("--pred-names", nargs="*", default=None, help="Optional names for --pred-dirs")
    parser.add_argument("--max-images", type=int, default=200, help="Calibration max_images")
    parser.add_argument(
        "--real-max-images",
        type=int,
        default=None,
        help="Optional limit for real-prediction sanity pairs; defaults to full matched set",
    )
    parser.add_argument("--max-instances", type=int, default=300, help="Calibration max_instances")
    parser.add_argument("--max-classes-per-image", type=int, default=2, help="Calibration max_classes_per_image")
    parser.add_argument("--min-mask-pixels", type=int, default=128, help="Calibration min_mask_pixels")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8, help="Worker count for real-prediction evaluation")
    parser.add_argument("--preview-pairs", type=int, default=0, help="Preview matched prediction/GT pairs")
    parser.add_argument(
        "--plateau-ratio",
        type=float,
        default=0.98,
        help="Pairs with final_score >= plateau_ratio * best_score form the high-score plateau",
    )
    parser.add_argument(
        "--controlled-weight",
        type=float,
        default=0.75,
        help="Weight for controlled calibration score in final selection",
    )
    parser.add_argument(
        "--real-weight",
        type=float,
        default=0.25,
        help="Weight for real-prediction stability score in final selection",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for the full selection run")
    parser.add_argument("--experiment-name", type=str, default="csp_nondegenerate_selection", help="Run name suffix")
    parser.add_argument("--notes", type=str, default="", help="Optional notes recorded with the run")
    return parser


def _validate_pred_names(pred_dirs: Sequence[str], pred_names: Optional[Sequence[str]]) -> List[str]:
    if not pred_dirs:
        return []
    names = list(pred_names) if pred_names else []
    while len(names) < len(pred_dirs):
        names.append(Path(pred_dirs[len(names)]).name)
    if len(names) != len(pred_dirs):
        raise ValueError("pred_names must have the same length as pred_dirs")
    return names


def _run_controlled_calibration(args: argparse.Namespace, pairs: Sequence[Tuple[int, int]], controlled_dir: Path) -> pd.DataFrame:
    specs = build_perturbation_specs(args.perturbations)
    samples = build_perturbation_dataset(
        argparse.Namespace(
            gt_dir=args.gt_dir,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            max_images=args.max_images,
            max_instances=args.max_instances,
            max_classes_per_image=args.max_classes_per_image,
            min_mask_pixels=args.min_mask_pixels,
            seed=args.seed,
            kernel_type=args.kernel_type,
        ),
        specs,
    )
    if not samples:
        raise RuntimeError("No calibration samples were generated. Try increasing max_images/max_instances.")

    rows: List[Dict[str, Any]] = []
    detail_dir = controlled_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for radius, boundary_width in pairs:
        print(f"[Select] controlled calibration: radius={radius}, boundary_width={boundary_width}")
        start = time.perf_counter()
        summary, evaluated = evaluate_parameter_pair(
            samples=samples,
            radius=radius,
            boundary_width=boundary_width,
            kernel_type=args.kernel_type,
        )
        elapsed = time.perf_counter() - start
        summary["elapsed_sec"] = float(elapsed)
        summary["degenerate_radius"] = bool(radius == 0)
        rows.append(summary)

        detail_path = detail_dir / f"radius_{radius:03d}_bw_{boundary_width:03d}.json"
        _save_json(
            detail_path,
            {
                "summary": summary,
                "config": {
                    "gt_dir": str(Path(args.gt_dir).resolve()),
                    "num_classes": args.num_classes,
                    "ignore_index": args.ignore_index,
                    "kernel_type": args.kernel_type,
                    "perturbations": specs,
                    "max_images": args.max_images,
                    "max_instances": args.max_instances,
                    "max_classes_per_image": args.max_classes_per_image,
                    "min_mask_pixels": args.min_mask_pixels,
                    "seed": args.seed,
                },
                "samples": evaluated,
            },
        )

    rows = sorted(rows, key=lambda row: (-row["calibration_score"], row["radius"], row["boundary_width"]))
    _write_csv(controlled_dir / "summary.csv", rows)
    _save_json(
        controlled_dir / "summary.json",
        {
            "config": {
                "gt_dir": str(Path(args.gt_dir).resolve()),
                "num_classes": args.num_classes,
                "ignore_index": args.ignore_index,
                "kernel_type": args.kernel_type,
                "radii": sorted({int(r) for r, _ in pairs}),
                "boundary_widths": sorted({int(bw) for _, bw in pairs}),
                "perturbations": specs,
                "max_images": args.max_images,
                "max_instances": args.max_instances,
                "max_classes_per_image": args.max_classes_per_image,
                "min_mask_pixels": args.min_mask_pixels,
                "seed": args.seed,
            },
            "num_samples": len(samples),
            "summary_rows": rows,
            "top5": rows[:5],
        },
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["radius"] = df["radius"].astype(int)
        df["boundary_width"] = df["boundary_width"].astype(int)
        df["degenerate_radius"] = df["degenerate_radius"].astype(bool)
    return df


def _run_real_sanity(
    args: argparse.Namespace,
    pairs: Sequence[Tuple[int, int]],
    pred_dirs: Sequence[str],
    pred_names: Sequence[str],
    real_dir: Path,
) -> pd.DataFrame:
    if not pred_dirs:
        empty = pd.DataFrame(
            columns=[
                "pred_name",
                "pred_dir",
                "radius",
                "boundary_width",
                "degenerate_radius",
                "num_images",
                "csp_image_mean",
                "persist_image_mean",
                "edge_image_mean",
                "csp_class_mean",
                "persist_class_mean",
                "edge_class_mean",
                "csp_miou_style",
                "persist_miou_style",
                "edge_miou_style",
                "miou",
                "elapsed_sec",
            ]
        )
        _write_csv(real_dir / "per_prediction_scores.csv", empty.to_dict("records"))
        return empty

    rows: List[Dict[str, Any]] = []
    for pred_name, pred_dir in zip(pred_names, pred_dirs):
        for radius, boundary_width in pairs:
            print(f"[Select] real sanity: {pred_name} | radius={radius}, boundary_width={boundary_width}")
            start = time.perf_counter()
            if args.real_max_images is None:
                metrics = evaluate_dataset_from_dirs(
                    pred_dir=pred_dir,
                    gt_dir=args.gt_dir,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                    max_radius=radius,
                    boundary_width=boundary_width,
                    kernel_type=args.kernel_type,
                    preview_pairs=args.preview_pairs,
                    num_workers=args.num_workers,
                )
            else:
                metrics = _evaluate_dataset_subset_from_dirs(
                    pred_dir=pred_dir,
                    gt_dir=args.gt_dir,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                    max_radius=radius,
                    boundary_width=boundary_width,
                    kernel_type=args.kernel_type,
                    preview_pairs=args.preview_pairs,
                    num_workers=args.num_workers,
                    max_pairs=args.real_max_images,
                )
            elapsed = time.perf_counter() - start
            row = {
                "pred_name": pred_name,
                "pred_dir": str(Path(pred_dir).resolve()),
                "radius": radius,
                "boundary_width": boundary_width,
                "degenerate_radius": bool(radius == 0),
                "num_images": int(metrics["num_images"]),
                "csp_image_mean": float(metrics["csp_image_mean"]),
                "persist_image_mean": float(metrics["persist_image_mean"]),
                "edge_image_mean": float(metrics["edge_image_mean"]),
                "csp_class_mean": float(metrics["csp_class_mean"]),
                "persist_class_mean": float(metrics["persist_class_mean"]),
                "edge_class_mean": float(metrics["edge_class_mean"]),
                "csp_miou_style": float(metrics["csp_miou_style"]),
                "persist_miou_style": float(metrics["persist_miou_style"]),
                "edge_miou_style": float(metrics["edge_miou_style"]),
                "miou": float(metrics["miou"]),
                "elapsed_sec": float(elapsed),
            }
            rows.append(row)

    _write_csv(real_dir / "per_prediction_scores.csv", rows)
    _save_json(
        real_dir / "summary.json",
        {
            "config": {
                "gt_dir": str(Path(args.gt_dir).resolve()),
                "pred_dirs": [str(Path(p).resolve()) for p in pred_dirs],
                "pred_names": list(pred_names),
                "num_classes": args.num_classes,
                "ignore_index": args.ignore_index,
                "kernel_type": args.kernel_type,
                "radii": sorted({int(r) for r, _ in pairs}),
                "boundary_widths": sorted({int(bw) for _, bw in pairs}),
                "num_workers": args.num_workers,
                "preview_pairs": args.preview_pairs,
                "real_max_images": args.real_max_images,
            },
            "rows": rows,
        },
    )
    return pd.DataFrame(rows)


def _evaluate_dataset_subset_from_dirs(
    pred_dir: str,
    gt_dir: str,
    num_classes: int,
    ignore_index: int,
    max_radius: int,
    boundary_width: int,
    kernel_type: str,
    preview_pairs: int,
    num_workers: int,
    max_pairs: Optional[int],
) -> Dict[str, Union[float, int, np.ndarray]]:
    pairs = _match_pred_gt_files(pred_dir, gt_dir, preview_pairs=preview_pairs)
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("real_max_images must be > 0 when provided")
        pairs = pairs[:max_pairs]

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


def _compute_real_pair_summary(real_df: pd.DataFrame, degenerate_radii: Sequence[int]) -> pd.DataFrame:
    if real_df.empty:
        return pd.DataFrame(
            columns=[
                "radius",
                "boundary_width",
                "degenerate_radius",
                "real_mean_struct",
                "real_std_struct",
                "real_range_struct",
                "real_rank_stability",
                "real_rank_corr_to_consensus",
            ]
        )

    pred_names = sorted(real_df["pred_name"].unique().tolist())
    n_methods = len(pred_names)
    nondeg_df = real_df[~real_df["radius"].isin(degenerate_radii)].copy()
    if nondeg_df.empty:
        nondeg_df = real_df.copy()

    per_pair_ranks: Dict[Tuple[int, int], Dict[str, int]] = {}
    for (radius, boundary_width), group in real_df.groupby(["radius", "boundary_width"], sort=False):
        scores = {row["pred_name"]: float(row["csp_miou_style"]) for _, row in group.iterrows()}
        per_pair_ranks[(int(radius), int(boundary_width))] = _rank_methods(scores)

    consensus_scores: Dict[str, float] = {}
    for pred_name in pred_names:
        values = []
        for (radius, boundary_width), ranks in per_pair_ranks.items():
            if radius in degenerate_radii:
                continue
            values.append(ranks[pred_name])
        if not values:
            values = [ranks[pred_name] for ranks in per_pair_ranks.values()]
        consensus_scores[pred_name] = float(np.mean(values))

    consensus_order = sorted(consensus_scores.items(), key=lambda item: (item[1], item[0]))
    consensus_rank = {name: idx + 1 for idx, (name, _) in enumerate(consensus_order)}
    consensus_vec = np.asarray([consensus_rank[name] for name in pred_names], dtype=np.float64)

    rows: List[Dict[str, Any]] = []
    denom = max(1, n_methods - 1)
    for (radius, boundary_width), group in real_df.groupby(["radius", "boundary_width"], sort=False):
        current_ranks = per_pair_ranks[(int(radius), int(boundary_width))]
        rank_vec = np.asarray([current_ranks[name] for name in pred_names], dtype=np.float64)
        mean_abs_rank_gap = float(np.mean(np.abs(rank_vec - consensus_vec)))
        stability = float(np.clip(1.0 - (mean_abs_rank_gap / denom), 0.0, 1.0))
        if np.std(rank_vec) < 1e-12 or np.std(consensus_vec) < 1e-12:
            corr = 1.0
        else:
            corr = float(np.corrcoef(rank_vec, consensus_vec)[0, 1])
            corr = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

        struct_scores = group["csp_miou_style"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "radius": int(radius),
                "boundary_width": int(boundary_width),
                "degenerate_radius": bool(int(radius) in set(int(v) for v in degenerate_radii)),
                "real_mean_struct": float(np.mean(struct_scores)),
                "real_std_struct": float(np.std(struct_scores)),
                "real_range_struct": float(np.max(struct_scores) - np.min(struct_scores)),
                "real_rank_stability": stability,
                "real_rank_corr_to_consensus": corr,
            }
        )
    return pd.DataFrame(rows)


def _merge_selection(
    controlled_df: pd.DataFrame,
    real_pair_df: pd.DataFrame,
    plateau_ratio: float,
    controlled_weight: float,
    real_weight: float,
    degenerate_radii: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    merged = controlled_df.merge(
        real_pair_df,
        on=["radius", "boundary_width", "degenerate_radius"],
        how="left",
    )
    if merged["real_rank_stability"].notna().any():
        merged["real_score"] = 0.5 * merged["real_rank_stability"].fillna(0.0) + 0.5 * merged[
            "real_rank_corr_to_consensus"
        ].fillna(0.0)
        total_weight = controlled_weight + real_weight
        merged["final_score"] = (
            controlled_weight * merged["calibration_score"] + real_weight * merged["real_score"]
        ) / total_weight
    else:
        merged["real_score"] = np.nan
        merged["final_score"] = merged["calibration_score"]

    merged["eligible_final"] = ~merged["radius"].isin([int(v) for v in degenerate_radii])
    candidate_df = merged[merged["eligible_final"]].copy()
    if candidate_df.empty:
        raise RuntimeError("No non-degenerate candidates remain after filtering. Check radii / degenerate-radii.")

    best_score = float(candidate_df["final_score"].max())
    threshold = best_score * plateau_ratio
    candidate_df["in_plateau"] = candidate_df["final_score"] >= threshold
    plateau_df = candidate_df[candidate_df["in_plateau"]].copy()
    center_radius = float(plateau_df["radius"].median())
    center_bw = float(plateau_df["boundary_width"].median())
    plateau_df["plateau_center_distance"] = plateau_df.apply(
        lambda row: _center_distance(float(row["radius"]), float(row["boundary_width"]), center_radius, center_bw),
        axis=1,
    )
    plateau_df = plateau_df.sort_values(
        by=["plateau_center_distance", "final_score", "radius", "boundary_width"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    chosen = plateau_df.iloc[0].to_dict()

    merged["in_plateau"] = False
    plateau_keys = {(int(row["radius"]), int(row["boundary_width"])) for _, row in plateau_df.iterrows()}
    merged["in_plateau"] = merged.apply(
        lambda row: (bool(row["eligible_final"]) and (int(row["radius"]), int(row["boundary_width"])) in plateau_keys),
        axis=1,
    )
    merged["is_selected"] = merged.apply(
        lambda row: int(row["radius"]) == int(chosen["radius"]) and int(row["boundary_width"]) == int(chosen["boundary_width"]),
        axis=1,
    )

    summary = {
        "best_score": best_score,
        "plateau_ratio": plateau_ratio,
        "plateau_threshold": threshold,
        "plateau_center": {"radius": center_radius, "boundary_width": center_bw},
        "selected_pair": {
            "radius": int(chosen["radius"]),
            "boundary_width": int(chosen["boundary_width"]),
            "calibration_score": float(chosen["calibration_score"]),
            "real_score": float(chosen["real_score"]) if not pd.isna(chosen["real_score"]) else None,
            "final_score": float(chosen["final_score"]),
        },
    }
    return merged, candidate_df, plateau_df, summary


def _plot_heatmap(df: pd.DataFrame, value_col: str, title: str, output_path: Path) -> None:
    if df.empty or value_col not in df.columns:
        return
    pivot = df.pivot(index="radius", columns="boundary_width", values=value_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Boundary Width")
    ax.set_ylabel("Radius")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(v)) for v in pivot.index])
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            value = pivot.iloc[y, x]
            if pd.isna(value):
                continue
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, summary: Dict[str, Any], plateau_df: pd.DataFrame) -> None:
    selected = summary["selected_pair"]
    lines = [
        "# Non-Degenerate StructScore Parameter Selection",
        "",
        "## Selection Rule",
        "- Degenerate radii such as `r=0` are retained as ablations but excluded from the final default-choice ranking.",
        "- Final score is computed from the controlled calibration score and the real-prediction rank-stability score.",
        f"- High-score plateau threshold: final_score >= {summary['plateau_ratio']:.3f} * best_score = {summary['plateau_threshold']:.6f}.",
        f"- Plateau center: radius={summary['plateau_center']['radius']:.3f}, boundary_width={summary['plateau_center']['boundary_width']:.3f}.",
        "",
        "## Selected Pair",
        f"- radius = {selected['radius']}",
        f"- boundary_width = {selected['boundary_width']}",
        f"- calibration_score = {selected['calibration_score']:.6f}",
        f"- real_score = {selected['real_score'] if selected['real_score'] is not None else 'N/A'}",
        f"- final_score = {selected['final_score']:.6f}",
        "",
        "## Plateau Members",
    ]
    for _, row in plateau_df.iterrows():
        lines.append(
            f"- (r={int(row['radius'])}, bw={int(row['boundary_width'])}) | "
            f"final_score={row['final_score']:.6f} | calibration={row['calibration_score']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if args.controlled_weight < 0 or args.real_weight < 0:
        raise ValueError("controlled_weight and real_weight must be >= 0")
    if (args.controlled_weight + args.real_weight) <= 0:
        raise ValueError("controlled_weight + real_weight must be > 0")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or "csp_nondegenerate_selection"
    run_dir = Path(args.output_dir) / f"{timestamp}_{experiment_name}"
    controlled_dir = run_dir / "controlled"
    real_dir = run_dir / "real"
    analysis_dir = run_dir / "analysis"
    controlled_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pred_dirs = list(args.pred_dirs) if args.pred_dirs else []
    pred_names = _validate_pred_names(pred_dirs, args.pred_names)

    nondeg_radii = sorted({int(v) for v in args.radii})
    degenerate_radii = sorted({int(v) for v in args.degenerate_radii})
    all_radii = sorted(set(nondeg_radii) | set(degenerate_radii))
    boundary_widths = sorted({int(v) for v in args.boundary_widths})
    pairs = list(itertools.product(all_radii, boundary_widths))

    config = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "gt_dir": str(Path(args.gt_dir).resolve()),
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "kernel_type": args.kernel_type,
        "nondegenerate_radii": nondeg_radii,
        "degenerate_radii": degenerate_radii,
        "all_radii": all_radii,
        "boundary_widths": boundary_widths,
        "pred_dirs": [str(Path(p).resolve()) for p in pred_dirs],
        "pred_names": pred_names,
        "max_images": args.max_images,
        "real_max_images": args.real_max_images,
        "max_instances": args.max_instances,
        "max_classes_per_image": args.max_classes_per_image,
        "min_mask_pixels": args.min_mask_pixels,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "preview_pairs": args.preview_pairs,
        "plateau_ratio": args.plateau_ratio,
        "controlled_weight": args.controlled_weight,
        "real_weight": args.real_weight,
        "notes": args.notes,
    }
    _save_json(run_dir / "config.json", config)

    print(f"[Select] run directory: {run_dir}")
    print(f"[Select] candidate pairs: {pairs}")
    if degenerate_radii:
        print(f"[Select] degenerate radii retained only for ablation: {degenerate_radii}")

    controlled_df = _run_controlled_calibration(args, pairs, controlled_dir)
    real_df = _run_real_sanity(args, pairs, pred_dirs, pred_names, real_dir)
    real_pair_df = _compute_real_pair_summary(real_df, degenerate_radii)
    merged_df, candidate_df, plateau_df, summary = _merge_selection(
        controlled_df=controlled_df,
        real_pair_df=real_pair_df,
        plateau_ratio=args.plateau_ratio,
        controlled_weight=args.controlled_weight,
        real_weight=args.real_weight,
        degenerate_radii=degenerate_radii,
    )

    merged_rows = merged_df.sort_values(
        by=["eligible_final", "final_score", "calibration_score", "radius", "boundary_width"],
        ascending=[False, False, False, True, True],
    ).to_dict("records")
    plateau_rows = plateau_df.to_dict("records")
    degenerate_rows = merged_df[merged_df["degenerate_radius"]].to_dict("records")

    _write_csv(analysis_dir / "selection_summary.csv", merged_rows)
    _write_csv(analysis_dir / "plateau_pairs.csv", plateau_rows)
    _write_csv(analysis_dir / "degenerate_ablation.csv", degenerate_rows)
    _save_json(
        analysis_dir / "final_choice.json",
        {
            "config": config,
            "summary": summary,
            "selected_row": next(row for row in merged_rows if row["is_selected"]),
        },
    )

    _plot_heatmap(candidate_df, "final_score", "Final Score (Non-Degenerate)", analysis_dir / "heatmap_final_score.png")
    _plot_heatmap(candidate_df, "calibration_score", "Controlled Calibration Score", analysis_dir / "heatmap_calibration_score.png")
    if candidate_df["real_score"].notna().any():
        _plot_heatmap(candidate_df, "real_score", "Real-Prediction Stability Score", analysis_dir / "heatmap_real_score.png")

    _write_report(analysis_dir / "report.md", summary, plateau_df)

    print("[Select] selected pair:")
    print(
        f"  radius={summary['selected_pair']['radius']}, "
        f"boundary_width={summary['selected_pair']['boundary_width']}, "
        f"final_score={summary['selected_pair']['final_score']:.6f}"
    )
    print(f"[Select] analysis summary: {analysis_dir / 'selection_summary.csv'}")
    print(f"[Select] final choice json: {analysis_dir / 'final_choice.json'}")


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
