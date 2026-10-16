#!/usr/bin/env python3
"""
Differential diagnosis analysis for semantic-vs-structure behavior.

This script compares two already-evaluated prediction directories, focusing on
where the structurally stronger model improves structure without necessarily
improving semantic metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
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

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "metrics"))

from evaluate_predictions import (  # type: ignore  # noqa: E402
    colorize_label_map,
    ensure_dir,
    load_label_map,
    match_gt_pred_files,
)
from io_utils import load_class_names  # type: ignore  # noqa: E402


IMAGE_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "pred_name": ("pred_name", "pred", "prediction", "model", "method"),
    "image_id": ("image_id", "img_id", "image", "filename", "file_name", "name", "id"),
    "miou": ("miou", "m_iou", "image_miou", "image_iou"),
    "hieriou": ("hieriou", "hier_iou", "hieriou_mean", "hier_miou", "hier_iou_mean"),
    "struct": ("struct", "structscore", "struct_score", "image_score"),
    "persist": ("persist", "persistence", "image_persist_mean"),
    "edge": ("edge", "image_edge_mean", "boundary_edge", "edge_score"),
}

CLASS_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "pred_name": ("pred_name", "pred", "prediction", "model", "method"),
    "class_id": ("class_id", "class", "label_id", "category_id"),
    "iou": ("iou", "class_iou", "miou"),
    "hieriou": ("hieriou", "hier_iou", "hier_miou", "class_hieriou"),
    "persist": ("persist", "persistence", "class_persist", "persist_mean"),
    "edge": ("edge", "edge_score", "class_edge", "edge_mean"),
    "struct": ("struct", "structscore", "struct_score", "class_struct"),
}

OVERALL_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "pred_name": ("pred_name", "pred", "prediction", "model", "method"),
    "miou_mean": ("miou_mean", "miou", "m_iou", "overall_miou"),
    "hieriou_mean": ("hieriou_mean", "hieriou", "hier_iou", "hier_miou", "overall_hieriou"),
    "struct_mean": ("struct_mean", "struct", "structscore", "struct_score", "overall_struct"),
}

DEFAULT_CLASS_JSON_CANDIDATES = (
    ROOT / "configs" / "ade150_classes.json",
    ROOT / "configs" / "class_names.json",
)


def canonicalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def safe_slug(text: str) -> str:
    slug = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            slug.append(ch)
        else:
            slug.append("_")
    return "".join(slug).strip("._") or "item"


def is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def normalize_column_names(
    df: pd.DataFrame,
    alias_map: Dict[str, Sequence[str]],
    required: Sequence[str],
    table_name: str,
) -> pd.DataFrame:
    if df.empty and required:
        raise ValueError(f"{table_name} is empty")

    canonical_lookup = {canonicalize_column_name(col): col for col in df.columns}
    rename_map: Dict[str, str] = {}
    used_originals = set()

    for canonical_name, aliases in alias_map.items():
        chosen_original: Optional[str] = None
        for alias in aliases:
            alias_key = canonicalize_column_name(alias)
            if alias_key in canonical_lookup:
                candidate = canonical_lookup[alias_key]
                if candidate not in used_originals:
                    chosen_original = candidate
                    break
        if chosen_original is not None:
            rename_map[chosen_original] = canonical_name
            used_originals.add(chosen_original)

    normalized = df.rename(columns=rename_map).copy()
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return normalized


def coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_class_names_optional(path: Optional[Path]) -> Optional[List[str]]:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"class names file not found: {path}")
        return load_class_names(path)

    for candidate in DEFAULT_CLASS_JSON_CANDIDATES:
        if candidate.exists():
            return load_class_names(candidate)
    return None


def load_metrics_tables(
    image_metrics: Path,
    class_metrics: Path,
    overall_metrics: Optional[Path],
) -> Dict[str, Optional[pd.DataFrame]]:
    image_df = normalize_column_names(
        load_csv_table(image_metrics),
        IMAGE_COLUMN_ALIASES,
        required=("pred_name", "image_id", "miou", "hieriou", "struct"),
        table_name="image_level_metrics",
    )
    image_df = coerce_numeric_columns(image_df, ("miou", "hieriou", "struct", "persist", "edge"))
    image_df["pred_name"] = image_df["pred_name"].astype(str)
    image_df["image_id"] = image_df["image_id"].astype(str)

    class_df = normalize_column_names(
        load_csv_table(class_metrics),
        CLASS_COLUMN_ALIASES,
        required=("pred_name", "class_id", "iou", "hieriou", "persist", "edge", "struct"),
        table_name="per_class_metrics",
    )
    class_df = coerce_numeric_columns(class_df, ("class_id", "iou", "hieriou", "persist", "edge", "struct"))
    class_df["pred_name"] = class_df["pred_name"].astype(str)
    class_df["class_id"] = class_df["class_id"].astype("Int64")

    overall_df: Optional[pd.DataFrame] = None
    if overall_metrics is not None:
        overall_df = normalize_column_names(
            load_csv_table(overall_metrics),
            OVERALL_COLUMN_ALIASES,
            required=("pred_name", "miou_mean", "hieriou_mean", "struct_mean"),
            table_name="overall_metrics",
        )
        overall_df = coerce_numeric_columns(overall_df, ("miou_mean", "hieriou_mean", "struct_mean"))
        overall_df["pred_name"] = overall_df["pred_name"].astype(str)

    return {
        "image": image_df,
        "class": class_df,
        "overall": overall_df,
    }


def ensure_predictions_present(df: pd.DataFrame, pred_a: str, pred_b: str, table_name: str) -> None:
    available = sorted(set(df["pred_name"].astype(str)))
    missing = [name for name in (pred_a, pred_b) if name not in available]
    if missing:
        raise ValueError(f"{table_name} is missing predictions {missing}. Available: {available}")


def build_image_diff_table(image_df: pd.DataFrame, pred_a: str, pred_b: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ensure_predictions_present(image_df, pred_a, pred_b, "image_level_metrics")
    cols = [col for col in ("image_id", "miou", "hieriou", "struct", "persist", "edge") if col in image_df.columns]

    a_df = image_df.loc[image_df["pred_name"] == pred_a, cols].copy()
    b_df = image_df.loc[image_df["pred_name"] == pred_b, cols].copy()

    a_df = a_df.rename(columns={col: f"{col}_crf" for col in cols if col != "image_id"})
    b_df = b_df.rename(columns={col: f"{col}_best" for col in cols if col != "image_id"})

    merged = a_df.merge(b_df, on="image_id", how="outer", indicator=True)
    matched = merged.loc[merged["_merge"] == "both"].copy()

    matched["delta_miou"] = matched["miou_best"] - matched["miou_crf"]
    matched["delta_hieriou"] = matched["hieriou_best"] - matched["hieriou_crf"]
    matched["delta_struct"] = matched["struct_best"] - matched["struct_crf"]
    if "persist_crf" in matched.columns and "persist_best" in matched.columns:
        matched["delta_persist"] = matched["persist_best"] - matched["persist_crf"]
    if "edge_crf" in matched.columns and "edge_best" in matched.columns:
        matched["delta_edge"] = matched["edge_best"] - matched["edge_crf"]
    matched["abs_delta_hieriou"] = matched["delta_hieriou"].abs()
    matched["abs_delta_struct"] = matched["delta_struct"].abs()

    diagnostics = {
        "num_rows_pred_a": int(len(a_df)),
        "num_rows_pred_b": int(len(b_df)),
        "num_matched_images": int(len(matched)),
        "num_only_in_pred_a": int((merged["_merge"] == "left_only").sum()),
        "num_only_in_pred_b": int((merged["_merge"] == "right_only").sum()),
        "missing_in_pred_b": sorted(merged.loc[merged["_merge"] == "left_only", "image_id"].astype(str).tolist()),
        "missing_in_pred_a": sorted(merged.loc[merged["_merge"] == "right_only", "image_id"].astype(str).tolist()),
    }

    ordered_cols = [
        "image_id",
        "miou_crf",
        "miou_best",
        "delta_miou",
        "hieriou_crf",
        "hieriou_best",
        "delta_hieriou",
        "struct_crf",
        "struct_best",
        "delta_struct",
    ]
    optional_cols = [col for col in ("persist_crf", "persist_best", "delta_persist", "edge_crf", "edge_best", "delta_edge") if col in matched.columns]
    tail_cols = [col for col in ("abs_delta_hieriou", "abs_delta_struct") if col in matched.columns]
    matched = matched[ordered_cols + optional_cols + tail_cols]
    matched = matched.sort_values(["delta_struct", "delta_hieriou", "image_id"], ascending=[False, False, True]).reset_index(drop=True)
    return matched, diagnostics


def build_class_diff_table(
    class_df: pd.DataFrame,
    pred_a: str,
    pred_b: str,
    class_names: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ensure_predictions_present(class_df, pred_a, pred_b, "per_class_metrics")
    cols = [col for col in ("class_id", "iou", "hieriou", "persist", "edge", "struct") if col in class_df.columns]

    a_df = class_df.loc[class_df["pred_name"] == pred_a, cols].copy()
    b_df = class_df.loc[class_df["pred_name"] == pred_b, cols].copy()
    a_df = a_df.rename(columns={col: f"{col}_crf" for col in cols if col != "class_id"})
    b_df = b_df.rename(columns={col: f"{col}_best" for col in cols if col != "class_id"})

    merged = a_df.merge(b_df, on="class_id", how="outer", indicator=True)
    matched = merged.loc[merged["_merge"] == "both"].copy()
    matched["delta_iou"] = matched["iou_best"] - matched["iou_crf"]
    matched["delta_hieriou"] = matched["hieriou_best"] - matched["hieriou_crf"]
    matched["delta_persist"] = matched["persist_best"] - matched["persist_crf"]
    matched["delta_edge"] = matched["edge_best"] - matched["edge_crf"]
    matched["delta_struct"] = matched["struct_best"] - matched["struct_crf"]

    if class_names is not None:
        matched["class_name"] = matched["class_id"].apply(
            lambda cid: class_names[int(cid)] if 0 <= int(cid) < len(class_names) else f"class_{int(cid)}"
        )
    else:
        matched["class_name"] = matched["class_id"].apply(lambda cid: f"class_{int(cid)}")

    matched["struct_gain_source"] = np.where(
        matched["delta_edge"] > matched["delta_persist"],
        "edge_dominant",
        np.where(
            matched["delta_persist"] > matched["delta_edge"],
            "persist_dominant",
            "tie",
        ),
    )

    diagnostics = {
        "num_rows_pred_a": int(len(a_df)),
        "num_rows_pred_b": int(len(b_df)),
        "num_matched_classes": int(len(matched)),
        "num_only_in_pred_a": int((merged["_merge"] == "left_only").sum()),
        "num_only_in_pred_b": int((merged["_merge"] == "right_only").sum()),
    }

    ordered_cols = [
        "class_id",
        "class_name",
        "iou_crf",
        "iou_best",
        "delta_iou",
        "hieriou_crf",
        "hieriou_best",
        "delta_hieriou",
        "persist_crf",
        "persist_best",
        "delta_persist",
        "edge_crf",
        "edge_best",
        "delta_edge",
        "struct_crf",
        "struct_best",
        "delta_struct",
        "struct_gain_source",
    ]
    matched = matched[ordered_cols].sort_values(["delta_struct", "delta_hieriou", "class_id"], ascending=[False, False, True]).reset_index(drop=True)
    return matched, diagnostics


def assign_image_diff_buckets(
    image_diff_df: pd.DataFrame,
    threshold_struct_pos: float,
    threshold_struct_nonpos: float,
    threshold_hier_pos: float,
    threshold_hier_nonpos: float,
    threshold_struct_neg: float,
    threshold_hier_neg: float,
) -> pd.DataFrame:
    df = image_diff_df.copy()
    cond_structure_gain = (df["delta_struct"] > threshold_struct_pos) & (df["delta_hieriou"] <= threshold_hier_nonpos)
    cond_semantic_gain = (df["delta_hieriou"] > threshold_hier_pos) & (df["delta_struct"] <= threshold_struct_nonpos)
    cond_both_gain = (df["delta_hieriou"] > threshold_hier_pos) & (df["delta_struct"] > threshold_struct_pos)
    cond_both_drop = (df["delta_hieriou"] < threshold_hier_neg) & (df["delta_struct"] < threshold_struct_neg)

    df["bucket_name"] = np.select(
        [cond_structure_gain, cond_semantic_gain, cond_both_gain, cond_both_drop],
        [
            "structure_gain_semantic_non_gain",
            "semantic_gain_structure_non_gain",
            "both_gain",
            "both_drop",
        ],
        default="other",
    )

    df["bucket_priority_score"] = 0.0
    df.loc[df["bucket_name"] == "structure_gain_semantic_non_gain", "bucket_priority_score"] = (
        df["delta_struct"] - df["delta_hieriou"]
    )
    df.loc[df["bucket_name"] == "semantic_gain_structure_non_gain", "bucket_priority_score"] = (
        df["delta_hieriou"] - df["delta_struct"]
    )
    df.loc[df["bucket_name"] == "both_gain", "bucket_priority_score"] = df["delta_struct"] + df["delta_hieriou"]
    df.loc[df["bucket_name"] == "both_drop", "bucket_priority_score"] = (
        df["delta_struct"].abs() + df["delta_hieriou"].abs()
    )
    df.loc[df["bucket_name"] == "other", "bucket_priority_score"] = (
        0.5 * df["delta_struct"].abs() + 0.5 * df["delta_hieriou"].abs()
    )
    return df


def summarize_bucket_counts(image_bucket_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bucket_name, group in image_bucket_df.groupby("bucket_name", dropna=False):
        rows.append(
            {
                "bucket_name": str(bucket_name),
                "count": int(len(group)),
                "fraction": float(len(group) / len(image_bucket_df)) if len(image_bucket_df) else float("nan"),
                "mean_delta_miou": float(group["delta_miou"].mean()),
                "mean_delta_hieriou": float(group["delta_hieriou"].mean()),
                "mean_delta_struct": float(group["delta_struct"].mean()),
                "median_delta_miou": float(group["delta_miou"].median()),
                "median_delta_hieriou": float(group["delta_hieriou"].median()),
                "median_delta_struct": float(group["delta_struct"].median()),
                "max_delta_struct": float(group["delta_struct"].max()),
                "min_delta_struct": float(group["delta_struct"].min()),
                "max_delta_hieriou": float(group["delta_hieriou"].max()),
                "min_delta_hieriou": float(group["delta_hieriou"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "bucket_name"], ascending=[False, True]).reset_index(drop=True)


def select_top_bucket_examples(image_bucket_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selections: List[pd.DataFrame] = []
    bucket_order = [
        "structure_gain_semantic_non_gain",
        "semantic_gain_structure_non_gain",
        "both_gain",
        "both_drop",
    ]
    for bucket_name in bucket_order:
        bucket_df = image_bucket_df.loc[image_bucket_df["bucket_name"] == bucket_name].copy()
        if bucket_df.empty:
            continue
        bucket_df = bucket_df.sort_values(
            ["bucket_priority_score", "delta_struct", "delta_hieriou", "image_id"],
            ascending=[False, False, False, True],
        ).head(top_n)
        bucket_df.insert(0, "rank_in_bucket", np.arange(1, len(bucket_df) + 1))
        selections.append(bucket_df)
    if not selections:
        return pd.DataFrame(columns=["bucket_name", "rank_in_bucket", "image_id"])
    return pd.concat(selections, ignore_index=True)


def rank_top_struct_gain_classes(class_diff_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return class_diff_df.sort_values(
        ["delta_struct", "delta_hieriou", "delta_iou", "class_id"],
        ascending=[False, False, False, True],
    ).head(top_k).reset_index(drop=True)


def rank_semantic_structure_conflicts(class_diff_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    filtered = class_diff_df.loc[(class_diff_df["delta_hieriou"] > 0.0) & (class_diff_df["delta_struct"] <= 0.0)].copy()
    return filtered.sort_values(
        ["delta_hieriou", "delta_iou", "delta_struct", "class_id"],
        ascending=[False, False, True, True],
    ).head(top_k).reset_index(drop=True)


def rank_struct_gain_conflicts(class_diff_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    filtered = class_diff_df.loc[(class_diff_df["delta_struct"] > 0.0) & (class_diff_df["delta_hieriou"] <= 0.0)].copy()
    return filtered.sort_values(
        ["delta_struct", "delta_edge", "delta_persist", "class_id"],
        ascending=[False, False, False, True],
    ).head(top_k).reset_index(drop=True)


def plot_image_diff_scatter(image_bucket_df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 6))
    bucket_order = [
        "structure_gain_semantic_non_gain",
        "semantic_gain_structure_non_gain",
        "both_gain",
        "both_drop",
        "other",
    ]
    for bucket_name in bucket_order:
        subset = image_bucket_df.loc[image_bucket_df["bucket_name"] == bucket_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["delta_hieriou"],
            subset["delta_struct"],
            s=18,
            alpha=0.7,
            label=f"{bucket_name} ({len(subset)})",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("delta hierIoU (pred_b - pred_a)")
    ax.set_ylabel("delta StructScore (pred_b - pred_a)")
    ax.set_title("Image-level Differential Scatter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_class_diff_scatter(class_diff_df: pd.DataFrame, out_path: Path, annotate_top_k: int = 10) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(class_diff_df["delta_hieriou"], class_diff_df["delta_struct"], s=28, alpha=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("delta hierIoU (pred_b - pred_a)")
    ax.set_ylabel("delta StructScore (pred_b - pred_a)")
    ax.set_title("Class-level Differential Scatter")

    top_to_annotate = pd.concat(
        [
            class_diff_df.nlargest(annotate_top_k // 2, "delta_struct"),
            class_diff_df.nsmallest(annotate_top_k // 2, "delta_struct"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["class_id"])
    for _, row in top_to_annotate.iterrows():
        ax.annotate(
            str(row["class_name"]),
            (row["delta_hieriou"], row["delta_struct"]),
            fontsize=7,
            alpha=0.85,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_top_struct_gain_classes(top_struct_df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [str(x) for x in top_struct_df["class_name"]]
    ax.barh(labels[::-1], top_struct_df["delta_struct"].to_numpy()[::-1])
    ax.set_xlabel("delta StructScore")
    ax.set_ylabel("class")
    ax.set_title("Top Struct Gain Classes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_semantic_structure_conflict_classes(conflict_df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [str(x) for x in conflict_df["class_name"]]
    positions = np.arange(len(conflict_df))
    ax.barh(positions + 0.18, conflict_df["delta_hieriou"], height=0.36, label="delta hierIoU")
    ax.barh(positions - 0.18, conflict_df["delta_struct"], height=0.36, label="delta StructScore")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("metric delta")
    ax.set_ylabel("class")
    ax.set_title("Top Semantic-Structure Conflict Classes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_pair_lookup(gt_dir: Path, pred_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    pairs = match_gt_pred_files(gt_dir, pred_dir, warnings)
    return {pair.image_id: pair for pair in pairs}, warnings


def format_metric_triplet(label: str, miou: float, hieriou: float, struct: float) -> str:
    return f"{label}\nmIoU={miou:.4f}\nhierIoU={hieriou:.4f}\nStruct={struct:.4f}"


def save_case_visualizations(
    case_df: pd.DataFrame,
    gt_dir: Path,
    pred_dir_a: Path,
    pred_dir_b: Path,
    ignore_index: int,
    out_dir: Path,
    pred_a_label: str,
    pred_b_label: str,
) -> Dict[str, Any]:
    pair_lookup_a, warnings_a = build_pair_lookup(gt_dir, pred_dir_a)
    pair_lookup_b, warnings_b = build_pair_lookup(gt_dir, pred_dir_b)
    warnings = warnings_a + warnings_b
    saved = 0
    skipped: List[str] = []

    for _, row in case_df.iterrows():
        image_id = str(row["image_id"])
        pair_a = pair_lookup_a.get(image_id)
        pair_b = pair_lookup_b.get(image_id)
        if pair_a is None or pair_b is None:
            skipped.append(image_id)
            continue

        gt = load_label_map(pair_a.gt_path, offset=0, ignore_index=ignore_index)
        pred_a = load_label_map(pair_a.pred_path, offset=0, ignore_index=ignore_index)
        pred_b = load_label_map(pair_b.pred_path, offset=0, ignore_index=ignore_index)

        num_classes = int(
            max(
                np.max(gt[gt != ignore_index]) if np.any(gt != ignore_index) else 0,
                np.max(pred_a[pred_a != ignore_index]) if np.any(pred_a != ignore_index) else 0,
                np.max(pred_b[pred_b != ignore_index]) if np.any(pred_b != ignore_index) else 0,
            )
            + 1
        )
        gt_rgb = colorize_label_map(gt, num_classes=num_classes, ignore_index=ignore_index)
        pred_a_rgb = colorize_label_map(pred_a, num_classes=num_classes, ignore_index=ignore_index)
        pred_b_rgb = colorize_label_map(pred_b, num_classes=num_classes, ignore_index=ignore_index)

        bucket_name = str(row["bucket_name"])
        out_path = ensure_dir(out_dir / safe_slug(bucket_name)) / f"{int(row['rank_in_bucket']):02d}_{safe_slug(image_id)}.png"
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_rgb, interpolation="nearest")
        axes[0].set_title(f"GT\n{image_id}")
        axes[1].imshow(pred_a_rgb, interpolation="nearest")
        axes[1].set_title(
            format_metric_triplet(pred_a_label, row["miou_crf"], row["hieriou_crf"], row["struct_crf"])
        )
        axes[2].imshow(pred_b_rgb, interpolation="nearest")
        axes[2].set_title(
            format_metric_triplet(pred_b_label, row["miou_best"], row["hieriou_best"], row["struct_best"])
        )
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{bucket_name} | delta_mIoU={row['delta_miou']:.4f} "
            f"| delta_hierIoU={row['delta_hieriou']:.4f} | delta_struct={row['delta_struct']:.4f}"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved += 1

    return {
        "saved_visualizations": saved,
        "skipped_images": skipped,
        "warnings": warnings,
    }


def resolve_overall_deltas(
    overall_df: Optional[pd.DataFrame],
    image_diff_df: pd.DataFrame,
    pred_a: str,
    pred_b: str,
) -> Dict[str, Any]:
    if overall_df is not None:
        ensure_predictions_present(overall_df, pred_a, pred_b, "overall_metrics")
        row_a = overall_df.loc[overall_df["pred_name"] == pred_a].iloc[0]
        row_b = overall_df.loc[overall_df["pred_name"] == pred_b].iloc[0]
        return {
            "source": "overall_metrics.csv",
            "miou_pred_a": float(row_a["miou_mean"]),
            "miou_pred_b": float(row_b["miou_mean"]),
            "hieriou_pred_a": float(row_a["hieriou_mean"]),
            "hieriou_pred_b": float(row_b["hieriou_mean"]),
            "struct_pred_a": float(row_a["struct_mean"]),
            "struct_pred_b": float(row_b["struct_mean"]),
            "delta_miou_overall": float(row_b["miou_mean"] - row_a["miou_mean"]),
            "delta_hieriou_overall": float(row_b["hieriou_mean"] - row_a["hieriou_mean"]),
            "delta_struct_overall": float(row_b["struct_mean"] - row_a["struct_mean"]),
        }

    return {
        "source": "image_level_metrics_mean_fallback",
        "miou_pred_a": float(image_diff_df["miou_crf"].mean()),
        "miou_pred_b": float(image_diff_df["miou_best"].mean()),
        "hieriou_pred_a": float(image_diff_df["hieriou_crf"].mean()),
        "hieriou_pred_b": float(image_diff_df["hieriou_best"].mean()),
        "struct_pred_a": float(image_diff_df["struct_crf"].mean()),
        "struct_pred_b": float(image_diff_df["struct_best"].mean()),
        "delta_miou_overall": float(image_diff_df["delta_miou"].mean()),
        "delta_hieriou_overall": float(image_diff_df["delta_hieriou"].mean()),
        "delta_struct_overall": float(image_diff_df["delta_struct"].mean()),
    }


def dataframe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    return records if isinstance(records, list) else []


def export_diff_analysis_summary(
    summary_path: Path,
    overall_deltas: Dict[str, Any],
    image_diagnostics: Dict[str, Any],
    class_diagnostics: Dict[str, Any],
    bucket_summary_df: pd.DataFrame,
    top_struct_df: pd.DataFrame,
    semantic_conflict_df: pd.DataFrame,
    struct_conflict_df: pd.DataFrame,
    class_diff_df: pd.DataFrame,
    visual_summary: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    finite_class_df = class_diff_df.loc[
        np.isfinite(class_diff_df["delta_edge"]) & np.isfinite(class_diff_df["delta_persist"])
    ].copy()
    summary = {
        "config": {
            "image_metrics": str(args.image_metrics),
            "class_metrics": str(args.class_metrics),
            "overall_metrics": str(args.overall_metrics) if args.overall_metrics is not None else None,
            "pred_a": args.pred_a,
            "pred_b": args.pred_b,
            "output_dir": str(args.output_dir),
            "threshold_struct_pos": args.threshold_struct_pos,
            "threshold_struct_nonpos": args.threshold_struct_nonpos,
            "threshold_hier_pos": args.threshold_hier_pos,
            "threshold_hier_nonpos": args.threshold_hier_nonpos,
            "threshold_struct_neg": args.threshold_struct_neg,
            "threshold_hier_neg": args.threshold_hier_neg,
            "top_k_classes": args.top_k_classes,
            "top_n_cases_per_bucket": args.top_n_cases_per_bucket,
            "gt_dir": str(args.gt_dir) if args.gt_dir is not None else None,
            "pred_dir_a": str(args.pred_dir_a) if args.pred_dir_a is not None else None,
            "pred_dir_b": str(args.pred_dir_b) if args.pred_dir_b is not None else None,
        },
        "overall_deltas": overall_deltas,
        "image_diagnostics": image_diagnostics,
        "class_diagnostics": class_diagnostics,
        "image_bucket_counts": dataframe_records(bucket_summary_df),
        "top_struct_gain_classes": dataframe_records(top_struct_df),
        "top_semantic_structure_conflict_classes": dataframe_records(semantic_conflict_df),
        "top_struct_gain_conflict_classes": dataframe_records(struct_conflict_df),
        "edge_vs_persist_statistics": {
            "num_classes_edge_dominant": int((finite_class_df["delta_edge"] > finite_class_df["delta_persist"]).sum()),
            "num_classes_persist_dominant": int((finite_class_df["delta_persist"] >= finite_class_df["delta_edge"]).sum()),
            "num_positive_struct_gain_edge_dominant": int(
                ((finite_class_df["delta_struct"] > 0.0) & (finite_class_df["delta_edge"] > finite_class_df["delta_persist"])).sum()
            ),
            "num_positive_struct_gain_persist_dominant": int(
                ((finite_class_df["delta_struct"] > 0.0) & (finite_class_df["delta_persist"] >= finite_class_df["delta_edge"])).sum()
            ),
        },
        "case_visualizations": visual_summary,
    }
    save_json(summary, summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze structural-vs-semantic differential behavior between two prediction results.")
    parser.add_argument("--image_metrics", type=Path, required=True, help="Path to image_level_metrics.csv")
    parser.add_argument("--class_metrics", type=Path, required=True, help="Path to per_class_metrics.csv")
    parser.add_argument("--overall_metrics", type=Path, default=None, help="Optional path to overall_metrics.csv")
    parser.add_argument("--pred_a", type=str, required=True, help="Baseline prediction name, e.g. CRF")
    parser.add_argument("--pred_b", type=str, required=True, help="Comparison prediction name, e.g. CRF_Morph_Small")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for differential analysis")
    parser.add_argument("--gt_dir", type=Path, default=None, help="Optional GT directory for case visualization")
    parser.add_argument("--pred_dir_a", type=Path, default=None, help="Optional prediction directory for pred_a")
    parser.add_argument("--pred_dir_b", type=Path, default=None, help="Optional prediction directory for pred_b")
    parser.add_argument("--class_names", type=Path, default=None, help="Optional class names json/txt file")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore label value for optional visualizations")
    parser.add_argument("--threshold_struct_pos", type=float, default=0.01)
    parser.add_argument("--threshold_struct_nonpos", type=float, default=0.0)
    parser.add_argument("--threshold_hier_pos", type=float, default=0.002)
    parser.add_argument("--threshold_hier_nonpos", type=float, default=0.0)
    parser.add_argument("--threshold_struct_neg", type=float, default=-0.01)
    parser.add_argument("--threshold_hier_neg", type=float, default=-0.002)
    parser.add_argument("--top_k_classes", type=int, default=20, help="Top-K classes to export for ranking tables")
    parser.add_argument("--top_n_cases_per_bucket", type=int, default=10, help="How many case-study pairs to export per bucket")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "plots")

    tables = load_metrics_tables(args.image_metrics, args.class_metrics, args.overall_metrics)
    class_names = load_class_names_optional(args.class_names)

    image_diff_df, image_diagnostics = build_image_diff_table(tables["image"], args.pred_a, args.pred_b)
    class_diff_df, class_diagnostics = build_class_diff_table(tables["class"], args.pred_a, args.pred_b, class_names)
    image_bucket_df = assign_image_diff_buckets(
        image_diff_df,
        threshold_struct_pos=args.threshold_struct_pos,
        threshold_struct_nonpos=args.threshold_struct_nonpos,
        threshold_hier_pos=args.threshold_hier_pos,
        threshold_hier_nonpos=args.threshold_hier_nonpos,
        threshold_struct_neg=args.threshold_struct_neg,
        threshold_hier_neg=args.threshold_hier_neg,
    )
    bucket_summary_df = summarize_bucket_counts(image_bucket_df)
    case_pairs_df = select_top_bucket_examples(image_bucket_df, top_n=args.top_n_cases_per_bucket)

    top_struct_df = rank_top_struct_gain_classes(class_diff_df, top_k=args.top_k_classes)
    semantic_conflict_df = rank_semantic_structure_conflicts(class_diff_df, top_k=args.top_k_classes)
    struct_conflict_df = rank_struct_gain_conflicts(class_diff_df, top_k=args.top_k_classes)
    overall_deltas = resolve_overall_deltas(tables["overall"], image_diff_df, args.pred_a, args.pred_b)

    image_diff_df.to_csv(args.output_dir / "image_diff_metrics.csv", index=False)
    bucket_summary_df.to_csv(args.output_dir / "image_diff_bucket_summary.csv", index=False)
    class_diff_df.to_csv(args.output_dir / "class_diff_metrics.csv", index=False)
    top_struct_df.to_csv(args.output_dir / "class_struct_gain_topk.csv", index=False)
    semantic_conflict_df.to_csv(args.output_dir / "class_semantic_gain_conflict_topk.csv", index=False)
    struct_conflict_df.to_csv(args.output_dir / "class_struct_gain_conflict_topk.csv", index=False)
    case_pairs_df.to_csv(args.output_dir / "case_study_pairs.csv", index=False)

    plot_image_diff_scatter(image_bucket_df, args.output_dir / "plots" / "image_diff_scatter.png")
    plot_class_diff_scatter(class_diff_df, args.output_dir / "plots" / "class_diff_scatter.png")
    plot_top_struct_gain_classes(top_struct_df.head(min(args.top_k_classes, 15)), args.output_dir / "plots" / "top_struct_gain_classes.png")
    plot_semantic_structure_conflict_classes(
        semantic_conflict_df.head(min(args.top_k_classes, 15)),
        args.output_dir / "plots" / "top_semantic_structure_conflict_classes.png",
    )

    visual_summary: Optional[Dict[str, Any]] = None
    if args.gt_dir is not None or args.pred_dir_a is not None or args.pred_dir_b is not None:
        if args.gt_dir is None or args.pred_dir_a is None or args.pred_dir_b is None:
            raise ValueError("gt_dir, pred_dir_a, and pred_dir_b must all be provided together for case visualizations")
        visual_summary = save_case_visualizations(
            case_pairs_df,
            gt_dir=args.gt_dir,
            pred_dir_a=args.pred_dir_a,
            pred_dir_b=args.pred_dir_b,
            ignore_index=args.ignore_index,
            out_dir=args.output_dir / "case_visualizations",
            pred_a_label=args.pred_a,
            pred_b_label=args.pred_b,
        )

    export_diff_analysis_summary(
        summary_path=args.output_dir / "summary.json",
        overall_deltas=overall_deltas,
        image_diagnostics=image_diagnostics,
        class_diagnostics=class_diagnostics,
        bucket_summary_df=bucket_summary_df,
        top_struct_df=top_struct_df,
        semantic_conflict_df=semantic_conflict_df,
        struct_conflict_df=struct_conflict_df,
        class_diff_df=class_diff_df,
        visual_summary=visual_summary,
        args=args,
    )

    print(f"Saved differential analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
