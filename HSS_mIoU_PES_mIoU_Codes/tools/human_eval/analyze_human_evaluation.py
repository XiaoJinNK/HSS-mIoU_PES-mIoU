#!/usr/bin/env python3
"""
Analyze responses for the bundled human-evaluation questionnaire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def normalize_choice(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.upper()
    if text in {"A", "B", "TIE"}:
        return text
    return ""


def load_hidden_items(package_dir: Path) -> pd.DataFrame:
    path = package_dir / "answer_key.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing answer key table in: {package_dir}")
    df = pd.read_csv(path)
    return df


def melt_responses(response_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(response_csv)
    if "respondent_id" not in df.columns:
        raise ValueError("Response CSV must contain a respondent_id column.")
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        rid = row["respondent_id"]
        for col in df.columns:
            if not col.endswith("_choice"):
                continue
            q = col[:-7]
            choice = normalize_choice(row[col])
            conf_col = f"{q}_confidence"
            conf = row[conf_col] if conf_col in df.columns else np.nan
            try:
                conf_val = float(conf) if str(conf).strip() else np.nan
            except Exception:
                conf_val = np.nan
            rows.append(
                {
                    "respondent_id": rid,
                    "question_code": q,
                    "choice": choice,
                    "confidence": conf_val,
                }
            )
    return pd.DataFrame(rows)


def compute_repeat_consistency(answer_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    repeat_items = item_df.loc[item_df["role"] == "repeat", ["question_code", "repeat_of", "metric_preference"]].copy()
    if repeat_items.empty:
        return pd.DataFrame(columns=["respondent_id", "repeat_consistency"])

    source_lookup = (
        item_df.loc[item_df["role"] == "core", ["source_item_id", "question_code", "metric_preference"]]
        .drop_duplicates("source_item_id")
        .set_index("source_item_id")
        .to_dict("index")
    )

    rows = []
    for respondent_id, group in answer_df.groupby("respondent_id"):
        agree = 0
        total = 0
        answers = group.set_index("question_code")["choice"].to_dict()
        for _, rep in repeat_items.iterrows():
            source = source_lookup.get(rep["repeat_of"])
            if source is None:
                continue
            a1 = normalize_choice(answers.get(source["question_code"], ""))
            a2 = normalize_choice(answers.get(rep["question_code"], ""))
            if not a1 or not a2 or a1 == "TIE" or a2 == "TIE":
                continue
            total += 1
            metric1 = source["metric_preference"]
            metric2 = rep["metric_preference"]
            same = (a1 == metric1 and a2 == metric2) or (a1 != metric1 and a2 != metric2)
            if same:
                agree += 1
        rows.append(
            {
                "respondent_id": respondent_id,
                "repeat_consistency": float(agree) / float(total) if total else np.nan,
                "repeat_pairs_used": total,
            }
        )
    return pd.DataFrame(rows)


def analyze(package_dir: Path, response_csv: Path, out_dir: Path) -> None:
    item_df = load_hidden_items(package_dir)
    answer_df = melt_responses(response_csv)
    merged = answer_df.merge(item_df, on="question_code", how="left")

    merged["choice"] = merged["choice"].map(normalize_choice)
    merged["is_answered"] = merged["choice"] != ""
    merged["is_tie"] = merged["choice"] == "TIE"
    merged["agrees_metric_strict"] = (merged["choice"] == merged["metric_preference"]).astype(float)
    merged.loc[~merged["is_answered"] | merged["is_tie"], "agrees_metric_strict"] = np.nan
    merged["agrees_metric_tie05"] = merged["agrees_metric_strict"]
    merged.loc[merged["is_tie"], "agrees_metric_tie05"] = 0.5
    merged.loc[~merged["is_answered"], "agrees_metric_tie05"] = np.nan
    merged["confidence_filled"] = merged["confidence"].fillna(1.0).clip(lower=1.0, upper=4.0)
    merged["weighted_agreement"] = merged["agrees_metric_tie05"] * merged["confidence_filled"]

    repeat_df = compute_repeat_consistency(answer_df, item_df)

    summary_rows = []
    for task in ["semantic", "structural"]:
        for role in ["practice", "core", "repeat"]:
            subset = merged.loc[(merged["task"] == task) & (merged["role"] == role)].copy()
            if subset.empty:
                continue
            strict = subset["agrees_metric_strict"].dropna()
            tie05 = subset["agrees_metric_tie05"].dropna()
            weighted = subset["weighted_agreement"].dropna()
            weight_denom = subset.loc[subset["agrees_metric_tie05"].notna(), "confidence_filled"].sum()
            summary_rows.append(
                {
                    "task": task,
                    "role": role,
                    "n_answers": int(subset["is_answered"].sum()),
                    "n_ties": int(subset["is_tie"].sum()),
                    "strict_agreement": float(strict.mean()) if not strict.empty else np.nan,
                    "tie05_agreement": float(tie05.mean()) if not tie05.empty else np.nan,
                    "confidence_weighted_agreement": float(weighted.sum() / weight_denom) if weight_denom > 0 else np.nan,
                }
            )

    respondent_rows = []
    for respondent_id, group in merged.groupby("respondent_id"):
        row = {"respondent_id": respondent_id}
        for task in ["semantic", "structural"]:
            subset = group.loc[(group["task"] == task) & (group["role"] == "core")]
            row[f"{task}_core_agreement"] = float(subset["agrees_metric_tie05"].dropna().mean()) if not subset.empty else np.nan
            row[f"{task}_practice_agreement"] = float(
                group.loc[(group["task"] == task) & (group["role"] == "practice"), "agrees_metric_tie05"].dropna().mean()
            ) if not group.loc[group["task"] == task].empty else np.nan
        respondent_rows.append(row)
    respondent_df = pd.DataFrame(respondent_rows).merge(repeat_df, on="respondent_id", how="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "merged_answers.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_task.csv", index=False)
    respondent_df.to_csv(out_dir / "respondent_quality.csv", index=False)

    report = {
        "response_csv": str(response_csv),
        "n_respondents": int(answer_df["respondent_id"].nunique()),
        "summary_by_task": summary_rows,
        "mean_repeat_consistency": float(repeat_df["repeat_consistency"].dropna().mean()) if not repeat_df.empty else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Human Evaluation Analysis",
        "",
        f"- Respondents: {report['n_respondents']}",
        f"- Mean repeat consistency: {report['mean_repeat_consistency']}",
        "",
        "## Task Summary",
        "",
    ]
    for row in summary_rows:
        md_lines.append(
            f"- {row['task']} / {row['role']}: strict={row['strict_agreement']}, "
            f"tie05={row['tie05_agreement']}, weighted={row['confidence_weighted_agreement']}"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human-evaluation questionnaire responses.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--response-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.package_dir / "analysis_results")
    analyze(args.package_dir, args.response_csv, out_dir)
    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
