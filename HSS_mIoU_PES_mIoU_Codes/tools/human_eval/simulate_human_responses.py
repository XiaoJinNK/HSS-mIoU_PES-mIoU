#!/usr/bin/env python3
"""
Generate transparent synthetic responses for the bundled human-evaluation package.

The output is for pipeline validation, power checks, and drafting a clearly
marked simulated-study analysis. It must not be reported as collected human
responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def opposite_choice(choice: str) -> str:
    if choice == "A":
        return "B"
    if choice == "B":
        return "A"
    raise ValueError(f"Expected A or B, got {choice!r}")


def clip_probability(p: float, lo: float = 0.02, hi: float = 0.96) -> float:
    return float(min(max(p, lo), hi))


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def choose_confidence(rng: np.random.Generator, decision: str, ability: float, role: str) -> int:
    role_bonus = 0.20 if role == "practice" else 0.0
    if decision == "metric":
        p4 = clip_probability(0.34 + 0.12 * ability + role_bonus, 0.15, 0.58)
        p3 = 0.45
        p2 = max(0.02, 1.0 - p4 - p3 - 0.05)
        probs = np.array([0.05, p2, p3, p4], dtype=float)
    elif decision == "other":
        p4 = clip_probability(0.12 + 0.04 * ability, 0.06, 0.24)
        p3 = 0.34
        p2 = 0.36
        probs = np.array([1.0 - p2 - p3 - p4, p2, p3, p4], dtype=float)
    else:
        probs = np.array([0.44, 0.40, 0.13, 0.03], dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice([1, 2, 3, 4], p=probs))


def decision_to_choice(decision: str, metric_preference: str) -> str:
    if decision == "tie":
        return "Tie"
    if decision == "metric":
        return metric_preference
    return opposite_choice(metric_preference)


def sample_base_decision(
    rng: np.random.Generator,
    item: pd.Series,
    item_effect: float,
    ability: float,
    low_quality: bool,
) -> str:
    task = str(item["task"])
    role = str(item["role"])

    if low_quality:
        base_align = 0.56 if role == "core" else 0.60
        base_tie = 0.15 if role == "core" else 0.12
    elif task == "semantic":
        base_align = {"practice": 0.88, "core": 0.76, "repeat": 0.76}[role]
        base_tie = {"practice": 0.035, "core": 0.070, "repeat": 0.070}[role]
    else:
        base_align = {"practice": 0.90, "core": 0.80, "repeat": 0.80}[role]
        base_tie = {"practice": 0.035, "core": 0.075, "repeat": 0.075}[role]

    align = clip_probability(base_align + 0.055 * ability + 0.035 * item_effect, 0.10, 0.94)
    tie = clip_probability(base_tie - 0.020 * ability - 0.010 * item_effect, 0.015, 0.22)
    other = max(0.0, 1.0 - align - tie)
    probs = np.array([align, other, tie], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(["metric", "other", "tie"], p=probs))


def sample_repeat_decision(
    rng: np.random.Generator,
    source_decision: str,
    task: str,
    ability: float,
    low_quality: bool,
) -> str:
    if low_quality:
        consistency = 0.58
    else:
        base = 0.88 if task == "semantic" else 0.90
        consistency = clip_probability(base + 0.045 * ability, 0.62, 0.98)

    if rng.random() < consistency:
        return source_decision

    if source_decision == "tie":
        return str(rng.choice(["metric", "other"], p=[0.58, 0.42]))

    if source_decision == "metric":
        return str(rng.choice(["other", "tie"], p=[0.72, 0.28]))
    return str(rng.choice(["metric", "tie"], p=[0.72, 0.28]))


def respondent_profile(rng: np.random.Generator) -> Tuple[float, bool]:
    low_quality = bool(rng.random() < 0.08)
    if low_quality:
        ability = float(rng.normal(-1.15, 0.55))
    else:
        ability = float(rng.normal(0.15, 0.85))
    return ability, low_quality


def generate(package_dir: Path, n: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    answer_key = package_dir / "answer_key.csv"
    item_df = pd.read_csv(answer_key)
    item_df["item_effect"] = 0.0
    for task, idx in item_df.groupby("task").groups.items():
        item_df.loc[idx, "item_effect"] = zscore(item_df.loc[idx, "selection_score"])

    rng = np.random.default_rng(seed)
    rows = []
    profile_rows = []

    for respondent_idx in range(1, n + 1):
        respondent_id = f"sim_{respondent_idx:03d}"
        ability, low_quality = respondent_profile(rng)
        row: Dict[str, object] = {
            "respondent_id": respondent_id,
            "data_origin": "synthetic_simulation",
            "simulation_seed": seed,
        }
        source_decisions: Dict[str, str] = {}

        for _, item in item_df.iterrows():
            q = str(item["question_code"])
            role = str(item["role"])
            task = str(item["task"])
            metric_preference = str(item["metric_preference"])

            if role == "repeat":
                source_key = str(item["repeat_of"])
                source_decision = source_decisions.get(source_key)
                if source_decision is None:
                    source_decision = sample_base_decision(
                        rng,
                        item,
                        float(item["item_effect"]),
                        ability,
                        low_quality,
                    )
                decision = sample_repeat_decision(rng, source_decision, task, ability, low_quality)
            else:
                decision = sample_base_decision(
                    rng,
                    item,
                    float(item["item_effect"]),
                    ability,
                    low_quality,
                )
                source_decisions[str(item["source_item_id"])] = decision

            row[f"{q}_choice"] = decision_to_choice(decision, metric_preference)
            row[f"{q}_confidence"] = choose_confidence(rng, decision, ability, role)

        rows.append(row)
        profile_rows.append(
            {
                "respondent_id": respondent_id,
                "latent_ability": ability,
                "low_quality_simulated": low_quality,
            }
        )

    response_df = pd.DataFrame(rows)
    ordered_cols = ["respondent_id", "data_origin", "simulation_seed"]
    for q in item_df["question_code"].tolist():
        ordered_cols.extend([f"{q}_choice", f"{q}_confidence"])
    response_df = response_df[ordered_cols]

    metadata: Dict[str, object] = {
        "data_origin": "synthetic_simulation",
        "warning": "Synthetic responses for validation/drafting only; do not report as collected human data.",
        "n_respondents": n,
        "seed": seed,
        "package_dir": str(package_dir),
        "respondent_profile": {
            "low_quality_rate": float(pd.DataFrame(profile_rows)["low_quality_simulated"].mean()),
            "latent_ability_mean": float(pd.DataFrame(profile_rows)["latent_ability"].mean()),
            "latent_ability_sd": float(pd.DataFrame(profile_rows)["latent_ability"].std(ddof=1)),
        },
    }
    return response_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path("survey_materials/human_evaluation"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/simulated_human_responses.csv"))
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("outputs/simulated_human_responses_metadata.json"),
    )
    args = parser.parse_args()

    response_df, metadata = generate(args.package_dir, args.n, args.seed)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    response_df.to_csv(args.out_csv, index=False)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved synthetic responses to: {args.out_csv}")
    print(f"Saved metadata to: {args.metadata_json}")


if __name__ == "__main__":
    main()
