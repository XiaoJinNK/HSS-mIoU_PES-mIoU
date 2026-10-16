import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Bootstrap the structural parameter-selection run to quantify raw-score uncertainty and plateau-center stability."
    )
    parser.add_argument(
        "--selection-run-dir",
        type=str,
        required=True,
        help="Path to a completed non-degenerate parameter-selection run directory.",
    )
    parser.add_argument("--num-bootstrap", type=int, default=2000, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument(
        "--plateau-ratio",
        type=float,
        default=None,
        help="Optional override for the plateau ratio. Defaults to the original run config.",
    )
    parser.add_argument(
        "--focus-pairs",
        type=str,
        default="3,2;5,2",
        help="Semicolon-separated list of r,bw pairs for pairwise score-gap summaries, e.g. '3,2;5,2'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <selection-run-dir>/bootstrap_stability.",
    )
    return parser


def parse_focus_pairs(text: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        left, right = part.split(",")
        pairs.append((int(left), int(right)))
    if len(pairs) < 2:
        raise ValueError("focus-pairs must contain at least two r,bw pairs")
    return pairs


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_pair(pair: Tuple[int, int]) -> str:
    return f"({pair[0]},{pair[1]})"


def _compute_instance_contributions(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["instance_id"]), str(row["perturbation_name"]))].append(row)

    monotone_struct = []
    monotone_target = []
    separability_struct = []
    separability_target = []
    stability_values = []
    edge_decoupling = []
    persist_decoupling = []
    structure_advantage = []

    for group_rows in groups.values():
        group_rows = sorted(group_rows, key=lambda item: int(item["level_index"]))
        family = str(group_rows[0]["family"])
        struct_seq = np.asarray([float(row["struct"]) for row in group_rows], dtype=np.float64)
        persist_seq = np.asarray([float(row["persist"]) for row in group_rows], dtype=np.float64)
        edge_seq = np.asarray([float(row["edge"]) for row in group_rows], dtype=np.float64)
        iou_seq = np.asarray([float(row["binary_iou"]) for row in group_rows], dtype=np.float64)

        if family == "edge":
            target_seq = edge_seq
        elif family == "persist":
            target_seq = persist_seq
        else:
            target_seq = struct_seq

        monotone_struct.append(float(np.all(np.diff(struct_seq) <= 1e-9)))
        monotone_target.append(float(np.all(np.diff(target_seq) <= 1e-9)))

        separability_struct.append(float(np.clip(struct_seq[0] - struct_seq[-1], 0.0, 1.0)))
        separability_target.append(float(np.clip(target_seq[0] - target_seq[-1], 0.0, 1.0)))

        target_drop = 1.0 - target_seq
        if target_drop[-1] > 1e-8:
            stability_values.append(float(np.clip(1.0 - min(target_drop[0] / target_drop[-1], 1.0), 0.0, 1.0)))

        for row in group_rows:
            persist_drop = 1.0 - float(row["persist"])
            edge_drop = 1.0 - float(row["edge"])
            struct_drop = 1.0 - float(row["struct"])
            iou_drop = 1.0 - float(row["binary_iou"])

            if family == "edge":
                edge_decoupling.append(float(edge_drop > persist_drop))
            elif family == "persist":
                persist_decoupling.append(float(persist_drop > edge_drop))
                structure_advantage.append(float(struct_drop > iou_drop))

    return {
        "mono_struct": float(np.mean(monotone_struct)),
        "mono_target": float(np.mean(monotone_target)),
        "sep_struct": float(np.mean(separability_struct)),
        "sep_target": float(np.mean(separability_target)),
        "stab_sum": float(np.sum(stability_values)),
        "stab_cnt": float(len(stability_values)),
        "edge_dec": float(np.mean(edge_decoupling)),
        "persist_dec": float(np.mean(persist_decoupling)),
        "comp_adv": float(np.mean(structure_advantage)),
    }


def load_pair_contributions(run_dir: Path) -> Tuple[List[int], Dict[Tuple[int, int], Dict[str, np.ndarray]], Dict[Tuple[int, int], float]]:
    detail_dir = run_dir / "controlled" / "details"
    summary_df = pd.read_csv(run_dir / "analysis" / "selection_summary.csv")
    real_score_map = {
        (int(row.radius), int(row.boundary_width)): float(row.real_score)
        for row in summary_df.itertuples(index=False)
    }

    pair_data: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    base_instance_ids: List[int] = []

    for path in sorted(detail_dir.glob("radius_*_bw_*.json")):
        obj = _load_json(path)
        radius = int(obj["summary"]["radius"])
        boundary_width = int(obj["summary"]["boundary_width"])
        pair = (radius, boundary_width)

        rows = obj["samples"]
        by_instance: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_instance[int(row["instance_id"])].append(row)

        instance_ids = sorted(by_instance.keys())
        if not base_instance_ids:
            base_instance_ids = instance_ids
        elif instance_ids != base_instance_ids:
            raise RuntimeError(f"Instance ids mismatch for pair {pair}")

        component_names = [
            "mono_struct",
            "mono_target",
            "sep_struct",
            "sep_target",
            "stab_sum",
            "stab_cnt",
            "edge_dec",
            "persist_dec",
            "comp_adv",
        ]
        arrays = {name: np.zeros(len(base_instance_ids), dtype=np.float64) for name in component_names}

        for idx, instance_id in enumerate(base_instance_ids):
            contrib = _compute_instance_contributions(by_instance[instance_id])
            for name in component_names:
                arrays[name][idx] = contrib[name]

        pair_data[pair] = arrays

    return base_instance_ids, pair_data, real_score_map


def compute_bootstrap_scores(
    pair_data: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    real_score_map: Dict[Tuple[int, int], float],
    instance_count: int,
    num_bootstrap: int,
    seed: int,
    controlled_weight: float,
    real_weight: float,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, instance_count, size=(num_bootstrap, instance_count), endpoint=False)

    pairs = sorted(pair for pair in pair_data.keys() if pair[0] != 0)
    scores = np.zeros((num_bootstrap, len(pairs)), dtype=np.float64)
    total_weight = controlled_weight + real_weight

    for col, pair in enumerate(pairs):
        data = pair_data[pair]
        mono_struct = data["mono_struct"][bootstrap_indices].mean(axis=1)
        mono_target = data["mono_target"][bootstrap_indices].mean(axis=1)
        sep_struct = data["sep_struct"][bootstrap_indices].mean(axis=1)
        sep_target = data["sep_target"][bootstrap_indices].mean(axis=1)
        edge_dec = data["edge_dec"][bootstrap_indices].mean(axis=1)
        persist_dec = data["persist_dec"][bootstrap_indices].mean(axis=1)
        comp_adv = data["comp_adv"][bootstrap_indices].mean(axis=1)
        stab_sum = data["stab_sum"][bootstrap_indices].sum(axis=1)
        stab_cnt = data["stab_cnt"][bootstrap_indices].sum(axis=1)
        stability = np.divide(stab_sum, stab_cnt, out=np.zeros_like(stab_sum), where=stab_cnt > 0)

        monotonicity = 0.5 * (mono_struct + mono_target)
        decoupling = 0.5 * (edge_dec + persist_dec)
        separability = 0.5 * (sep_struct + sep_target)

        calibration = (
            0.30 * monotonicity
            + 0.25 * decoupling
            + 0.20 * separability
            + 0.15 * stability
            + 0.10 * comp_adv
        )
        real_score = float(real_score_map[pair])
        if not np.isfinite(real_score):
            real_score = 0.0
        scores[:, col] = (controlled_weight * calibration + real_weight * real_score) / total_weight

    return pairs, scores


def choose_plateau_center(
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    plateau_ratio: float,
) -> np.ndarray:
    chosen = np.zeros(scores.shape[0], dtype=np.int64)
    pair_arr = np.asarray(pairs, dtype=np.int64)

    for row_idx in range(scores.shape[0]):
        row_scores = scores[row_idx]
        best = float(np.max(row_scores))
        plateau_mask = row_scores >= (plateau_ratio * best)
        plateau_pairs = pair_arr[plateau_mask]
        plateau_scores = row_scores[plateau_mask]

        center_radius = float(np.median(plateau_pairs[:, 0]))
        center_bw = float(np.median(plateau_pairs[:, 1]))
        distances = np.abs(plateau_pairs[:, 0] - center_radius) + np.abs(plateau_pairs[:, 1] - center_bw)

        order = sorted(
            range(len(plateau_pairs)),
            key=lambda idx: (
                float(distances[idx]),
                -float(plateau_scores[idx]),
                int(plateau_pairs[idx, 0]),
                int(plateau_pairs[idx, 1]),
            ),
        )
        chosen_pair = tuple(int(v) for v in plateau_pairs[order[0]])
        chosen[row_idx] = pairs.index(chosen_pair)

    return chosen


def plot_summary(
    output_path: Path,
    raw_counter: Counter,
    center_counter: Counter,
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    focus_a: Tuple[int, int],
    focus_b: Tuple[int, int],
) -> None:
    pair_to_col = {pair: idx for idx, pair in enumerate(pairs)}
    gap = scores[:, pair_to_col[focus_b]] - scores[:, pair_to_col[focus_a]]

    def _counter_df(counter: Counter) -> pd.DataFrame:
        rows = [{"pair": _format_pair(pair), "count": count, "freq": count / scores.shape[0]} for pair, count in counter.items()]
        rows.sort(key=lambda item: (-item["count"], item["pair"]))
        return pd.DataFrame(rows)

    raw_df = _counter_df(raw_counter).head(6)
    center_df = _counter_df(center_counter).head(6)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    axes[0].bar(raw_df["pair"], raw_df["freq"], color="#4C78A8")
    axes[0].set_title("Raw Argmax Frequency")
    axes[0].set_ylabel("Bootstrap frequency")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylim(0.0, max(0.05, raw_df["freq"].max() * 1.15))

    axes[1].bar(center_df["pair"], center_df["freq"], color="#F58518")
    axes[1].set_title("Plateau-Center Frequency")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0.0, max(0.05, center_df["freq"].max() * 1.15))

    axes[2].hist(gap, bins=40, color="#54A24B", alpha=0.85)
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title(f"Score Gap: {_format_pair(focus_b)} - {_format_pair(focus_a)}")
    axes[2].set_xlabel("Bootstrap final-score gap")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.selection_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "bootstrap_stability")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_choice = _load_json(run_dir / "analysis" / "final_choice.json")
    config = final_choice["config"]
    controlled_weight = float(config["controlled_weight"])
    real_weight = float(config["real_weight"])
    plateau_ratio = float(args.plateau_ratio) if args.plateau_ratio is not None else float(config["plateau_ratio"])

    focus_pairs = parse_focus_pairs(args.focus_pairs)
    focus_a, focus_b = focus_pairs[0], focus_pairs[1]

    instance_ids, pair_data, real_score_map = load_pair_contributions(run_dir)
    pairs, scores = compute_bootstrap_scores(
        pair_data=pair_data,
        real_score_map=real_score_map,
        instance_count=len(instance_ids),
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        controlled_weight=controlled_weight,
        real_weight=real_weight,
    )

    pair_to_col = {pair: idx for idx, pair in enumerate(pairs)}
    raw_choice = np.argmax(scores, axis=1)
    center_choice = choose_plateau_center(pairs=pairs, scores=scores, plateau_ratio=plateau_ratio)

    raw_counter = Counter(pairs[idx] for idx in raw_choice.tolist())
    center_counter = Counter(pairs[idx] for idx in center_choice.tolist())

    pair_rows = []
    for pair in pairs:
        col = pair_to_col[pair]
        pair_scores = scores[:, col]
        pair_rows.append(
            {
                "radius": pair[0],
                "boundary_width": pair[1],
                "mean_final_score": float(np.mean(pair_scores)),
                "std_final_score": float(np.std(pair_scores)),
                "ci95_low": float(np.quantile(pair_scores, 0.025)),
                "ci95_high": float(np.quantile(pair_scores, 0.975)),
                "raw_best_count": int(raw_counter[pair]),
                "raw_best_freq": float(raw_counter[pair] / args.num_bootstrap),
                "plateau_center_count": int(center_counter[pair]),
                "plateau_center_freq": float(center_counter[pair] / args.num_bootstrap),
            }
        )
    pair_df = pd.DataFrame(pair_rows).sort_values(["boundary_width", "radius"])
    pair_df.to_csv(output_dir / "bootstrap_pair_summary.csv", index=False)

    focus_gap = scores[:, pair_to_col[focus_b]] - scores[:, pair_to_col[focus_a]]
    focus_summary = {
        "pair_a": {"radius": focus_a[0], "boundary_width": focus_a[1]},
        "pair_b": {"radius": focus_b[0], "boundary_width": focus_b[1]},
        "pair_b_minus_pair_a": {
            "mean_gap": float(np.mean(focus_gap)),
            "median_gap": float(np.median(focus_gap)),
            "ci95_low": float(np.quantile(focus_gap, 0.025)),
            "ci95_high": float(np.quantile(focus_gap, 0.975)),
            "prob_pair_b_gt_pair_a": float(np.mean(focus_gap > 0.0)),
        },
    }

    raw_top = [
        {"radius": pair[0], "boundary_width": pair[1], "count": int(count), "freq": float(count / args.num_bootstrap)}
        for pair, count in raw_counter.most_common()
    ]
    center_top = [
        {"radius": pair[0], "boundary_width": pair[1], "count": int(count), "freq": float(count / args.num_bootstrap)}
        for pair, count in center_counter.most_common()
    ]

    summary = {
        "selection_run_dir": str(run_dir),
        "num_instances": len(instance_ids),
        "num_bootstrap": args.num_bootstrap,
        "seed": args.seed,
        "plateau_ratio": plateau_ratio,
        "controlled_weight": controlled_weight,
        "real_weight": real_weight,
        "focus_gap_summary": focus_summary,
        "raw_argmax_frequency": raw_top,
        "plateau_center_frequency": center_top,
    }
    (output_dir / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_summary(
        output_path=output_dir / "bootstrap_selection_stability.png",
        raw_counter=raw_counter,
        center_counter=center_counter,
        pairs=pairs,
        scores=scores,
        focus_a=focus_a,
        focus_b=focus_b,
    )

    print(f"[bootstrap] output_dir={output_dir}")
    print(f"[bootstrap] raw argmax top3={raw_top[:3]}")
    print(f"[bootstrap] plateau-center top3={center_top[:3]}")
    print(f"[bootstrap] focus gap summary={focus_summary['pair_b_minus_pair_a']}")


if __name__ == "__main__":
    main()
