import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_HEATMAP_METRICS = [
    "calibration_score",
    "monotonicity_score",
    "decoupling_score",
    "separability_score",
    "stability_score",
    "complementarity_score",
]


def parse_name_list(text: str) -> List[str]:
    """Parse a comma-separated list of column names."""
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one metric name must be provided")
    return names


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for CSP calibration analysis."""
    parser = argparse.ArgumentParser("Analyze CSP calibration summary")
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to calibration summary.csv")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to store analysis outputs")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top parameter pairs to export")
    parser.add_argument(
        "--heatmap-metrics",
        type=parse_name_list,
        default=DEFAULT_HEATMAP_METRICS,
        help="Comma-separated metric columns to draw as heatmaps",
    )
    return parser


def _choose_output_dir(summary_csv: Path, output_dir_arg: str) -> Path:
    """Resolve the output directory for analysis artifacts."""
    if output_dir_arg:
        return Path(output_dir_arg)
    return summary_csv.parent / "analysis"


def _safe_float(value: float) -> str:
    """Format a float value for report text."""
    return f"{value:.6f}"


def _write_report(
    report_path: Path,
    summary_csv: Path,
    df: pd.DataFrame,
    top_df: pd.DataFrame,
    heatmap_metrics: List[str],
) -> None:
    """Write a compact markdown report for the calibration run."""
    best = top_df.iloc[0]
    worst = df.sort_values("calibration_score", ascending=True).iloc[0]

    lines = [
        "# CSP Calibration Analysis",
        "",
        f"- Source summary: `{summary_csv}`",
        f"- Number of parameter pairs: {len(df)}",
        f"- Best pair: radius={int(best['radius'])}, boundary_width={int(best['boundary_width'])}, calibration_score={_safe_float(best['calibration_score'])}",
        f"- Worst pair: radius={int(worst['radius'])}, boundary_width={int(worst['boundary_width'])}, calibration_score={_safe_float(worst['calibration_score'])}",
        "",
        "## Best Pair Detail",
        "",
        f"- monotonicity_score: {_safe_float(best['monotonicity_score'])}",
        f"- decoupling_score: {_safe_float(best['decoupling_score'])}",
        f"- separability_score: {_safe_float(best['separability_score'])}",
        f"- stability_score: {_safe_float(best['stability_score'])}",
        f"- complementarity_score: {_safe_float(best['complementarity_score'])}",
        f"- mean_struct: {_safe_float(best['mean_struct'])}",
        f"- mean_persist: {_safe_float(best['mean_persist'])}",
        f"- mean_edge: {_safe_float(best['mean_edge'])}",
        f"- mean_binary_iou: {_safe_float(best['mean_binary_iou'])}",
        "",
        "## Heatmaps",
        "",
    ]

    for metric in heatmap_metrics:
        lines.append(f"- `{metric}.png`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _save_heatmap(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save one radius-by-boundary_width heatmap."""
    pivot = df.pivot(index="boundary_width", columns="radius", values=metric)
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.title(metric)
    plt.xlabel("radius")
    plt.ylabel("boundary_width")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main(args: argparse.Namespace) -> None:
    """Analyze calibration summary results and save plots/tables."""
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    output_dir = _choose_output_dir(summary_csv, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    required = {
        "radius",
        "boundary_width",
        "calibration_score",
        "monotonicity_score",
        "decoupling_score",
        "separability_score",
        "stability_score",
        "complementarity_score",
        "mean_struct",
        "mean_persist",
        "mean_edge",
        "mean_binary_iou",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"summary.csv is missing required columns: {missing}")

    df = df.sort_values("calibration_score", ascending=False).reset_index(drop=True)
    top_df = df.head(args.top_k).copy()

    top_csv = output_dir / "topk.csv"
    top_df.to_csv(top_csv, index=False)

    full_sorted_csv = output_dir / "summary_sorted.csv"
    df.to_csv(full_sorted_csv, index=False)

    report_path = output_dir / "report.md"
    _write_report(report_path, summary_csv, df, top_df, args.heatmap_metrics)

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    created_heatmaps: Dict[str, str] = {}

    for metric in args.heatmap_metrics:
        if metric not in df.columns:
            raise ValueError(f"Requested heatmap metric '{metric}' not found in summary.csv")
        output_path = heatmap_dir / f"{metric}.png"
        _save_heatmap(df, metric, output_path)
        created_heatmaps[metric] = str(output_path.resolve())

    analysis_json = output_dir / "analysis.json"
    analysis_json.write_text(
        json.dumps(
            {
                "summary_csv": str(summary_csv.resolve()),
                "output_dir": str(output_dir.resolve()),
                "top_k": args.top_k,
                "best_pair": {
                    "radius": int(top_df.iloc[0]["radius"]),
                    "boundary_width": int(top_df.iloc[0]["boundary_width"]),
                    "calibration_score": float(top_df.iloc[0]["calibration_score"]),
                },
                "created_heatmaps": created_heatmaps,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[Analyze] summary source: {summary_csv}")
    print(f"[Analyze] analysis output: {output_dir}")
    print("[Analyze] Top parameter pairs:")
    for _, row in top_df.iterrows():
        print(
            f"  radius={int(row['radius'])}, boundary_width={int(row['boundary_width'])}, "
            f"calibration_score={row['calibration_score']:.4f}"
        )
    print(f"[Analyze] top-k CSV: {top_csv}")
    print(f"[Analyze] report: {report_path}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
