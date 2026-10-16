#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多相似度矩阵对比实验

比较 Hybrid、PathSim、WuP 三种相似度矩阵在 Corruption 实验中的表现
"""

import argparse
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

from semantic_metrics import (
    compute_confusion_matrix,
    compute_miou_from_confusion,
    compute_hier_soft_iou_from_confusion,
    evaluate_dataset
)
from corrupt import (
    compute_corruption_targets,
    generate_synthetic_predictions
)
from io_utils import (
    load_Hij, find_image_pairs, save_json, save_csv, save_latex_table,
    get_experiment_meta, load_mask
)


def load_all_similarity_matrices(sim_dirs: dict) -> dict:
    """
    加载所有相似度矩阵
    
    Args:
        sim_dirs: {'hybrid': Path, 'pathsim': Path, 'wup': Path}
    
    Returns:
        {'hybrid': np.ndarray, 'pathsim': np.ndarray, 'wup': np.ndarray}
    """
    matrices = {}
    for name, path in sim_dirs.items():
        if path is None:
            continue
        s_path = Path(path) / "S.npy" if Path(path).is_dir() else Path(path)
        if s_path.exists():
            matrices[name] = np.load(s_path).astype(np.float64)
            print(f"  加载 {name}: {s_path}")
            print(f"    形状: {matrices[name].shape}")
            print(f"    值域: [{matrices[name].min():.4f}, {matrices[name].max():.4f}]")
            print(f"    均值 (非对角): {matrices[name][np.triu_indices(150, k=1)].mean():.4f}")
        else:
            print(f"  警告: {name} 不存在: {s_path}")
    return matrices


def analyze_similarity_matrices(matrices: dict):
    """分析各相似度矩阵的特性"""
    print("\n" + "=" * 60)
    print("相似度矩阵特性分析")
    print("=" * 60)
    
    for name, S in matrices.items():
        n = S.shape[0]
        off_diag = S[np.triu_indices(n, k=1)]
        
        print(f"\n{name}:")
        print(f"  对角线: min={np.diag(S).min():.4f}, max={np.diag(S).max():.4f}")
        print(f"  非对角线: min={off_diag.min():.4f}, max={off_diag.max():.4f}, "
              f"mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
        
        # 相似度分布
        percentiles = [10, 25, 50, 75, 90]
        pct_values = np.percentile(off_diag, percentiles)
        print(f"  百分位数: " + ", ".join([f"P{p}={v:.4f}" for p, v in zip(percentiles, pct_values)]))
    
    # 两两相关性
    if len(matrices) > 1:
        print("\n矩阵间相关性 (Pearson):")
        names = list(matrices.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                S1 = matrices[n1][np.triu_indices(150, k=1)]
                S2 = matrices[n2][np.triu_indices(150, k=1)]
                corr = np.corrcoef(S1, S2)[0, 1]
                print(f"  {n1} vs {n2}: r = {corr:.4f}")


def run_comparison_experiment(
    image_pairs: list,
    matrices: dict,
    ratios: list,
    synth_dir: Path,
    num_classes: int = 150,
    ignore_label: int = 255,
    gt_offset: int = 0,
    seed: int = 42,
    target_k: int = 5,
    skip_generate: bool = False
) -> dict:
    """
    运行对比实验
    
    使用同一套 corrupted predictions，用不同相似度矩阵计算 Hier-mIoU
    """
    
    # 使用第一个矩阵（通常是 hybrid）来生成 corruption targets
    primary_matrix_name = list(matrices.keys())[0]
    primary_matrix = matrices[primary_matrix_name]
    
    if not skip_generate:
        # 1. 计算替换目标（使用主矩阵）
        print(f"\n[1] 使用 {primary_matrix_name} 计算替换目标...")
        targets = compute_corruption_targets(primary_matrix, k=target_k, seed=seed)
        
        # 2. 生成合成预测
        print("\n[2] 生成合成预测...")
        generate_synthetic_predictions(
            image_pairs, targets, ratios, synth_dir,
            num_classes, ignore_label, gt_offset,
            seed=seed, save_format='png', verbose=True
        )
    else:
        print("\n[1-2] 跳过生成，使用已有预测...")
    
    # 3. 用每种相似度矩阵评测
    print("\n[3] 使用各相似度矩阵评测...")
    
    all_results = []
    
    for sim_name, S in matrices.items():
        print(f"\n  === {sim_name.upper()} ===")
        
        for r in ratios:
            r_str = f"r{int(r*100):02d}"
            
            for corrupt_type in ['fine', 'coarse', 'random']:
                pred_dir = synth_dir / corrupt_type / r_str
                pairs = [(gt_path, pred_dir / f"{name}.png", name) 
                         for gt_path, _, name in image_pairs]
                
                # 累计混淆矩阵
                confusion_total = np.zeros((num_classes, num_classes), dtype=np.int64)
                for gt_path, pred_path, name in pairs:
                    gt = load_mask(gt_path, offset=gt_offset)
                    pred = load_mask(pred_path)
                    confusion = compute_confusion_matrix(gt, pred, num_classes, ignore_label)
                    confusion_total += confusion
                
                # 计算指标
                miou_result = compute_miou_from_confusion(confusion_total)
                hier_result = compute_hier_soft_iou_from_confusion(confusion_total, S)
                
                delta = hier_result['hier_miou'] - miou_result['miou']
                
                all_results.append({
                    'similarity': sim_name,
                    'r': r,
                    'type': corrupt_type,
                    'mIoU': miou_result['miou'],
                    'hierIoU': hier_result['hier_miou'],
                    'delta': delta
                })
            
            # 打印该比例的结果
            sim_results = [res for res in all_results 
                          if res['similarity'] == sim_name and res['r'] == r]
            deltas = {res['type']: res['delta'] for res in sim_results}
            print(f"    r={r}: Δfine={deltas['fine']:.4f}, Δrand={deltas['random']:.4f}, Δcoarse={deltas['coarse']:.4f}")
    
    return all_results


def compute_discrimination_score(results: list, sim_name: str) -> dict:
    """
    计算区分度指标
    
    关键指标：
    1. Delta Gap: (Δ_fine - Δ_coarse) 越大越好
    2. Ordering Score: fine > random > coarse 的比例
    3. Sensitivity: Δ 对 r 的斜率
    """
    sim_results = [r for r in results if r['similarity'] == sim_name]
    
    ratios = sorted(set(r['r'] for r in sim_results))
    
    delta_gaps = []
    ordering_correct = 0
    total_ratios = len(ratios)
    
    sensitivity_fine = []
    sensitivity_coarse = []
    
    for r in ratios:
        r_results = {res['type']: res['delta'] for res in sim_results if res['r'] == r}
        
        delta_fine = r_results['fine']
        delta_coarse = r_results['coarse']
        delta_random = r_results['random']
        
        # Delta Gap
        gap = delta_fine - delta_coarse
        delta_gaps.append(gap)
        
        # Ordering
        if delta_fine > delta_random > delta_coarse:
            ordering_correct += 1
        
        sensitivity_fine.append((r, delta_fine))
        sensitivity_coarse.append((r, delta_coarse))
    
    # 计算敏感度（斜率）
    def linear_slope(points):
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0.0
    
    return {
        'mean_delta_gap': np.mean(delta_gaps),
        'max_delta_gap': np.max(delta_gaps),
        'min_delta_gap': np.min(delta_gaps),
        'ordering_accuracy': ordering_correct / total_ratios,
        'sensitivity_fine': linear_slope(sensitivity_fine),
        'sensitivity_coarse': linear_slope(sensitivity_coarse),
        'delta_gaps_by_r': {r: g for r, g in zip(ratios, delta_gaps)}
    }


def plot_comparison_results(results: list, out_dir: Path):
    """绘制对比图表"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    # 颜色和样式
    colors = {'hybrid': '#2196F3', 'pathsim': '#4CAF50', 'wup': '#FF9800'}
    markers = {'hybrid': 'o', 'pathsim': 's', 'wup': '^'}
    
    # 图1: Delta (hierIoU - mIoU) 对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, corrupt_type in enumerate(['fine', 'random', 'coarse']):
        ax = axes[idx]
        
        for sim_name in sim_names:
            sim_data = [r for r in results if r['similarity'] == sim_name and r['type'] == corrupt_type]
            sim_data.sort(key=lambda x: x['r'])
            
            x = [d['r'] for d in sim_data]
            y = [d['delta'] for d in sim_data]
            
            ax.plot(x, y, f'-{markers.get(sim_name, "o")}', 
                   color=colors.get(sim_name, 'gray'),
                   label=sim_name.upper(), markersize=8, linewidth=2)
        
        ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
        ax.set_ylabel('Δ (Hier-mIoU - mIoU)', fontsize=12)
        ax.set_title(f'{corrupt_type.capitalize()} Corruption', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'delta_comparison_by_type.png', dpi=150)
    plt.savefig(out_dir / 'delta_comparison_by_type.pdf')
    plt.close()
    
    # 图2: Delta Gap (fine - coarse) 对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    x = np.arange(len(ratios))
    
    for i, sim_name in enumerate(sim_names):
        gaps = []
        for r in ratios:
            fine_delta = next((res['delta'] for res in results 
                              if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'fine'), 0)
            coarse_delta = next((res['delta'] for res in results 
                                if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'coarse'), 0)
            gaps.append(fine_delta - coarse_delta)
        
        ax.bar(x + i * width, gaps, width, 
               label=sim_name.upper(), 
               color=colors.get(sim_name, 'gray'),
               alpha=0.8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Delta Gap (Δ_fine - Δ_coarse)', fontsize=12)
    ax.set_title('Discrimination Ability: Delta Gap Comparison', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{r:.0%}' for r in ratios])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'delta_gap_comparison.png', dpi=150)
    plt.savefig(out_dir / 'delta_gap_comparison.pdf')
    plt.close()
    
    # 图3: 综合对比（核心图）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: mIoU (应该三种方法一样)
    ax = axes[0]
    for corrupt_type, style in [('fine', '-'), ('random', '--'), ('coarse', ':')]:
        # 只画一个 sim_name 的 mIoU（因为都一样）
        data = [r for r in results if r['similarity'] == sim_names[0] and r['type'] == corrupt_type]
        data.sort(key=lambda x: x['r'])
        x = [d['r'] for d in data]
        y = [d['mIoU'] for d in data]
        ax.plot(x, y, style, color='black', label=corrupt_type, linewidth=2, marker='o', markersize=6)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('mIoU (Same for All Similarity Matrices)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    # 右图: Delta Gap 对比
    ax = axes[1]
    for sim_name in sim_names:
        gaps = []
        for r in ratios:
            fine_delta = next((res['delta'] for res in results 
                              if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'fine'), 0)
            coarse_delta = next((res['delta'] for res in results 
                                if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'coarse'), 0)
            gaps.append(fine_delta - coarse_delta)
        
        ax.plot(ratios, gaps, f'-{markers.get(sim_name, "o")}',
               color=colors.get(sim_name, 'gray'),
               label=sim_name.upper(), markersize=8, linewidth=2)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Delta Gap (Δ_fine - Δ_coarse)', fontsize=12)
    ax.set_title('Discrimination Ability Comparison', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'main_comparison.png', dpi=150)
    plt.savefig(out_dir / 'main_comparison.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


def generate_latex_comparison_table(results: list, scores: dict, out_path: Path):
    """生成 LaTeX 对比表格"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of Similarity Matrices for Hierarchical Metric}",
        r"\label{tab:sim_comparison}",
        r"\small",
        r"\begin{tabular}{l" + "c" * (len(ratios) + 3) + "}",
        r"\toprule",
        r"Method & " + " & ".join([f"$r={r:.0%}$" for r in ratios]) + r" & Mean $\Delta$Gap & Ordering Acc. \\",
        r"\midrule",
    ]
    
    for sim_name in sim_names:
        gaps = []
        for r in ratios:
            fine_delta = next((res['delta'] for res in results 
                              if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'fine'), 0)
            coarse_delta = next((res['delta'] for res in results 
                                if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'coarse'), 0)
            gaps.append(fine_delta - coarse_delta)
        
        score = scores[sim_name]
        
        row = f"{sim_name.upper()} & "
        row += " & ".join([f"{g:.4f}" for g in gaps])
        row += f" & {score['mean_delta_gap']:.4f}"
        row += f" & {score['ordering_accuracy']:.0%}"
        row += r" \\"
        
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="多相似度矩阵对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_comparison_experiments.py \\
      --gt_dir /path/to/ade150_val_gt \\
      --hybrid_dir similarity_output/ade150_hybrid_inv \\
      --pathsim_dir similarity_output/ade150 \\
      --wup_dir similarity_output/ade150_wup \\
      --out_dir output/comparison_experiment \\
      --ratios 0.05 0.10 0.20 0.30
        """
    )
    
    # 输入路径
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="GT masks 目录")
    parser.add_argument("--hybrid_dir", type=Path, required=True,
                        help="Hybrid 相似度矩阵目录")
    parser.add_argument("--pathsim_dir", type=Path, default=None,
                        help="PathSim 相似度矩阵目录")
    parser.add_argument("--wup_dir", type=Path, default=None,
                        help="WuP 相似度矩阵目录")
    
    # 输出
    parser.add_argument("--out_dir", type=Path, default=Path("./output/comparison"),
                        help="输出目录")
    
    # 参数
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--gt_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_k", type=int, default=5)
    parser.add_argument("--ratios", type=float, nargs='+',
                        default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--skip_generate", action="store_true",
                        help="跳过生成步骤")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("多相似度矩阵对比实验")
    print("=" * 70)
    
    # 加载相似度矩阵
    print("\n[1] 加载相似度矩阵...")
    sim_dirs = {
        'hybrid': args.hybrid_dir,
        'pathsim': args.pathsim_dir,
        'wup': args.wup_dir
    }
    matrices = load_all_similarity_matrices(sim_dirs)
    
    if not matrices:
        print("错误: 未加载任何相似度矩阵")
        return 1
    
    # 分析矩阵
    analyze_similarity_matrices(matrices)
    
    # 加载图片
    print("\n[2] 加载图片...")
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"  找到 {len(image_pairs)} 张图片")
    
    if not image_pairs:
        print("错误: 未找到图片")
        return 1
    
    # 运行对比实验
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_dir = out_dir / "synth_preds"
    
    results = run_comparison_experiment(
        image_pairs, matrices, args.ratios, synth_dir,
        args.num_classes, args.ignore_label, args.gt_offset,
        args.seed, args.target_k, args.skip_generate
    )
    
    # 计算区分度得分
    print("\n[4] 计算区分度指标...")
    scores = {}
    for sim_name in matrices.keys():
        scores[sim_name] = compute_discrimination_score(results, sim_name)
        print(f"\n  {sim_name.upper()}:")
        print(f"    Mean Delta Gap: {scores[sim_name]['mean_delta_gap']:.4f}")
        print(f"    Ordering Accuracy: {scores[sim_name]['ordering_accuracy']:.0%}")
        print(f"    Sensitivity (fine): {scores[sim_name]['sensitivity_fine']:.4f}")
    
    # 绘图
    print("\n[5] 绘制图表...")
    plot_comparison_results(results, out_dir / "plots")
    
    # 保存结果
    print("\n[6] 保存结果...")
    save_json(results, out_dir / "logs" / "comparison_results.json")
    save_json(scores, out_dir / "logs" / "discrimination_scores.json")
    save_csv(results, out_dir / "tables" / "comparison_results.csv")
    generate_latex_comparison_table(results, scores, out_dir / "tables" / "comparison_table.tex")
    
    # 打印最终排名
    print("\n" + "=" * 70)
    print("最终结果排名 (按 Mean Delta Gap)")
    print("=" * 70)
    
    ranked = sorted(scores.items(), key=lambda x: x[1]['mean_delta_gap'], reverse=True)
    for rank, (name, score) in enumerate(ranked, 1):
        print(f"  {rank}. {name.upper()}: Mean ΔGap = {score['mean_delta_gap']:.4f}, "
              f"Ordering = {score['ordering_accuracy']:.0%}")
    
    best = ranked[0][0]
    print(f"\n  → 最佳方法: {best.upper()}")
    
    print("\n" + "=" * 70)
    print(f"实验完成，结果保存到: {out_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())