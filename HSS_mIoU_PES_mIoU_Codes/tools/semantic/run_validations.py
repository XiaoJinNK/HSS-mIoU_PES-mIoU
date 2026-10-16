#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标验证实验套件 V3

包含：
- 实验 1：Near/Far-credit 防洗白测试
- 实验 2：真实预测验证——多指标排名
- 实验 3：Near/Mid/Far 错误分解
"""

import argparse
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

from semantic_metrics import (
    compute_confusion_matrix,
    compute_miou_from_confusion,
    compute_hier_soft_iou_from_confusion
)
from corrupt import load_mask
from io_utils import (
    find_image_pairs, save_json, save_csv, get_experiment_meta
)


# ============================================================================
# 通用工具函数
# ============================================================================

def load_similarity_matrices(paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
    """加载所有相似度矩阵"""
    matrices = {}
    print("\n加载相似度矩阵:")
    for name, path in paths.items():
        if path is None:
            continue
        path = Path(path)
        if path.is_dir():
            s_path = path / "S.npy"
        else:
            s_path = path
        
        if s_path.exists():
            S = np.load(s_path).astype(np.float64)
            matrices[name] = S
            off_diag = S[np.triu_indices(S.shape[0], k=1)]
            print(f"  {name}: shape={S.shape}, range=[{S.min():.4f}, {S.max():.4f}], "
                  f"off-diag mean={off_diag.mean():.4f}")
        else:
            print(f"  {name}: 未找到 ({s_path})")
    
    return matrices


def bootstrap_single_metric(values: np.ndarray, n_bootstrap: int = 1000,
                           confidence_level: float = 0.95, seed: int = 42) -> dict:
    """对单个指标进行 Bootstrap"""
    rng = np.random.default_rng(seed)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means.append(values[indices].mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'bootstrap_std': float(bootstrap_means.std())
    }


def generate_latex_table(lines: List[str], path: Path):
    """保存 LaTeX 表格"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def collect_error_similarities(pairs: List[Tuple[Path, Path, str]],
                               S: np.ndarray,
                               num_classes: int,
                               ignore_label: int,
                               gt_offset: int) -> Tuple[np.ndarray, defaultdict]:
    """
    收集一组 GT/Pred 配对中的所有错误像素相似度与 confusion 统计
    """
    all_error_sims = []
    confusion_pairs = defaultdict(int)

    for gt_path, pred_path, _ in tqdm(pairs, desc="  Collecting errors"):
        gt = load_mask(gt_path, offset=gt_offset)
        pred = load_mask(pred_path)

        valid = (gt != ignore_label) & (gt >= 0) & (gt < num_classes)
        valid &= (pred >= 0) & (pred < num_classes)
        error_mask = (gt != pred) & valid

        if error_mask.sum() == 0:
            continue

        gt_err = gt[error_mask]
        pred_err = pred[error_mask]

        for g, p in zip(gt_err, pred_err):
            sim = S[g, p]
            all_error_sims.append(sim)
            confusion_pairs[(int(g), int(p))] += 1

    return np.array(all_error_sims, dtype=np.float64), confusion_pairs


# ============================================================================
# 实验 1：Near/Far-credit 防洗白测试
# ============================================================================

def run_far_credit_experiment(args) -> dict:
    """
    实验 1：Near/Far-credit 防洗白测试
    
    对每个矩阵 S、每个 r、每种 corruption 类型，计算：
    1. 替换对相似度分布（P10/P25/P50/P75/P90、均值、std）
    2. Near-credit / Far-credit
    3. Strictness 指标
    """
    print("=" * 70)
    print("实验 1：Near/Far-credit 防洗白测试")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相似度矩阵
    matrix_paths = {
        'hybrid': args.hybrid_path,
        'pathsim': args.pathsim_path,
        'wup': args.wup_path
    }
    matrices = load_similarity_matrices(matrix_paths)
    
    if not matrices:
        return {'error': 'No matrices loaded'}
    
    # 加载 GT 图片
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"\n找到 {len(image_pairs)} 张 GT 图片")
    
    synth_root = Path(args.synth_pred_root)
    num_classes = args.num_classes
    ratios = args.ratios
    quantile = args.far_quantile  # 默认 0.1
    
    all_results = []
    similarity_distributions = defaultdict(list)
    
    # 遍历每个矩阵
    for sim_name, S in matrices.items():
        print(f"\n{'='*60}")
        print(f"处理矩阵: {sim_name.upper()}")
        print(f"{'='*60}")
        
        # 确定预测目录（如果每个矩阵有独立目录）
        # 优先检查 synth_root/sim_name，否则直接用 synth_root
        if (synth_root / sim_name).exists():
            matrix_synth_dir = synth_root / sim_name
        else:
            matrix_synth_dir = synth_root
        
        for r in ratios:
            r_str = f"r{int(r*100):02d}"
            print(f"\n  r={r}:")
            
            for corrupt_type in ['fine', 'coarse', 'random']:
                pred_dir = matrix_synth_dir / corrupt_type / r_str
                
                if not pred_dir.exists():
                    print(f"    跳过 {corrupt_type}（目录不存在: {pred_dir}）")
                    continue
                
                # 收集所有替换像素的相似度
                all_sims = []
                
                for gt_path, _, name in image_pairs:
                    pred_path = pred_dir / f"{name}.png"
                    if not pred_path.exists():
                        continue
                    
                    gt = load_mask(gt_path, offset=args.gt_offset)
                    pred = load_mask(pred_path)
                    
                    # 找到被替换的像素（gt != pred 且有效）
                    valid = (gt != args.ignore_label) & (gt >= 0) & (gt < num_classes)
                    valid &= (pred >= 0) & (pred < num_classes)
                    changed = (gt != pred) & valid
                    
                    if changed.sum() == 0:
                        continue
                    
                    gt_classes = gt[changed]
                    pred_classes = pred[changed]
                    
                    # 获取相似度
                    for g, p in zip(gt_classes, pred_classes):
                        all_sims.append(S[g, p])
                
                if not all_sims:
                    print(f"    {corrupt_type}: 无有效数据")
                    continue
                
                all_sims = np.array(all_sims)
                
                # 计算统计量
                percentiles = [10, 25, 50, 75, 90]
                pct_values = np.percentile(all_sims, percentiles)
                
                # Near/Far 阈值
                far_threshold = np.percentile(all_sims, quantile * 100)
                near_threshold = np.percentile(all_sims, (1 - quantile) * 100)
                
                # Far credit: bottom quantile
                far_mask = all_sims <= far_threshold
                far_credit = all_sims[far_mask].mean() if far_mask.sum() > 0 else 0
                far_count = far_mask.sum()
                
                # Near credit: top quantile
                near_mask = all_sims >= near_threshold
                near_credit = all_sims[near_mask].mean() if near_mask.sum() > 0 else 0
                near_count = near_mask.sum()
                
                result = {
                    'similarity': sim_name,
                    'r': r,
                    'type': corrupt_type,
                    'num_pixels': len(all_sims),
                    'mean_sim': float(all_sims.mean()),
                    'std_sim': float(all_sims.std()),
                    'P10': float(pct_values[0]),
                    'P25': float(pct_values[1]),
                    'P50': float(pct_values[2]),
                    'P75': float(pct_values[3]),
                    'P90': float(pct_values[4]),
                    'far_threshold': float(far_threshold),
                    'near_threshold': float(near_threshold),
                    'far_credit': float(far_credit),
                    'near_credit': float(near_credit),
                    'near_far_gap': float(near_credit - far_credit),
                    'far_count': int(far_count),
                    'near_count': int(near_count)
                }
                
                all_results.append(result)
                
                # 保存分布数据用于绘图
                key = (sim_name, r, corrupt_type)
                similarity_distributions[key] = all_sims
                
                print(f"    {corrupt_type}: mean={all_sims.mean():.4f}, "
                      f"Far={far_credit:.4f}, Near={near_credit:.4f}, "
                      f"Gap={near_credit-far_credit:.4f}")
    
    # 计算 Strictness 指标
    print("\n" + "=" * 60)
    print("Strictness 指标汇总")
    print("=" * 60)
    
    strictness_results = []
    for sim_name in matrices.keys():
        sim_results = [r for r in all_results if r['similarity'] == sim_name]
        
        # Mean Δrand, Δcoarse, Far-credit
        rand_results = [r for r in sim_results if r['type'] == 'random']
        coarse_results = [r for r in sim_results if r['type'] == 'coarse']
        fine_results = [r for r in sim_results if r['type'] == 'fine']
        
        mean_far_credit_fine = np.mean([r['far_credit'] for r in fine_results]) if fine_results else 0
        mean_far_credit_rand = np.mean([r['far_credit'] for r in rand_results]) if rand_results else 0
        mean_far_credit_coarse = np.mean([r['far_credit'] for r in coarse_results]) if coarse_results else 0
        
        mean_near_far_gap_fine = np.mean([r['near_far_gap'] for r in fine_results]) if fine_results else 0
        mean_near_far_gap_rand = np.mean([r['near_far_gap'] for r in rand_results]) if rand_results else 0
        mean_near_far_gap_coarse = np.mean([r['near_far_gap'] for r in coarse_results]) if coarse_results else 0
        
        strictness = {
            'similarity': sim_name,
            'mean_far_credit_fine': mean_far_credit_fine,
            'mean_far_credit_random': mean_far_credit_rand,
            'mean_far_credit_coarse': mean_far_credit_coarse,
            'mean_near_far_gap_fine': mean_near_far_gap_fine,
            'mean_near_far_gap_random': mean_near_far_gap_rand,
            'mean_near_far_gap_coarse': mean_near_far_gap_coarse,
        }
        strictness_results.append(strictness)
        
        print(f"\n{sim_name.upper()}:")
        print(f"  Mean Far-credit (fine/rand/coarse): "
              f"{mean_far_credit_fine:.4f} / {mean_far_credit_rand:.4f} / {mean_far_credit_coarse:.4f}")
        print(f"  Mean Near-Far Gap (fine/rand/coarse): "
              f"{mean_near_far_gap_fine:.4f} / {mean_near_far_gap_rand:.4f} / {mean_near_far_gap_coarse:.4f}")
    
    # 保存结果
    print("\n保存结果...")
    
    # CSV
    save_csv(all_results, out_dir / "tables" / "far_credit_summary.csv")
    save_csv(strictness_results, out_dir / "tables" / "strictness_summary.csv")
    
    # JSON
    save_json(all_results, out_dir / "logs" / "far_credit_results.json")
    save_json(strictness_results, out_dir / "logs" / "strictness_results.json")
    
    # LaTeX
    generate_far_credit_latex_table(all_results, strictness_results, 
                                    out_dir / "tables" / "far_credit_summary.tex")
    
    # 绘图
    plot_far_credit_curves(all_results, out_dir / "plots")
    plot_similarity_histograms(similarity_distributions, out_dir / "plots")
    
    return {
        'results': all_results,
        'strictness': strictness_results
    }


def generate_far_credit_latex_table(results: List[dict], strictness: List[dict], out_path: Path):
    """生成 Far-credit LaTeX 表格"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Near/Far Credit Analysis for Fine Corruption}",
        r"\label{tab:far_credit}",
        r"\small",
        r"\begin{tabular}{l" + "ccc" * len(ratios) + "}",
        r"\toprule",
    ]
    
    # 表头
    header1 = "Method"
    for r in ratios:
        header1 += f" & \\multicolumn{{3}}{{c}}{{$r={r:.0%}$}}"
    header1 += r" \\"
    lines.append(header1)
    
    header2 = ""
    for _ in ratios:
        header2 += " & Near & Far & Gap"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")
    
    # 数据行（只显示 fine corruption）
    for sim_name in sim_names:
        row = sim_name.upper()
        for r in ratios:
            fine_data = [res for res in results 
                        if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'fine']
            if fine_data:
                d = fine_data[0]
                row += f" & {d['near_credit']:.3f} & {d['far_credit']:.3f} & {d['near_far_gap']:.3f}"
            else:
                row += " & - & - & -"
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    generate_latex_table(lines, out_path)


def plot_far_credit_curves(results: List[dict], out_dir: Path):
    """绘制 Near/Far credit 曲线"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    colors = {'hybrid': '#2196F3', 'pathsim': '#4CAF50', 'wup': '#FF9800'}
    markers = {'hybrid': 'o', 'pathsim': 's', 'wup': '^'}
    
    # 图1: Near/Far credit vs r（三矩阵对比，只看 fine corruption）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图: Near credit
    ax = axes[0]
    for sim_name in sim_names:
        fine_data = [r for r in results 
                    if r['similarity'] == sim_name and r['type'] == 'fine']
        fine_data.sort(key=lambda x: x['r'])
        
        if fine_data:
            x = [d['r'] for d in fine_data]
            y = [d['near_credit'] for d in fine_data]
            ax.plot(x, y, f'-{markers.get(sim_name, "o")}',
                   color=colors.get(sim_name, 'gray'),
                   label=sim_name.upper(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Near Credit (top 10%)', fontsize=12)
    ax.set_title('Near Credit vs Corruption Ratio', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 右图: Far credit
    ax = axes[1]
    for sim_name in sim_names:
        fine_data = [r for r in results 
                    if r['similarity'] == sim_name and r['type'] == 'fine']
        fine_data.sort(key=lambda x: x['r'])
        
        if fine_data:
            x = [d['r'] for d in fine_data]
            y = [d['far_credit'] for d in fine_data]
            ax.plot(x, y, f'-{markers.get(sim_name, "o")}',
                   color=colors.get(sim_name, 'gray'),
                   label=sim_name.upper(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Far Credit (bottom 10%)', fontsize=12)
    ax.set_title('Far Credit vs Corruption Ratio (Lower is Better)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(out_dir / 'far_credit_vs_r.png', dpi=150)
    plt.savefig(out_dir / 'far_credit_vs_r.pdf')
    plt.close()
    
    # 图2: Near-Far Gap 对比（三种 corruption 类型）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    x = np.arange(len(sim_names))
    
    for i, corrupt_type in enumerate(['fine', 'random', 'coarse']):
        gaps = []
        for sim_name in sim_names:
            type_data = [r for r in results 
                        if r['similarity'] == sim_name and r['type'] == corrupt_type]
            if type_data:
                mean_gap = np.mean([d['near_far_gap'] for d in type_data])
                gaps.append(mean_gap)
            else:
                gaps.append(0)
        
        ax.bar(x + i * width, gaps, width, label=corrupt_type.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Similarity Matrix', fontsize=12)
    ax.set_ylabel('Mean Near-Far Gap', fontsize=12)
    ax.set_title('Near-Far Gap by Corruption Type', fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.upper() for s in sim_names])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'near_far_gap_by_type.png', dpi=150)
    plt.savefig(out_dir / 'near_far_gap_by_type.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


def plot_similarity_histograms(distributions: dict, out_dir: Path):
    """绘制相似度分布直方图"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择 r=0.2 作为示例
    target_r = 0.2
    
    sim_names = sorted(set(k[0] for k in distributions.keys()))
    corrupt_types = ['fine', 'random', 'coarse']
    
    for sim_name in sim_names:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, corrupt_type in enumerate(corrupt_types):
            key = (sim_name, target_r, corrupt_type)
            ax = axes[idx]
            
            if key in distributions and len(distributions[key]) > 0:
                sims = distributions[key]
                ax.hist(sims, bins=50, density=True, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(sims), color='red', linestyle='--', 
                          label=f'Mean={np.mean(sims):.3f}')
                ax.axvline(np.percentile(sims, 10), color='blue', linestyle=':',
                          label=f'P10={np.percentile(sims, 10):.3f}')
                ax.axvline(np.percentile(sims, 90), color='green', linestyle=':',
                          label=f'P90={np.percentile(sims, 90):.3f}')
            
            ax.set_xlabel('Similarity $S_{ij}$', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{corrupt_type.capitalize()} Corruption', fontsize=12)
            ax.legend(fontsize=9)
            ax.set_xlim([0, 1])
        
        fig.suptitle(f'{sim_name.upper()} - Similarity Distribution (r={target_r})', fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / f'similarity_histogram_{sim_name}.png', dpi=150)
        plt.savefig(out_dir / f'similarity_histogram_{sim_name}.pdf')
        plt.close()


def find_matching_pred(pred_dir: Path, gt_name: str) -> Optional[Path]:
    """
    尝试多种命名模式找到匹配的预测文件
    优先匹配 _pred_mask.png（真正的预测 mask）
    """
    patterns = [
        f"{gt_name}_pred_mask.png",   # ← 最优先
        f"{gt_name}.png",
        f"{gt_name}_pred_mask.png",
        f"{gt_name}_seg.png",
        f"{gt_name.replace('_val_', '_')}.png",
    ]
    
    for pattern in patterns:
        pred_path = pred_dir / pattern
        if pred_path.exists():
            return pred_path
    
    # 模糊匹配时排除可视化文件
    if '_' in gt_name:
        core_id = gt_name.split('_')[-1]
        for f in sorted(pred_dir.glob(f"*{core_id}*.png")):
            if '_labeled' not in f.name and '_overlay' not in f.name:
                return f
    
    return None



# ============================================================================
# 实验 2：真实预测验证——多指标排名
# ============================================================================

def run_real_ranking_experiment(args) -> dict:
    """
    实验 2：真实预测验证——多指标排名
    
    对每个 pred_dir 评测多个指标，做 bootstrap 获取 CI
    """
    print("=" * 70)
    print("实验 2：真实预测验证——多指标排名")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相似度矩阵
    matrix_paths = {
        'hybrid': args.hybrid_path,
        'pathsim': args.pathsim_path,
        'wup': args.wup_path
    }
    matrices = load_similarity_matrices(matrix_paths)
    
    # Open-mIoU 矩阵（如果提供）
    if args.open_miou_path and Path(args.open_miou_path).exists():
        S_open = np.load(args.open_miou_path).astype(np.float64)
        matrices['open'] = S_open
        print(f"  open: shape={S_open.shape}")
    
    if not matrices:
        return {'error': 'No matrices loaded'}
    
    # 加载 GT
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"\n找到 {len(image_pairs)} 张 GT 图片")
    
    # 获取预测目录和名称
    pred_dirs = [Path(p) for p in args.pred_dirs]
    pred_names = args.pred_names if args.pred_names else [p.name for p in pred_dirs]
    
    print(f"\n预测目录 ({len(pred_dirs)} 个):")
    for name, pd in zip(pred_names, pred_dirs):
        print(f"  - {name}: {pd}")
        # 显示预测目录中的文件示例
        if pd.exists():
            pred_files = list(pd.glob("*.png"))[:3]
            if pred_files:
                print(f"    示例文件: {[f.name for f in pred_files]}")
    
    # 显示 GT 文件示例
    if image_pairs:
        print(f"\nGT 文件示例: {[p[2] for p in image_pairs[:3]]}")
    
    num_classes = args.num_classes
    n_bootstrap = args.bootstrap
    
    all_ranking_results = []
    all_bootstrap_results = {}
    
    for pred_name, pred_dir in zip(pred_names, pred_dirs):
        print(f"\n{'='*50}")
        print(f"处理: {pred_name}")
        print(f"{'='*50}")
        
        # 构建配对（使用智能匹配）
        pairs = []
        unmatched = []
        for gt_path, _, name in image_pairs:
            # 尝试多种匹配方式
            pred_path = find_matching_pred(pred_dir, name)
            if pred_path is not None:
                pairs.append((gt_path, pred_path, name))
            else:
                unmatched.append(name)
        
        if not pairs:
            print(f"  警告: 未找到匹配的预测文件")
            if unmatched:
                print(f"  未匹配的 GT 文件示例: {unmatched[:5]}")
            # 尝试直接列出预测目录
            pred_files = list(pred_dir.glob("*.png"))
            if pred_files:
                print(f"  预测目录中的文件示例: {[f.name for f in pred_files[:5]]}")
            continue
        
        print(f"  找到 {len(pairs)} 对匹配文件")
        if unmatched:
            print(f"  未匹配: {len(unmatched)} 个")
        
        # 收集每张图的指标
        per_image_metrics = defaultdict(list)
        
        for gt_path, pred_path, name in tqdm(pairs, desc="  Computing metrics"):
            gt = load_mask(gt_path, offset=args.gt_offset)
            pred = load_mask(pred_path)
            
            confusion = compute_confusion_matrix(gt, pred, num_classes, args.ignore_label)
            
            # mIoU
            miou_res = compute_miou_from_confusion(confusion)
            per_image_metrics['mIoU'].append(miou_res['miou'])
            
            # hierIoU（各矩阵）
            for sim_name, S in matrices.items():
                hier_res = compute_hier_soft_iou_from_confusion(confusion, S)
                per_image_metrics[f'hierIoU_{sim_name}'].append(hier_res['hier_miou'])
        
        # Bootstrap 计算 CI
        model_result = {'model': pred_name}
        model_bootstrap = {}
        
        print(f"  Bootstrap ({n_bootstrap} 次)...")
        for metric_name, values in per_image_metrics.items():
            values = np.array(values)
            boot_result = bootstrap_single_metric(values, n_bootstrap=n_bootstrap, seed=args.seed)
            
            model_result[metric_name] = boot_result['mean']
            model_result[f'{metric_name}_ci_low'] = boot_result['ci_low']
            model_result[f'{metric_name}_ci_high'] = boot_result['ci_high']
            model_result[f'{metric_name}_std'] = boot_result['std']
            
            model_bootstrap[metric_name] = boot_result
            
            print(f"    {metric_name}: {boot_result['mean']:.4f} "
                  f"[{boot_result['ci_low']:.4f}, {boot_result['ci_high']:.4f}]")
        
        all_ranking_results.append(model_result)
        all_bootstrap_results[pred_name] = model_bootstrap
    
    # 排名
    print("\n" + "=" * 60)
    print("排名结果")
    print("=" * 60)
    
    metrics = ['mIoU'] + [f'hierIoU_{s}' for s in matrices.keys()]
    
    for metric in metrics:
        print(f"\n{metric}:")
        ranked = sorted(all_ranking_results, key=lambda x: x.get(metric, 0), reverse=True)
        for rank, r in enumerate(ranked, 1):
            val = r.get(metric, 0)
            ci_low = r.get(f'{metric}_ci_low', val)
            ci_high = r.get(f'{metric}_ci_high', val)
            print(f"  {rank}. {r['model']}: {val:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    # 保存结果
    print("\n保存结果...")
    
    save_csv(all_ranking_results, out_dir / "tables" / "real_ranking_with_ci.csv")
    save_json(all_bootstrap_results, out_dir / "logs" / "bootstrap_results.json")
    
    # LaTeX
    generate_ranking_latex_table(all_ranking_results, metrics, 
                                out_dir / "tables" / "real_ranking_with_ci.tex")
    
    # 绘图
    plot_ranking_bars(all_ranking_results, metrics, out_dir / "plots")
    
    return {
        'ranking': all_ranking_results,
        'bootstrap': all_bootstrap_results
    }


def generate_ranking_latex_table(results: List[dict], metrics: List[str], out_path: Path):
    """生成排名 LaTeX 表格"""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-Metric Ranking on Real Predictions with 95\% CI}",
        r"\label{tab:real_ranking}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Model & " + " & ".join([m.replace('_', r'\_') for m in metrics]) + r" \\",
        r"\midrule",
    ]
    
    for r in results:
        row = r['model'].replace('_', r'\_')
        for m in metrics:
            val = r.get(m, 0)
            ci_low = r.get(f'{m}_ci_low', val)
            ci_high = r.get(f'{m}_ci_high', val)
            # 简化显示：只显示 mean ± CI width
            ci_width = (ci_high - ci_low) / 2
            row += f" & {val:.3f}$\\pm${ci_width:.3f}"
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    generate_latex_table(lines, out_path)


def plot_ranking_bars(results: List[dict], metrics: List[str], out_dir: Path):
    """绘制排名柱状图"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    models = [r['model'] for r in results]
    n_models = len(models)
    
    # 为每个指标单独绘图
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 6))
        
        x = np.arange(n_models)
        values = [r.get(metric, 0) for r in results]
        ci_lows = [r.get(f'{metric}_ci_low', r.get(metric, 0)) for r in results]
        ci_highs = [r.get(f'{metric}_ci_high', r.get(metric, 0)) for r in results]
        
        yerr_low = [v - ci_l for v, ci_l in zip(values, ci_lows)]
        yerr_high = [ci_h - v for v, ci_h in zip(values, ci_highs)]
        
        bars = ax.bar(x, values, yerr=[yerr_low, yerr_high], 
                     capsize=5, alpha=0.8, color='steelblue')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} with 95% CI', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        metric_clean = metric.replace('/', '_')
        plt.savefig(out_dir / f'ranking_bars_{metric_clean}.png', dpi=150)
        plt.savefig(out_dir / f'ranking_bars_{metric_clean}.pdf')
        plt.close()
    
    # 综合对比图
    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 6))
    
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    x = np.arange(n_models)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results]
        ci_lows = [r.get(f'{metric}_ci_low', r.get(metric, 0)) for r in results]
        ci_highs = [r.get(f'{metric}_ci_high', r.get(metric, 0)) for r in results]
        
        yerr_low = [v - ci_l for v, ci_l in zip(values, ci_lows)]
        yerr_high = [ci_h - v for v, ci_h in zip(values, ci_highs)]
        
        ax.bar(x + i * width, values, width, 
               label=metric, color=colors[i], alpha=0.8,
               yerr=[yerr_low, yerr_high], capsize=3)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Multi-Metric Comparison', fontsize=14)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'multi_metric_comparison.png', dpi=150)
    plt.savefig(out_dir / 'multi_metric_comparison.pdf')
    plt.close()
    
    print(f"  排名图表已保存到: {out_dir}")


# ============================================================================
# 实验 3：Near/Mid/Far 错误分解
# ============================================================================

def run_error_stratification_experiment(args) -> dict:
    """
    实验 3：Near/Mid/Far 错误分解
    
    对每个 pred_dir，按统一的全局相似度阈值分桶统计错误
    """
    print("=" * 70)
    print("实验 3：Near/Mid/Far 错误分解")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相似度矩阵（主要用 hybrid）
    matrix_paths = {
        'hybrid': args.hybrid_path,
        'pathsim': args.pathsim_path,
        'wup': args.wup_path
    }
    matrices = load_similarity_matrices(matrix_paths)
    
    # 使用主矩阵
    primary_name = 'hybrid' if 'hybrid' in matrices else list(matrices.keys())[0]
    S = matrices[primary_name]
    print(f"\n使用 {primary_name} 进行错误分层分析")
    
    # 加载 GT
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"找到 {len(image_pairs)} 张 GT 图片")
    
    # 获取预测目录和名称
    pred_dirs = [Path(p) for p in args.pred_dirs]
    pred_names = args.pred_names if args.pred_names else [p.name for p in pred_dirs]
    
    num_classes = args.num_classes
    quantile = args.far_quantile
    
    if not (0 < quantile < 0.5):
        raise ValueError(f"--far_quantile 必须在 (0, 0.5) 之间，当前为 {quantile}")

    matched_pairs_by_model = {}
    error_sims_by_model = {}
    confusion_pairs_by_model = {}
    top_far_confusions = {}
    
    for pred_name, pred_dir in zip(pred_names, pred_dirs):
        print(f"\n{'='*50}")
        print(f"处理: {pred_name}")
        print(f"{'='*50}")
        
        # 构建配对（使用智能匹配）
        pairs = []
        for gt_path, _, name in image_pairs:
            pred_path = find_matching_pred(pred_dir, name)
            if pred_path is not None:
                pairs.append((gt_path, pred_path, name))
        
        if not pairs:
            print(f"  警告: 未找到匹配的预测文件")
            continue
        
        print(f"  找到 {len(pairs)} 对匹配文件")

        all_error_sims, confusion_pairs = collect_error_similarities(
            pairs=pairs,
            S=S,
            num_classes=num_classes,
            ignore_label=args.ignore_label,
            gt_offset=args.gt_offset,
        )

        if all_error_sims.size == 0:
            print(f"  无错误像素")
            continue

        matched_pairs_by_model[pred_name] = pairs
        error_sims_by_model[pred_name] = all_error_sims
        confusion_pairs_by_model[pred_name] = confusion_pairs

        print(f"  总错误像素: {len(all_error_sims):,}")
        print(f"  错误像素平均相似度: {all_error_sims.mean():.4f}")

    if not error_sims_by_model:
        print("未收集到任何有效错误像素，跳过保存。")
        return {'error': 'No valid error pixels collected'}

    # 使用跨模型共享的全局阈值，避免每个模型都被强行切成近似固定比例
    global_error_sims = np.concatenate(list(error_sims_by_model.values()))
    far_threshold = float(np.percentile(global_error_sims, quantile * 100))
    near_threshold = float(np.percentile(global_error_sims, (1 - quantile) * 100))

    print("\n" + "-" * 60)
    print("使用全局阈值进行错误分层")
    print(f"  far_quantile={quantile:.3f}")
    print(f"  全局 Far 阈值 : S <= {far_threshold:.4f}")
    print(f"  全局 Near 阈值: S >= {near_threshold:.4f}")
    print("-" * 60)

    stratification_results = []
    for pred_name in pred_names:
        if pred_name not in error_sims_by_model:
            continue

        all_error_sims = error_sims_by_model[pred_name]
        confusion_pairs = confusion_pairs_by_model[pred_name]
        total_errors = len(all_error_sims)

        far_mask = all_error_sims <= far_threshold
        mid_mask = (all_error_sims > far_threshold) & (all_error_sims < near_threshold)
        near_mask = all_error_sims >= near_threshold

        far_count = int(far_mask.sum())
        mid_count = int(mid_mask.sum())
        near_count = int(near_mask.sum())

        result = {
            'model': pred_name,
            'total_errors': total_errors,
            'far_count': far_count,
            'mid_count': mid_count,
            'near_count': near_count,
            'far_ratio': float(far_count / total_errors),
            'mid_ratio': float(mid_count / total_errors),
            'near_ratio': float(near_count / total_errors),
            'far_threshold': far_threshold,
            'near_threshold': near_threshold,
            'threshold_mode': 'global_quantile',
            'quantile': float(quantile),
            'mean_sim': float(all_error_sims.mean()),
            'std_sim': float(all_error_sims.std()),
            'P10': float(np.percentile(all_error_sims, 10)),
            'P25': float(np.percentile(all_error_sims, 25)),
            'P50': float(np.percentile(all_error_sims, 50)),
            'P75': float(np.percentile(all_error_sims, 75)),
            'P90': float(np.percentile(all_error_sims, 90))
        }
        stratification_results.append(result)

        print(f"\n{pred_name}:")
        print(f"  Far  (S <= {far_threshold:.4f}): {far_count:,} ({result['far_ratio']:.1%})")
        print(f"  Mid  ({far_threshold:.4f} < S < {near_threshold:.4f}): "
              f"{mid_count:,} ({result['mid_ratio']:.1%})")
        print(f"  Near (S >= {near_threshold:.4f}): {near_count:,} ({result['near_ratio']:.1%})")
        print(f"  平均相似度: {all_error_sims.mean():.4f}")

        far_confusions = [
            (k, v) for k, v in confusion_pairs.items()
            if S[k[0], k[1]] <= far_threshold
        ]
        far_confusions.sort(key=lambda x: x[1], reverse=True)

        top_far = [
            {
                'gt_class': int(k[0]),
                'pred_class': int(k[1]),
                'count': v,
                'similarity': float(S[k[0], k[1]])
            }
            for k, v in far_confusions[:10]
        ]
        top_far_confusions[pred_name] = top_far

        if top_far:
            print(f"  Top-10 Far Confusions (global S <= {far_threshold:.4f}):")
            for i, item in enumerate(top_far[:5], 1):
                print(f"    {i}. {item['gt_class']} → {item['pred_class']}: "
                      f"{item['count']:,} pixels (S={item['similarity']:.4f})")
    
    # 保存结果
    print("\n保存结果...")
    
    save_csv(stratification_results, out_dir / "tables" / f"error_stratification_{primary_name}.csv")
    save_json(top_far_confusions, out_dir / "logs" / "top_far_confusions.json")
    save_json(
        {
            'threshold_mode': 'global_quantile',
            'far_quantile': float(quantile),
            'far_threshold': far_threshold,
            'near_threshold': near_threshold,
            'num_models': len(stratification_results),
            'total_error_pixels': int(sum(len(v) for v in error_sims_by_model.values()))
        },
        out_dir / "logs" / "global_thresholds.json"
    )
    
    # LaTeX
    generate_stratification_latex_table(stratification_results, 
                                        out_dir / "tables" / f"error_stratification_{primary_name}.tex")
    
    # 绘图
    plot_error_stratification_stacked(
        stratification_results,
        out_dir / "plots",
        far_threshold=far_threshold,
        near_threshold=near_threshold,
        quantile=quantile,
    )
    plot_error_similarity_distributions(
        error_sims_by_model,
        out_dir / "plots",
        far_threshold=far_threshold,
        near_threshold=near_threshold,
        quantile=quantile,
    )
    
    return {
        'stratification': stratification_results,
        'top_far_confusions': top_far_confusions,
        'global_thresholds': {
            'far_quantile': quantile,
            'far_threshold': far_threshold,
            'near_threshold': near_threshold,
        }
    }


def generate_stratification_latex_table(results: List[dict], out_path: Path):
    """生成分层 LaTeX 表格"""
    quantile = results[0].get('quantile', None) if results else None
    if quantile is not None:
        far_pct = int(round(quantile * 100))
        caption = (
            f"Error Stratification by Semantic Similarity "
            f"(global thresholds: bottom {far_pct}\\% vs. top {far_pct}\\%)"
        )
    else:
        caption = "Error Stratification by Semantic Similarity"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:error_strat}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Far (\%) & Mid (\%) & Near (\%) & Mean Sim. & P50 \\",
        r"\midrule",
    ]
    
    for r in results:
        model_name = r['model'].replace('_', r'\_')
        row = f"{model_name} & {r['far_ratio']*100:.1f} & {r['mid_ratio']*100:.1f} "
        row += f"& {r['near_ratio']*100:.1f} & {r['mean_sim']:.3f} & {r['P50']:.3f}"
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    generate_latex_table(lines, out_path)


def plot_error_stratification_stacked(results: List[dict], out_dir: Path,
                                      far_threshold: float,
                                      near_threshold: float,
                                      quantile: float):
    """绘制错误分层堆叠柱状图"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    models = [r['model'] for r in results]
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 6))
    
    x = np.arange(n_models)
    
    far = [r['far_ratio'] for r in results]
    mid = [r['mid_ratio'] for r in results]
    near = [r['near_ratio'] for r in results]
    
    ax.bar(x, far, label=f'Far (S $\\leq$ {far_threshold:.3f})', color='#e74c3c', alpha=0.8)
    ax.bar(x, mid, bottom=far,
           label=f'Mid ({far_threshold:.3f} < S < {near_threshold:.3f})',
           color='#f39c12', alpha=0.8)
    ax.bar(x, near, bottom=[f+m for f, m in zip(far, mid)], 
           label=f'Near (S $\\geq$ {near_threshold:.3f})', color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error Ratio', fontsize=12)
    ax.set_title(f'Error Stratification by Semantic Similarity (global q={quantile:.2f})',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    for i, (f, m, n) in enumerate(zip(far, mid, near)):
        ax.text(i, f/2, f'{f*100:.0f}%', ha='center', va='center', fontsize=9, color='white')
        ax.text(i, f + m/2, f'{m*100:.0f}%', ha='center', va='center', fontsize=9)
        ax.text(i, f + m + n/2, f'{n*100:.0f}%', ha='center', va='center', fontsize=9, color='white')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'error_stratification_stacked.png', dpi=150)
    plt.savefig(out_dir / 'error_stratification_stacked.pdf')
    plt.close()
    
    print(f"  分层图表已保存到: {out_dir}")


def plot_error_similarity_distributions(error_sims_by_model: Dict[str, np.ndarray],
                                        out_dir: Path,
                                        far_threshold: float,
                                        near_threshold: float,
                                        quantile: float):
    """绘制各模型的错误相似度分布"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_names = list(error_sims_by_model.keys())
    
    if len(pred_names) <= 3:
        n_cols = len(pred_names)
        n_rows = 1
    else:
        n_cols = min(3, len(pred_names))
        n_rows = (len(pred_names) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, pred_name in enumerate(pred_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        error_sims = error_sims_by_model[pred_name]

        if error_sims.size > 0:
            ax.hist(error_sims, bins=50, density=True, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(error_sims), color='red', linestyle='--',
                      label=f'Mean={np.mean(error_sims):.3f}')
            ax.axvline(far_threshold, color='blue', linestyle=':',
                      label=f'Far={far_threshold:.3f}')
            ax.axvline(near_threshold, color='green', linestyle=':',
                      label=f'Near={near_threshold:.3f}')
            ax.legend(fontsize=8)
        
        ax.set_xlabel('Similarity $S_{ij}$', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(pred_name, fontsize=11)
        ax.set_xlim([0, 1])
    
    # 隐藏多余的子图
    for idx in range(len(pred_names), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'Error Similarity Distribution (global q={quantile:.2f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'error_similarity_distributions.png', dpi=150)
    plt.savefig(out_dir / 'error_similarity_distributions.pdf')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="指标验证实验套件 V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行命令示例:

  # 1. Far-credit 测试（synth preds）
  python run_validations.py --mode far_credit \\
      --gt_dir /path/to/ade150_val_gt \\
      --synth_pred_root output/end2end_comparison/synth_preds \\
      --hybrid_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pathsim_path similarity_output/ade150/S.npy \\
      --wup_path similarity_output/ade150_wup/S.npy \\
      --out_dir output/far_credit_test \\
      --ratios 0.05 0.10 0.20 0.30

  # 2. 真实预测排名（多个 pred_dirs）
  python run_validations.py --mode real_ranking \\
      --gt_dir /path/to/ade150_val_gt \\
      --pred_dirs /path/to/model1_preds /path/to/model2_preds \\
      --pred_names "Model-A" "Model-B" \\
      --hybrid_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pathsim_path similarity_output/ade150/S.npy \\
      --wup_path similarity_output/ade150_wup/S.npy \\
      --out_dir output/real_ranking \\
      --bootstrap 1000

  # 3. Near/Mid/Far 错误分解
  python run_validations.py --mode error_stratification \\
      --gt_dir /path/to/ade150_val_gt \\
      --pred_dirs /path/to/model1_preds /path/to/model2_preds \\
      --pred_names "Model-A" "Model-B" \\
      --hybrid_path similarity_output/ade150_hybrid_inv/S.npy \\
      --out_dir output/error_stratification
        """
    )
    
    # 模式
    parser.add_argument("--mode", type=str, required=True,
                        choices=['far_credit', 'real_ranking', 'error_stratification', 'all'],
                        help="运行模式")
    
    # 输入路径
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="GT masks 目录")
    parser.add_argument("--synth_pred_root", type=Path, default=None,
                        help="合成预测根目录（far_credit 模式需要）")
    parser.add_argument("--pred_dirs", type=Path, nargs='*', default=[],
                        help="真实预测目录列表")
    parser.add_argument("--pred_names", type=str, nargs='*', default=[],
                        help="预测目录对应的显示名称")
    
    # 相似度矩阵路径
    parser.add_argument("--hybrid_path", type=Path, default=None)
    parser.add_argument("--pathsim_path", type=Path, default=None)
    parser.add_argument("--wup_path", type=Path, default=None)
    parser.add_argument("--open_miou_path", type=Path, default=None,
                        help="Open-mIoU 矩阵路径（可选）")
    
    # 输出
    parser.add_argument("--out_dir", type=Path, default=Path("./output"),
                        help="输出目录")
    
    # 参数
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--gt_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs='+',
                        default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--far_quantile", type=float, default=0.1,
                        help="Far/Near 的全局共享分位数阈值 (默认 0.1 = bottom/top 10%%)")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap 采样次数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建输出目录
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存元数据
    save_json(get_experiment_meta(args), args.out_dir / "logs" / "experiment_meta.json")
    
    # 运行实验
    if args.mode == 'far_credit':
        if args.synth_pred_root is None:
            print("错误: far_credit 模式需要 --synth_pred_root")
            return 1
        run_far_credit_experiment(args)
    
    elif args.mode == 'real_ranking':
        if not args.pred_dirs:
            print("错误: real_ranking 模式需要 --pred_dirs")
            return 1
        run_real_ranking_experiment(args)
    
    elif args.mode == 'error_stratification':
        if not args.pred_dirs:
            print("错误: error_stratification 模式需要 --pred_dirs")
            return 1
        run_error_stratification_experiment(args)
    
    elif args.mode == 'all':
        if args.synth_pred_root:
            run_far_credit_experiment(args)
        if args.pred_dirs:
            run_real_ranking_experiment(args)
            run_error_stratification_experiment(args)
    
    print("\n" + "=" * 70)
    print(f"实验完成，结果保存到: {args.out_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
