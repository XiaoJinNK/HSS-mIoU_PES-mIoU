#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标有效性验证实验套件 V2

包含：
- A. 公平版多矩阵对比（End-to-End Targets）
- B. 防洗白验证（Far-credit Test）
- C. 真实预测验证（Ranking & Error Stratification）
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
    compute_hier_soft_iou_from_confusion,
    evaluate_dataset
)
from corrupt import (
    compute_corruption_targets,
    generate_synthetic_predictions,
    load_mask,
    save_mask,
    sample_replace_map_per_class,
    apply_replace_map,
    _stable_int_seed
)
from bootstrap import bootstrap_comparison, bootstrap_metric_difference
from io_utils import (
    load_Hij, find_image_pairs, save_json, save_csv, save_latex_table,
    get_experiment_meta
)


# ============================================================================
# A. 公平版多矩阵对比（End-to-End Targets）
# ============================================================================

def run_end2end_matrix_comparison(args) -> dict:
    """
    A. 公平版多矩阵对比
    
    为每个相似度矩阵 S 独立执行一套 corruption：
    - 用该 S 自己来选 fine_targets 和 coarse_targets
    - 使用同一套 corruption 协议（同位置替换）
    - 用同一个 S 评测
    """
    print("=" * 70)
    print("A. End-to-End Matrix Comparison")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有相似度矩阵
    matrices = load_similarity_matrices(args)
    if not matrices:
        print("错误: 未加载任何相似度矩阵")
        return {'error': 'No matrices loaded'}
    
    # 分析矩阵特性
    analyze_matrices(matrices)
    
    # 加载图片
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"\n找到 {len(image_pairs)} 张图片")
    
    if not image_pairs:
        return {'error': 'No images found'}
    
    num_classes = args.num_classes
    ratios = args.ratios
    
    all_results = []
    per_matrix_results = {}
    
    for sim_name, S in matrices.items():
        print(f"\n{'='*60}")
        print(f"处理矩阵: {sim_name.upper()}")
        print(f"{'='*60}")
        
        # 1. 用该矩阵计算 targets
        print(f"\n[1] 使用 {sim_name} 计算替换目标...")
        targets = compute_corruption_targets(S, k=args.target_k, seed=args.seed)
        
        # 2. 生成合成预测（每个矩阵独立目录）
        synth_dir = out_dir / "synth_preds" / sim_name
        
        if not args.skip_generate:
            print(f"\n[2] 生成合成预测...")
            generate_synthetic_predictions_controlled(
                image_pairs, targets, ratios, synth_dir,
                num_classes, args.ignore_label, args.gt_offset,
                seed=args.seed
            )
        else:
            print(f"\n[2] 跳过生成，使用已有预测...")
        
        # 3. 用该矩阵评测
        print(f"\n[3] 使用 {sim_name} 评测...")
        matrix_results = evaluate_corruption_results(
            image_pairs, synth_dir, S, ratios,
            num_classes, args.ignore_label, args.gt_offset
        )
        
        per_matrix_results[sim_name] = matrix_results
        
        # 汇总结果
        for res in matrix_results:
            res['similarity'] = sim_name
            all_results.append(res)
    
    # 4. 计算区分度得分
    print("\n" + "=" * 70)
    print("区分度得分汇总")
    print("=" * 70)
    
    scores = {}
    for sim_name in matrices.keys():
        scores[sim_name] = compute_discrimination_metrics(
            [r for r in all_results if r['similarity'] == sim_name]
        )
        print(f"\n{sim_name.upper()}:")
        print(f"  Mean Delta Gap: {scores[sim_name]['mean_delta_gap']:.4f}")
        print(f"  Ordering Accuracy: {scores[sim_name]['ordering_accuracy']:.0%}")
        for r in ratios:
            gap = scores[sim_name]['delta_gaps_by_r'].get(r, 0)
            print(f"  r={r}: ΔGap = {gap:.4f}")
    
    # 5. 保存结果
    print("\n[4] 保存结果...")
    
    # CSV
    save_csv(all_results, out_dir / "tables" / "end2end_matrix_comparison.csv")
    
    # LaTeX
    generate_end2end_latex_table(all_results, scores, out_dir / "tables" / "end2end_matrix_comparison.tex")
    
    # JSON
    save_json({
        'results': all_results,
        'scores': scores
    }, out_dir / "logs" / "end2end_results.json")
    
    # 6. 绘图
    print("\n[5] 绘制图表...")
    plot_end2end_comparison(all_results, scores, out_dir / "plots")
    
    return {
        'results': all_results,
        'scores': scores
    }


def generate_synthetic_predictions_controlled(
    image_pairs: List[Tuple],
    targets: Dict[str, np.ndarray],
    ratios: List[float],
    output_dir: Path,
    num_classes: int,
    ignore_label: int = 255,
    gt_offset: int = 0,
    seed: int = 42
):
    """
    生成合成预测（确保同一张图同一 r 内复用相同替换位置）
    """
    total_ops = len(image_pairs) * len(ratios) * 3
    pbar = tqdm(total=total_ops, desc="Generating predictions")
    
    # 预建目录
    for corrupt_type in ["fine", "coarse", "random"]:
        for r in ratios:
            r_str = f"r{int(r*100):02d}"
            (output_dir / corrupt_type / r_str).mkdir(parents=True, exist_ok=True)
    
    for r in ratios:
        r_str = f"r{int(r*100):02d}"
        
        for gt_path, _, name in image_pairs:
            gt = load_mask(gt_path, offset=gt_offset, ignore_label=ignore_label)
            
            # 同一张图同一 r 使用相同的替换位置
            img_seed = _stable_int_seed(seed, name, r)
            replace_map = sample_replace_map_per_class(
                gt, r, num_classes, ignore_label, seed=img_seed
            )
            
            # 三种 corruption 类型
            for corrupt_type in ["fine", "coarse", "random"]:
                target_arr = targets[corrupt_type]
                pred = apply_replace_map(gt, replace_map, target_arr)
                
                out_path = output_dir / corrupt_type / r_str / f"{name}.png"
                save_mask(pred, out_path)
                pbar.update(1)
    
    pbar.close()


def evaluate_corruption_results(
    image_pairs: List[Tuple],
    synth_dir: Path,
    S: np.ndarray,
    ratios: List[float],
    num_classes: int,
    ignore_label: int = 255,
    gt_offset: int = 0
) -> List[dict]:
    """评测所有 corruption 结果"""
    results = []
    
    for r in ratios:
        r_str = f"r{int(r*100):02d}"
        print(f"\n  r={r}:")
        
        for corrupt_type in ['fine', 'coarse', 'random']:
            pred_dir = synth_dir / corrupt_type / r_str
            
            # 累计混淆矩阵
            confusion_total = np.zeros((num_classes, num_classes), dtype=np.int64)
            per_image_results = []
            
            for gt_path, _, name in image_pairs:
                gt = load_mask(gt_path, offset=gt_offset)
                pred = load_mask(pred_dir / f"{name}.png")
                
                confusion = compute_confusion_matrix(gt, pred, num_classes, ignore_label)
                confusion_total += confusion
                
                # 单图指标
                miou_res = compute_miou_from_confusion(confusion)
                hier_res = compute_hier_soft_iou_from_confusion(confusion, S)
                per_image_results.append({
                    'name': name,
                    'miou': miou_res['miou'],
                    'hier_miou': hier_res['hier_miou']
                })
            
            # 全局指标（dataset-level）
            miou_result = compute_miou_from_confusion(confusion_total)
            hier_result = compute_hier_soft_iou_from_confusion(confusion_total, S)
            
            # image-level mean
            img_miou = np.mean([r['miou'] for r in per_image_results])
            img_hier_miou = np.mean([r['hier_miou'] for r in per_image_results])
            
            delta = img_hier_miou - img_miou
            
            results.append({
                'r': r,
                'type': corrupt_type,
                'mIoU': img_miou,
                'hierIoU': img_hier_miou,
                'delta': delta,
                'mIoU_dataset': miou_result['miou'],
                'hierIoU_dataset': hier_result['hier_miou'],
                'per_image': per_image_results
            })
            
            print(f"    {corrupt_type}: mIoU={img_miou:.4f}, hierIoU={img_hier_miou:.4f}, Δ={delta:.4f}")
    
    return results


def compute_discrimination_metrics(results: List[dict]) -> dict:
    """计算区分度指标"""
    ratios = sorted(set(r['r'] for r in results))
    
    delta_gaps = []
    ordering_correct = 0
    
    for r in ratios:
        r_results = {res['type']: res['delta'] for res in results if res['r'] == r}
        
        delta_fine = r_results.get('fine', 0)
        delta_coarse = r_results.get('coarse', 0)
        delta_random = r_results.get('random', 0)
        
        gap = delta_fine - delta_coarse
        delta_gaps.append((r, gap))
        
        if delta_fine > delta_random > delta_coarse:
            ordering_correct += 1
    
    return {
        'mean_delta_gap': np.mean([g for _, g in delta_gaps]),
        'ordering_accuracy': ordering_correct / len(ratios) if ratios else 0,
        'delta_gaps_by_r': {r: g for r, g in delta_gaps}
    }


def plot_end2end_comparison(results: List[dict], scores: dict, out_dir: Path):
    """绘制 End-to-End 对比图"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    colors = {'hybrid': '#2196F3', 'pathsim': '#4CAF50', 'wup': '#FF9800'}
    markers = {'hybrid': 'o', 'pathsim': 's', 'wup': '^'}
    
    # 图1: 每个矩阵的 Delta 曲线
    for sim_name in sim_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for corrupt_type, style in [('fine', '-'), ('random', '--'), ('coarse', ':')]:
            data = [r for r in results 
                   if r['similarity'] == sim_name and r['type'] == corrupt_type]
            data.sort(key=lambda x: x['r'])
            
            x = [d['r'] for d in data]
            y = [d['delta'] for d in data]
            
            ax.plot(x, y, style + 'o', label=corrupt_type.capitalize(), 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
        ax.set_ylabel('Δ (Hier-mIoU - mIoU)', fontsize=12)
        ax.set_title(f'{sim_name.upper()} - Delta Curves', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(out_dir / f'delta_curves_{sim_name}.png', dpi=150)
        plt.savefig(out_dir / f'delta_curves_{sim_name}.pdf')
        plt.close()
    
    # 图2: Mean Delta Gap 柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(sim_names))
    gaps = [scores[s]['mean_delta_gap'] for s in sim_names]
    
    bars = ax.bar(x, gaps, color=[colors.get(s, 'gray') for s in sim_names], alpha=0.8)
    
    ax.set_xlabel('Similarity Matrix', fontsize=12)
    ax.set_ylabel('Mean Delta Gap (Δfine - Δcoarse)', fontsize=12)
    ax.set_title('Discrimination Ability Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in sim_names])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{gap:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'mean_delta_gap_comparison.png', dpi=150)
    plt.savefig(out_dir / 'mean_delta_gap_comparison.pdf')
    plt.close()
    
    # 图3: Delta Gap by r
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    x = np.arange(len(ratios))
    
    for i, sim_name in enumerate(sim_names):
        gaps = [scores[sim_name]['delta_gaps_by_r'].get(r, 0) for r in ratios]
        ax.bar(x + i * width, gaps, width, 
               label=sim_name.upper(),
               color=colors.get(sim_name, 'gray'),
               alpha=0.8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Delta Gap (Δfine - Δcoarse)', fontsize=12)
    ax.set_title('Delta Gap by Corruption Ratio', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{r:.0%}' for r in ratios])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'delta_gap_by_ratio.png', dpi=150)
    plt.savefig(out_dir / 'delta_gap_by_ratio.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


def generate_end2end_latex_table(results: List[dict], scores: dict, out_path: Path):
    """生成 End-to-End LaTeX 表格"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{End-to-End Matrix Comparison: Delta Gap by Corruption Ratio}",
        r"\label{tab:end2end}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(ratios) + "cc}",
        r"\toprule",
        r"Method & " + " & ".join([f"$r={r:.0%}$" for r in ratios]) + r" & Mean $\Delta$Gap & Ordering \\",
        r"\midrule",
    ]
    
    for sim_name in sim_names:
        score = scores[sim_name]
        gaps = [score['delta_gaps_by_r'].get(r, 0) for r in ratios]
        
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


# ============================================================================
# B. 防洗白验证（Far-credit Test）
# ============================================================================

def run_far_credit_test(args) -> dict:
    """
    B. 防洗白验证
    
    计算 Near credit 和 Far credit，验证远错不被洗白
    """
    print("=" * 70)
    print("B. Far-Credit Test")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载矩阵
    matrices = load_similarity_matrices(args)
    if not matrices:
        return {'error': 'No matrices loaded'}
    
    # 加载图片
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"\n找到 {len(image_pairs)} 张图片")
    
    num_classes = args.num_classes
    ratios = args.ratios
    q = args.far_quantile  # 默认 0.1
    
    all_credit_results = []
    
    for sim_name, S in matrices.items():
        print(f"\n{'='*60}")
        print(f"处理矩阵: {sim_name.upper()}")
        print(f"{'='*60}")
        
        synth_dir = out_dir / "synth_preds" / sim_name
        
        for r in ratios:
            r_str = f"r{int(r*100):02d}"
            print(f"\n  r={r}:")
            
            # 只分析 fine corruption（最能体现洗白问题）
            for corrupt_type in ['fine', 'coarse', 'random']:
                pred_dir = synth_dir / corrupt_type / r_str
                
                if not pred_dir.exists():
                    print(f"    跳过 {corrupt_type}（目录不存在）")
                    continue
                
                # 收集所有替换像素的相似度
                all_sims = []
                
                for gt_path, _, name in image_pairs:
                    gt = load_mask(gt_path, offset=args.gt_offset)
                    pred = load_mask(pred_dir / f"{name}.png")
                    
                    # 找到被替换的像素
                    valid = (gt != args.ignore_label) & (gt >= 0) & (gt < num_classes)
                    changed = (gt != pred) & valid
                    
                    if changed.sum() == 0:
                        continue
                    
                    # 获取这些像素的 (gt_class, pred_class) 及相似度
                    gt_classes = gt[changed]
                    pred_classes = pred[changed]
                    
                    for g, p in zip(gt_classes, pred_classes):
                        if 0 <= g < num_classes and 0 <= p < num_classes:
                            all_sims.append(S[g, p])
                
                if not all_sims:
                    continue
                
                all_sims = np.array(all_sims)
                
                # 计算分位数阈值
                far_threshold = np.percentile(all_sims, q * 100)
                near_threshold = np.percentile(all_sims, (1 - q) * 100)
                
                # Far credit: bottom q%
                far_mask = all_sims <= far_threshold
                far_credit = all_sims[far_mask].mean() if far_mask.sum() > 0 else 0
                
                # Near credit: top q%
                near_mask = all_sims >= near_threshold
                near_credit = all_sims[near_mask].mean() if near_mask.sum() > 0 else 0
                
                result = {
                    'similarity': sim_name,
                    'r': r,
                    'type': corrupt_type,
                    'far_credit': float(far_credit),
                    'near_credit': float(near_credit),
                    'near_far_gap': float(near_credit - far_credit),
                    'mean_sim': float(all_sims.mean()),
                    'num_pixels': len(all_sims),
                    'far_threshold': float(far_threshold),
                    'near_threshold': float(near_threshold)
                }
                
                all_credit_results.append(result)
                
                print(f"    {corrupt_type}: Far={far_credit:.4f}, Near={near_credit:.4f}, "
                      f"Gap={near_credit-far_credit:.4f}")
    
    # 保存结果
    print("\n保存结果...")
    save_csv(all_credit_results, out_dir / "tables" / "far_credit_test.csv")
    save_json(all_credit_results, out_dir / "logs" / "far_credit_results.json")
    
    # 生成 LaTeX 表格
    generate_far_credit_latex(all_credit_results, out_dir / "tables" / "far_credit_test.tex")
    
    # 绘图
    plot_far_credit(all_credit_results, out_dir / "plots")
    
    return {'results': all_credit_results}


def plot_far_credit(results: List[dict], out_dir: Path):
    """绘制 Far-credit 图表"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    colors = {'hybrid': '#2196F3', 'pathsim': '#4CAF50', 'wup': '#FF9800'}
    
    # 图1: Near/Far credit 曲线（每个矩阵）
    for sim_name in sim_names:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 只看 fine corruption
        fine_data = [r for r in results 
                    if r['similarity'] == sim_name and r['type'] == 'fine']
        fine_data.sort(key=lambda x: x['r'])
        
        if fine_data:
            x = [d['r'] for d in fine_data]
            near = [d['near_credit'] for d in fine_data]
            far = [d['far_credit'] for d in fine_data]
            
            ax.plot(x, near, '-o', label='Near Credit (top 10%)', 
                   color='green', linewidth=2, markersize=8)
            ax.plot(x, far, '-s', label='Far Credit (bottom 10%)', 
                   color='red', linewidth=2, markersize=8)
            ax.fill_between(x, far, near, alpha=0.2, color='gray')
        
        ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
        ax.set_ylabel('Credit (Similarity)', fontsize=12)
        ax.set_title(f'{sim_name.upper()} - Near/Far Credit (Fine Corruption)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(out_dir / f'near_far_credit_{sim_name}.png', dpi=150)
        plt.savefig(out_dir / f'near_far_credit_{sim_name}.pdf')
        plt.close()
    
    # 图2: Near-Far Gap 对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    x = np.arange(len(ratios))
    
    for i, sim_name in enumerate(sim_names):
        gaps = []
        for r in ratios:
            fine_data = [res for res in results 
                        if res['similarity'] == sim_name and res['r'] == r and res['type'] == 'fine']
            if fine_data:
                gaps.append(fine_data[0]['near_far_gap'])
            else:
                gaps.append(0)
        
        ax.bar(x + i * width, gaps, width,
               label=sim_name.upper(),
               color=colors.get(sim_name, 'gray'),
               alpha=0.8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Near-Far Gap', fontsize=12)
    ax.set_title('Near-Far Credit Gap Comparison', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{r:.0%}' for r in ratios])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'near_far_gap_comparison.png', dpi=150)
    plt.savefig(out_dir / 'near_far_gap_comparison.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


def generate_far_credit_latex(results: List[dict], out_path: Path):
    """生成 Far-credit LaTeX 表格"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sim_names = sorted(set(r['similarity'] for r in results))
    ratios = sorted(set(r['r'] for r in results))
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Far-Credit Test: Near/Far Credit for Fine Corruption}",
        r"\label{tab:far_credit}",
        r"\small",
        r"\begin{tabular}{l" + "ccc" * len(ratios) + "}",
        r"\toprule",
    ]
    
    # 多级表头
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
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ============================================================================
# C. 真实预测验证（Ranking & Error Stratification）
# ============================================================================

def run_real_prediction_validation(args) -> dict:
    """
    C. 真实预测验证
    
    C1. 多指标排名与显著性
    C2. Near/Mid/Far 错误分解
    """
    print("=" * 70)
    print("C. Real Prediction Validation")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载矩阵
    matrices = load_similarity_matrices(args)
    if not matrices:
        return {'error': 'No matrices loaded'}
    
    # 加载图片
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"\n找到 {len(image_pairs)} 张 GT 图片")
    
    # 获取预测目录列表
    pred_dirs = args.pred_dirs if args.pred_dirs else []
    if not pred_dirs:
        print("警告: 未指定预测目录，跳过真实预测验证")
        return {'error': 'No prediction directories specified'}
    
    print(f"预测目录: {len(pred_dirs)} 个")
    for pd in pred_dirs:
        print(f"  - {pd}")
    
    num_classes = args.num_classes
    
    # C1: 多指标排名
    print("\n" + "=" * 60)
    print("C1. Multi-Metric Ranking")
    print("=" * 60)
    
    ranking_results = []
    bootstrap_results = {}
    
    for pred_dir in pred_dirs:
        pred_dir = Path(pred_dir)
        model_name = pred_dir.name
        print(f"\n处理: {model_name}")
        
        # 构建配对
        pairs = []
        for gt_path, _, name in image_pairs:
            pred_path = pred_dir / f"{name}.png"
            if pred_path.exists():
                pairs.append((gt_path, pred_path, name))
        
        if not pairs:
            print(f"  警告: 未找到匹配的预测文件")
            continue
        
        print(f"  找到 {len(pairs)} 对匹配文件")
        
        # 计算各指标
        model_results = {'model': model_name}
        per_image_metrics = defaultdict(list)
        
        # 累计混淆矩阵
        confusion_total = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        for gt_path, pred_path, name in tqdm(pairs, desc="  Computing metrics"):
            gt = load_mask(gt_path, offset=args.gt_offset)
            pred = load_mask(pred_path)
            
            confusion = compute_confusion_matrix(gt, pred, num_classes, args.ignore_label)
            confusion_total += confusion
            
            # 单图 mIoU
            miou_res = compute_miou_from_confusion(confusion)
            per_image_metrics['mIoU'].append(miou_res['miou'])
            
            # 单图 hierIoU（各矩阵）
            for sim_name, S in matrices.items():
                hier_res = compute_hier_soft_iou_from_confusion(confusion, S)
                per_image_metrics[f'hierIoU_{sim_name}'].append(hier_res['hier_miou'])
        
        # Image-level mean
        model_results['mIoU'] = np.mean(per_image_metrics['mIoU'])
        for sim_name in matrices.keys():
            model_results[f'hierIoU_{sim_name}'] = np.mean(per_image_metrics[f'hierIoU_{sim_name}'])
        
        # Dataset-level
        miou_ds = compute_miou_from_confusion(confusion_total)
        model_results['mIoU_dataset'] = miou_ds['miou']
        
        # Bootstrap CI
        model_bootstrap = {}
        for metric_name, values in per_image_metrics.items():
            values = np.array(values)
            boot_result = bootstrap_single_metric(values, n_bootstrap=1000, seed=args.seed)
            model_bootstrap[metric_name] = boot_result
            model_results[f'{metric_name}_ci_low'] = boot_result['ci_low']
            model_results[f'{metric_name}_ci_high'] = boot_result['ci_high']
        
        bootstrap_results[model_name] = model_bootstrap
        ranking_results.append(model_results)
        
        print(f"  mIoU: {model_results['mIoU']:.4f}")
        for sim_name in matrices.keys():
            print(f"  hierIoU({sim_name}): {model_results[f'hierIoU_{sim_name}']:.4f}")
    
    # C2: 错误分层
    print("\n" + "=" * 60)
    print("C2. Error Stratification")
    print("=" * 60)
    
    stratification_results = []
    top_far_confusions = {}
    
    # 使用主矩阵（hybrid）进行分层
    primary_matrix = matrices.get('hybrid', list(matrices.values())[0])
    primary_name = 'hybrid' if 'hybrid' in matrices else list(matrices.keys())[0]
    
    for pred_dir in pred_dirs:
        pred_dir = Path(pred_dir)
        model_name = pred_dir.name
        print(f"\n处理: {model_name}")
        
        pairs = []
        for gt_path, _, name in image_pairs:
            pred_path = pred_dir / f"{name}.png"
            if pred_path.exists():
                pairs.append((gt_path, pred_path, name))
        
        if not pairs:
            continue
        
        # 收集所有错误像素的相似度
        error_sims = []
        confusion_pairs = defaultdict(int)  # (gt, pred) -> count
        
        for gt_path, pred_path, name in tqdm(pairs, desc="  Collecting errors"):
            gt = load_mask(gt_path, offset=args.gt_offset)
            pred = load_mask(pred_path)
            
            valid = (gt != args.ignore_label) & (gt >= 0) & (gt < num_classes)
            error_mask = (gt != pred) & valid
            
            gt_err = gt[error_mask]
            pred_err = pred[error_mask]
            
            for g, p in zip(gt_err, pred_err):
                if 0 <= g < num_classes and 0 <= p < num_classes:
                    error_sims.append(primary_matrix[g, p])
                    confusion_pairs[(int(g), int(p))] += 1
        
        if not error_sims:
            continue
        
        error_sims = np.array(error_sims)
        total_errors = len(error_sims)
        
        # 分桶
        p25 = np.percentile(error_sims, 25)
        p75 = np.percentile(error_sims, 75)
        
        far_count = (error_sims < p25).sum()
        mid_count = ((error_sims >= p25) & (error_sims <= p75)).sum()
        near_count = (error_sims > p75).sum()
        
        strat_result = {
            'model': model_name,
            'total_errors': total_errors,
            'far_count': int(far_count),
            'mid_count': int(mid_count),
            'near_count': int(near_count),
            'far_ratio': far_count / total_errors,
            'mid_ratio': mid_count / total_errors,
            'near_ratio': near_count / total_errors,
            'p25': float(p25),
            'p75': float(p75),
            'mean_sim': float(error_sims.mean())
        }
        stratification_results.append(strat_result)
        
        print(f"  Far(<P25): {strat_result['far_ratio']:.1%}, "
              f"Mid(P25-P75): {strat_result['mid_ratio']:.1%}, "
              f"Near(>P75): {strat_result['near_ratio']:.1%}")
        
        # Top-10 far confusions
        far_threshold = p25
        far_confusions = [(k, v) for k, v in confusion_pairs.items() 
                         if primary_matrix[k[0], k[1]] < far_threshold]
        far_confusions.sort(key=lambda x: x[1], reverse=True)
        
        top_far_confusions[model_name] = [
            {'gt': int(k[0]), 'pred': int(k[1]), 'count': v, 
             'similarity': float(primary_matrix[k[0], k[1]])}
            for k, v in far_confusions[:10]
        ]
    
    # 保存结果
    print("\n保存结果...")
    
    # C1 结果
    save_csv(ranking_results, out_dir / "tables" / "real_ranking.csv")
    generate_ranking_latex(ranking_results, list(matrices.keys()), 
                          out_dir / "tables" / "real_ranking.tex")
    save_json(bootstrap_results, out_dir / "logs" / "ranking_bootstrap.json")
    
    # C2 结果
    save_csv(stratification_results, out_dir / "tables" / "error_stratification.csv")
    generate_stratification_latex(stratification_results, 
                                  out_dir / "tables" / "error_stratification.tex")
    save_json(top_far_confusions, out_dir / "logs" / "top_far_confusions.json")
    
    # 绘图
    plot_ranking_results(ranking_results, list(matrices.keys()), out_dir / "plots")
    plot_error_stratification(stratification_results, out_dir / "plots")
    
    return {
        'ranking': ranking_results,
        'stratification': stratification_results,
        'top_far_confusions': top_far_confusions
    }


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
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'std': float(bootstrap_means.std())
    }


def plot_ranking_results(results: List[dict], sim_names: List[str], out_dir: Path):
    """绘制排名结果图"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    models = [r['model'] for r in results]
    n_models = len(models)
    
    # 图1: 多指标对比（带误差线）
    metrics = ['mIoU'] + [f'hierIoU_{s}' for s in sim_names]
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 6))
    
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ci_lows = [r.get(f'{metric}_ci_low', r[metric]) for r in results]
        ci_highs = [r.get(f'{metric}_ci_high', r[metric]) for r in results]
        
        yerr_low = [v - ci_l for v, ci_l in zip(values, ci_lows)]
        yerr_high = [ci_h - v for v, ci_h in zip(values, ci_highs)]
        
        ax.bar(x + i * width, values, width, 
               label=metric, color=colors[i], alpha=0.8,
               yerr=[yerr_low, yerr_high], capsize=3)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Multi-Metric Comparison with 95% CI', fontsize=14)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'multi_metric_ranking.png', dpi=150)
    plt.savefig(out_dir / 'multi_metric_ranking.pdf')
    plt.close()
    
    print(f"  排名图表已保存到: {out_dir}")


def plot_error_stratification(results: List[dict], out_dir: Path):
    """绘制错误分层图"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    models = [r['model'] for r in results]
    
    # 堆叠柱状图
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    
    x = np.arange(len(models))
    
    far = [r['far_ratio'] for r in results]
    mid = [r['mid_ratio'] for r in results]
    near = [r['near_ratio'] for r in results]
    
    ax.bar(x, far, label='Far (S < P25)', color='#e74c3c', alpha=0.8)
    ax.bar(x, mid, bottom=far, label='Mid (P25-P75)', color='#f39c12', alpha=0.8)
    ax.bar(x, near, bottom=[f+m for f, m in zip(far, mid)], 
           label='Near (S > P75)', color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error Ratio', fontsize=12)
    ax.set_title('Error Stratification by Semantic Similarity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(out_dir / 'error_stratification.png', dpi=150)
    plt.savefig(out_dir / 'error_stratification.pdf')
    plt.close()
    
    print(f"  分层图表已保存到: {out_dir}")


def generate_ranking_latex(results: List[dict], sim_names: List[str], out_path: Path):
    """生成排名 LaTeX 表格"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics = ['mIoU'] + [f'hierIoU_{s}' for s in sim_names]
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-Metric Ranking on Real Predictions}",
        r"\label{tab:real_ranking}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Model & " + " & ".join(metrics) + r" \\",
        r"\midrule",
    ]
    
    for r in results:
        row = r['model']
        for m in metrics:
            val = r[m]
            ci_low = r.get(f'{m}_ci_low', val)
            ci_high = r.get(f'{m}_ci_high', val)
            row += f" & {val:.4f}"
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def generate_stratification_latex(results: List[dict], out_path: Path):
    """生成分层 LaTeX 表格"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Error Stratification by Semantic Similarity}",
        r"\label{tab:error_strat}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Far (\%) & Mid (\%) & Near (\%) & Mean Sim. \\",
        r"\midrule",
    ]
    
    for r in results:
        row = f"{r['model']} & {r['far_ratio']*100:.1f} & {r['mid_ratio']*100:.1f} "
        row += f"& {r['near_ratio']*100:.1f} & {r['mean_sim']:.3f}"
        row += r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ============================================================================
# 辅助函数
# ============================================================================

def load_similarity_matrices(args) -> dict:
    """加载所有相似度矩阵"""
    matrices = {}
    
    # 从参数中获取矩阵路径
    matrix_paths = {
        'hybrid': getattr(args, 'hybrid_path', None) or getattr(args, 'Hij_path', None),
        'pathsim': getattr(args, 'pathsim_path', None),
        'wup': getattr(args, 'wup_path', None)
    }
    
    print("\n加载相似度矩阵:")
    for name, path in matrix_paths.items():
        if path is None:
            continue
        
        path = Path(path)
        # 支持目录或文件
        if path.is_dir():
            s_path = path / "S.npy"
        else:
            s_path = path
        
        if s_path.exists():
            S = np.load(s_path).astype(np.float64)
            matrices[name] = S
            print(f"  {name}: {s_path}")
            print(f"    形状: {S.shape}, 值域: [{S.min():.4f}, {S.max():.4f}]")
        else:
            print(f"  {name}: 未找到 ({s_path})")
    
    return matrices


def analyze_matrices(matrices: dict):
    """分析矩阵特性"""
    print("\n相似度矩阵特性分析:")
    for name, S in matrices.items():
        n = S.shape[0]
        off_diag = S[np.triu_indices(n, k=1)]
        print(f"\n  {name}:")
        print(f"    对角线: min={np.diag(S).min():.4f}, max={np.diag(S).max():.4f}")
        print(f"    非对角: min={off_diag.min():.4f}, max={off_diag.max():.4f}, "
              f"mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="指标有效性验证实验套件 V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # A. 公平版多矩阵对比
  python run_metric_experiments_v2.py --mode end2end \\
      --gt_dir /path/to/gt \\
      --hybrid_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pathsim_path similarity_output/ade150/S.npy \\
      --wup_path similarity_output/ade150_wup/S.npy \\
      --out_dir output/end2end_comparison \\
      --ratios 0.05 0.10 0.20 0.30

  # B. 防洗白验证
  python run_metric_experiments_v2.py --mode far_credit \\
      --gt_dir /path/to/gt \\
      --hybrid_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pathsim_path similarity_output/ade150/S.npy \\
      --wup_path similarity_output/ade150_wup/S.npy \\
      --out_dir output/far_credit_test \\
      --far_quantile 0.1

  # C. 真实预测验证
  python run_metric_experiments_v2.py --mode real_validation \\
      --gt_dir /path/to/gt \\
      --pred_dirs /path/to/model1_preds /path/to/model2_preds \\
      --hybrid_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pathsim_path similarity_output/ade150/S.npy \\
      --wup_path similarity_output/ade150_wup/S.npy \\
      --out_dir output/real_validation
        """
    )
    
    # 模式
    parser.add_argument("--mode", type=str, required=True,
                        choices=['end2end', 'far_credit', 'real_validation', 'all'],
                        help="运行模式")
    
    # 输入路径
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="GT masks 目录")
    parser.add_argument("--pred_dirs", type=Path, nargs='*', default=[],
                        help="预测目录列表（real_validation 模式需要）")
    
    # 相似度矩阵路径
    parser.add_argument("--hybrid_path", type=Path, default=None,
                        help="Hybrid 相似度矩阵路径")
    parser.add_argument("--Hij_path", type=Path, default=None,
                        help="Hij 路径（等同于 hybrid_path）")
    parser.add_argument("--pathsim_path", type=Path, default=None,
                        help="PathSim 相似度矩阵路径")
    parser.add_argument("--wup_path", type=Path, default=None,
                        help="WuP 相似度矩阵路径")
    
    # 输出
    parser.add_argument("--out_dir", type=Path, default=Path("./output"),
                        help="输出目录")
    
    # 参数
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--gt_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_k", type=int, default=5)
    parser.add_argument("--ratios", type=float, nargs='+',
                        default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--far_quantile", type=float, default=0.1,
                        help="Far credit 的分位数阈值")
    parser.add_argument("--skip_generate", action="store_true",
                        help="跳过生成步骤")
    
    args = parser.parse_args()
    
    # 如果指定了 Hij_path 但没有 hybrid_path，使用 Hij_path
    if args.hybrid_path is None and args.Hij_path is not None:
        args.hybrid_path = args.Hij_path
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建输出目录
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存元数据
    save_json(get_experiment_meta(args), args.out_dir / "logs" / "experiment_meta.json")
    
    # 运行实验
    if args.mode == 'end2end':
        run_end2end_matrix_comparison(args)
    elif args.mode == 'far_credit':
        run_far_credit_test(args)
    elif args.mode == 'real_validation':
        run_real_prediction_validation(args)
    elif args.mode == 'all':
        run_end2end_matrix_comparison(args)
        run_far_credit_test(args)
        if args.pred_dirs:
            run_real_prediction_validation(args)
    
    print("\n" + "=" * 70)
    print(f"实验完成，结果保存到: {args.out_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())