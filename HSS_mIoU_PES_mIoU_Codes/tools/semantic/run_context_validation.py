#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情景条件化指标验证实验 V2

包含：
- 实验 A: Supported vs Unsupported Corruption (带公平性检查)
- 实验 B: 真实预测上的 Hallucination Penalty (统一口径)
- 实验 C: ctx_real_diffset - 真实预测对比显影
- 单元测试: beta=0 退化验证
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
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

from semantic_metrics import compute_confusion_matrix, compute_miou_from_confusion, compute_hier_soft_iou_from_confusion
from corrupt import load_mask, save_mask
from context_metrics import (
    get_context_set,
    compute_context_support,
    compute_context_scaling,
    evaluate_image_with_context,
    evaluate_dataset_with_context,
    compute_supported_unsupported_targets,
    compute_fairness_matched_targets,
    generate_context_corruption,
    compute_unsupported_error_ratio,
    load_mask_safe,
    assert_ctx_reduces_to_base,
    assert_support_mode_difference
)
from io_utils import find_image_pairs, save_json, save_csv, get_experiment_meta


def load_similarity_matrix(path: Path) -> np.ndarray:
    """加载相似度矩阵"""
    if path.is_dir():
        s_path = path / "S.npy"
    else:
        s_path = path
    return np.load(s_path).astype(np.float64)


def bootstrap_metric(values: np.ndarray, n_bootstrap: int = 1000,
                    confidence_level: float = 0.95, seed: int = 42) -> Dict:
    """Bootstrap 单个指标"""
    rng = np.random.default_rng(seed)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means.append(values[indices].mean())
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence_level
    
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'ci_low': float(np.percentile(bootstrap_means, 100 * alpha / 2)),
        'ci_high': float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    }


def bootstrap_difference(values1: np.ndarray, values2: np.ndarray,
                        n_bootstrap: int = 1000, seed: int = 42) -> Dict:
    """Bootstrap 两组值的差异"""
    rng = np.random.default_rng(seed)
    n = len(values1)
    diff = values1 - values2
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_diffs.append(diff[indices].mean())
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    return {
        'mean_diff': float(diff.mean()),
        'ci_low': float(np.percentile(bootstrap_diffs, 2.5)),
        'ci_high': float(np.percentile(bootstrap_diffs, 97.5)),
        'significant': bool(np.percentile(bootstrap_diffs, 2.5) > 0 or 
                          np.percentile(bootstrap_diffs, 97.5) < 0)
    }


# ============================================================================
# 单元测试模式
# ============================================================================

def run_unit_tests(args) -> Dict:
    """
    运行单元测试：验证 beta=0 退化和 support_mode 差异
    """
    print("=" * 70)
    print("单元测试：指标一致性验证")
    print("=" * 70)
    
    H = load_similarity_matrix(args.similarity_path)
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    
    # 需要预测目录来测试
    if not args.pred_dirs:
        print("警告：未指定 pred_dirs，使用 GT 作为预测")
        test_pairs = [(p[0], p[0], p[2]) for p in image_pairs[:50]]
    else:
        pred_dir = Path(args.pred_dirs[0])
        test_pairs = []
        for gt_path, _, name in image_pairs[:50]:
            pred_path = find_matching_pred(pred_dir, name)
            if pred_path:
                test_pairs.append((gt_path, pred_path, name))
    
    results = {}
    
    # 测试 1: beta=0 退化
    print("\n[测试 1] beta=0 退化验证...")
    try:
        assert_ctx_reduces_to_base(
            test_pairs, H, 
            num_samples=20,
            num_classes=args.num_classes,
            ignore_label=args.ignore_label,
            gt_offset=args.gt_offset
        )
        results['beta0_test'] = 'PASSED'
    except AssertionError as e:
        results['beta0_test'] = f'FAILED: {e}'
        print(f"  ✗ {e}")
    
    # 测试 2: support_mode 差异
    print("\n[测试 2] support_mode 差异验证...")
    for beta in [0.25, 0.5, 0.75, 1.0]:
        mode_diff = assert_support_mode_difference(
            test_pairs, H, beta=beta,
            num_samples=20,
            num_classes=args.num_classes,
            ignore_label=args.ignore_label,
            gt_offset=args.gt_offset
        )
        results[f'mode_diff_beta{beta}'] = mode_diff
    
    # 保存结果
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / "logs" / "unit_test_results.json")
    
    print(f"\n测试结果保存到: {out_dir / 'logs' / 'unit_test_results.json'}")
    
    return results


# ============================================================================
# 实验 A: Supported vs Unsupported Corruption (带公平性检查)
# ============================================================================

def run_supported_unsupported_experiment(args) -> Dict:
    """
    实验 A: Supported vs Unsupported Corruption
    
    带公平性检查：验证两类 corruption 的相似度分布是否接近
    """
    print("=" * 70)
    print("实验 A: Supported vs Unsupported Corruption")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    H = load_similarity_matrix(args.similarity_path)
    print(f"\n相似度矩阵: shape={H.shape}, range=[{H.min():.4f}, {H.max():.4f}]")
    
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"找到 {len(image_pairs)} 张 GT 图片")
    
    if args.max_images > 0 and len(image_pairs) > args.max_images:
        image_pairs = image_pairs[:args.max_images]
        print(f"限制使用前 {args.max_images} 张图片")
    
    num_classes = args.num_classes
    ratios = args.ratios
    beta_list = args.beta_list
    sim_range = (args.sim_low, args.sim_high)
    support_mode = args.support_mode
    use_fair_matching = args.fair_matching
    
    print(f"\n实验参数:")
    print(f"  替换比例: {ratios}")
    print(f"  Beta 列表: {beta_list}")
    print(f"  相似度区间: {sim_range}")
    print(f"  Support 模式: {support_mode}")
    print(f"  公平匹配: {use_fair_matching}")
    
    all_results = []
    fairness_stats = []
    g_stats_by_beta = {beta: {'sup': [], 'unsup': []} for beta in beta_list}
    
    for r in ratios:
        print(f"\n{'='*50}")
        print(f"替换比例 r = {r}")
        print(f"{'='*50}")
        
        supported_results = defaultdict(list)
        unsupported_results = defaultdict(list)
        
        # 公平性统计
        sup_sims_all = []
        unsup_sims_all = []
        sup_new_classes_all = []
        unsup_new_classes_all = []
        
        for gt_path, _, name in tqdm(image_pairs, desc=f"  Processing r={r}"):
            gt = load_mask_safe(gt_path, offset=args.gt_offset)
            context_set = get_context_set(gt, args.ignore_label, num_classes)
            
            if len(context_set) < 2:
                continue
            
            # 计算 targets
            if use_fair_matching:
                targets = compute_fairness_matched_targets(
                    H, context_set, sim_range, num_bins=5, seed=args.seed
                )
            else:
                targets = compute_supported_unsupported_targets(
                    H, context_set, sim_range, seed=args.seed
                )
            
            # 检查有效目标
            valid_sup = sum(1 for t in targets.values() if t['supported'] is not None)
            valid_unsup = sum(1 for t in targets.values() if t['unsupported'] is not None)
            
            if valid_sup < 3 or valid_unsup < 3:
                continue
            
            # 生成两类 corruption
            pred_sup, stats_sup = generate_context_corruption(
                gt, targets, 'supported', r, 
                seed=args.seed, ignore_label=args.ignore_label, num_classes=num_classes
            )
            pred_unsup, stats_unsup = generate_context_corruption(
                gt, targets, 'unsupported', r,
                seed=args.seed, ignore_label=args.ignore_label, num_classes=num_classes
            )
            
            # 收集公平性统计
            if stats_sup['replacement_sims']:
                sup_sims_all.extend(stats_sup['replacement_sims'])
                sup_new_classes_all.append(stats_sup['num_new_classes'])
            if stats_unsup['replacement_sims']:
                unsup_sims_all.extend(stats_unsup['replacement_sims'])
                unsup_new_classes_all.append(stats_unsup['num_new_classes'])
            
            # 对每个 beta 计算指标
            for beta in beta_list:
                result_sup = evaluate_image_with_context(
                    gt, pred_sup, H, beta, num_classes, args.ignore_label, support_mode
                )
                supported_results[beta].append({
                    'name': name,
                    'miou': result_sup['miou'],
                    'hier_base': result_sup['hier_base'],
                    'hier_ctx': result_sup['hier_ctx'],
                    'mean_g_sup': result_sup['mean_g_supported'],
                    'mean_g_unsup': result_sup['mean_g_unsupported']
                })
                
                result_unsup = evaluate_image_with_context(
                    gt, pred_unsup, H, beta, num_classes, args.ignore_label, support_mode
                )
                unsupported_results[beta].append({
                    'name': name,
                    'miou': result_unsup['miou'],
                    'hier_base': result_unsup['hier_base'],
                    'hier_ctx': result_unsup['hier_ctx'],
                    'mean_g_sup': result_unsup['mean_g_supported'],
                    'mean_g_unsup': result_unsup['mean_g_unsupported']
                })
                
                # 收集 g 统计
                g_stats_by_beta[beta]['sup'].append(result_sup['mean_g_supported'])
                g_stats_by_beta[beta]['unsup'].append(result_unsup['mean_g_unsupported'])
        
        # 公平性统计
        fairness = {
            'r': r,
            'sup_sim_mean': float(np.mean(sup_sims_all)) if sup_sims_all else 0,
            'sup_sim_p25': float(np.percentile(sup_sims_all, 25)) if sup_sims_all else 0,
            'sup_sim_p75': float(np.percentile(sup_sims_all, 75)) if sup_sims_all else 0,
            'unsup_sim_mean': float(np.mean(unsup_sims_all)) if unsup_sims_all else 0,
            'unsup_sim_p25': float(np.percentile(unsup_sims_all, 25)) if unsup_sims_all else 0,
            'unsup_sim_p75': float(np.percentile(unsup_sims_all, 75)) if unsup_sims_all else 0,
            'sup_new_classes_mean': float(np.mean(sup_new_classes_all)) if sup_new_classes_all else 0,
            'unsup_new_classes_mean': float(np.mean(unsup_new_classes_all)) if unsup_new_classes_all else 0
        }
        fairness_stats.append(fairness)
        
        print(f"\n  公平性检查:")
        print(f"    Supported sim: mean={fairness['sup_sim_mean']:.3f}, "
              f"P25={fairness['sup_sim_p25']:.3f}, P75={fairness['sup_sim_p75']:.3f}")
        print(f"    Unsupported sim: mean={fairness['unsup_sim_mean']:.3f}, "
              f"P25={fairness['unsup_sim_p25']:.3f}, P75={fairness['unsup_sim_p75']:.3f}")
        print(f"    Sup new classes: {fairness['sup_new_classes_mean']:.1f}, "
              f"Unsup new classes: {fairness['unsup_new_classes_mean']:.1f}")
        
        # 汇总该 r 的结果
        for beta in beta_list:
            if not supported_results[beta] or not unsupported_results[beta]:
                continue
            
            sup_miou = np.array([r['miou'] for r in supported_results[beta]])
            sup_hier_base = np.array([r['hier_base'] for r in supported_results[beta]])
            sup_hier_ctx = np.array([r['hier_ctx'] for r in supported_results[beta]])
            
            unsup_miou = np.array([r['miou'] for r in unsupported_results[beta]])
            unsup_hier_base = np.array([r['hier_base'] for r in unsupported_results[beta]])
            unsup_hier_ctx = np.array([r['hier_ctx'] for r in unsupported_results[beta]])
            
            diff_miou = bootstrap_difference(sup_miou, unsup_miou, args.bootstrap, args.seed)
            diff_base = bootstrap_difference(sup_hier_base, unsup_hier_base, args.bootstrap, args.seed)
            diff_ctx = bootstrap_difference(sup_hier_ctx, unsup_hier_ctx, args.bootstrap, args.seed)
            
            result = {
                'r': r,
                'beta': beta,
                'support_mode': support_mode,
                'n_images': len(supported_results[beta]),
                'sup_miou': float(sup_miou.mean()),
                'sup_hier_base': float(sup_hier_base.mean()),
                'sup_hier_ctx': float(sup_hier_ctx.mean()),
                'unsup_miou': float(unsup_miou.mean()),
                'unsup_hier_base': float(unsup_hier_base.mean()),
                'unsup_hier_ctx': float(unsup_hier_ctx.mean()),
                'diff_miou': diff_miou['mean_diff'],
                'diff_miou_ci': [diff_miou['ci_low'], diff_miou['ci_high']],
                'diff_base': diff_base['mean_diff'],
                'diff_base_ci': [diff_base['ci_low'], diff_base['ci_high']],
                'diff_ctx': diff_ctx['mean_diff'],
                'diff_ctx_ci': [diff_ctx['ci_low'], diff_ctx['ci_high']],
                'diff_ctx_significant': diff_ctx['significant']
            }
            all_results.append(result)
            
            print(f"\n  Beta={beta}:")
            print(f"    Sup: mIoU={sup_miou.mean():.4f}, base={sup_hier_base.mean():.4f}, ctx={sup_hier_ctx.mean():.4f}")
            print(f"    Unsup: mIoU={unsup_miou.mean():.4f}, base={unsup_hier_base.mean():.4f}, ctx={unsup_hier_ctx.mean():.4f}")
            print(f"    Diff: mIoU={diff_miou['mean_diff']:.4f}, base={diff_base['mean_diff']:.4f}, ctx={diff_ctx['mean_diff']:.4f}")
    
    # g 统计汇总
    g_summary = {}
    for beta in beta_list:
        g_summary[f'beta_{beta}'] = {
            'mean_g_supported': float(np.mean(g_stats_by_beta[beta]['sup'])) if g_stats_by_beta[beta]['sup'] else 1.0,
            'mean_g_unsupported': float(np.mean(g_stats_by_beta[beta]['unsup'])) if g_stats_by_beta[beta]['unsup'] else 0.0
        }
    
    # 保存结果
    print("\n保存结果...")
    save_csv(all_results, out_dir / "tables" / "supported_unsupported_comparison.csv")
    save_csv(fairness_stats, out_dir / "tables" / "fairness_check.csv")
    save_json({
        'results': all_results,
        'fairness': fairness_stats,
        'g_summary': g_summary,
        'support_mode': support_mode,
        'fair_matching': use_fair_matching
    }, out_dir / "logs" / "experiment_a_results.json")
    
    # 绘图
    plot_sup_unsup_with_g_curves(all_results, g_summary, beta_list, out_dir / "plots")
    
    return {'results': all_results, 'fairness': fairness_stats, 'g_summary': g_summary}


def plot_sup_unsup_with_g_curves(results: List[Dict], g_summary: Dict, 
                                  beta_list: List[float], out_dir: Path):
    """绘制 Supported vs Unsupported 对比图，包含 g 曲线"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # 图1: g(j|x) 随 beta 变化
    fig, ax = plt.subplots(figsize=(8, 6))
    
    betas = sorted(beta_list)
    g_sup = [g_summary.get(f'beta_{b}', {}).get('mean_g_supported', 1.0) for b in betas]
    g_unsup = [g_summary.get(f'beta_{b}', {}).get('mean_g_unsupported', 0.0) for b in betas]
    
    ax.plot(betas, g_sup, 'o-', label='g (supported)', linewidth=2, markersize=8, color='green')
    ax.plot(betas, g_unsup, 's-', label='g (unsupported)', linewidth=2, markersize=8, color='red')
    
    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('Mean g(j|x)', fontsize=12)
    ax.set_title('Context Scaling Factor vs β', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(out_dir / 'g_scaling_vs_beta.png', dpi=150)
    plt.savefig(out_dir / 'g_scaling_vs_beta.pdf')
    plt.close()
    
    # 图2: 差异曲线
    ratios = sorted(set(r['r'] for r in results))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['diff_miou', 'diff_base', 'diff_ctx']
    titles = ['mIoU', 'hierIoU (base)', 'hierIoU (ctx)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for beta in beta_list:
            beta_results = [r for r in results if r['beta'] == beta]
            beta_results.sort(key=lambda x: x['r'])
            
            if beta_results:
                x = [r['r'] for r in beta_results]
                y = [r[metric] for r in beta_results]
                ax.plot(x, y, 'o-', label=f'β={beta}', linewidth=2, markersize=8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
        ax.set_ylabel('Diff (Sup - Unsup)', fontsize=12)
        ax.set_title(f'{title}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'sup_unsup_diff_by_beta.png', dpi=150)
    plt.savefig(out_dir / 'sup_unsup_diff_by_beta.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


# ============================================================================
# 实验 B: Hallucination Penalty (统一口径)
# ============================================================================

def find_matching_pred(pred_dir: Path, gt_name: str) -> Optional[Path]:
    """查找匹配的预测文件（优先 _pred_mask.png）"""
    patterns = [
        f"{gt_name}_pred_mask.png",   # ← 最优先：真正的预测 mask
        f"{gt_name}.png",
        f"{gt_name}_pred.png",
        f"{gt_name}_seg.png",
    ]
    
    for pattern in patterns:
        pred_path = pred_dir / pattern
        if pred_path.exists():
            return pred_path
    
    # 模糊匹配时也要排除 labeled 和 overlay
    if '_' in gt_name:
        core_id = gt_name.split('_')[-1]
        for f in sorted(pred_dir.glob(f"*{core_id}*.png")):
            if '_labeled' not in f.name and '_overlay' not in f.name:
                return f
    
    return None


def run_hallucination_penalty_experiment(args) -> Dict:
    """
    实验 B: 真实预测上的 Hallucination Penalty (统一口径版)
    
    输出与 real_ranking 一致口径的 mIoU 和 hierIoU
    """
    print("=" * 70)
    print("实验 B: Hallucination Penalty Analysis (统一口径)")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.pred_dirs:
        print("警告: 未指定预测目录，跳过实验 B")
        return {'error': 'No prediction directories'}
    
    H = load_similarity_matrix(args.similarity_path)
    print(f"\n相似度矩阵: shape={H.shape}")
    
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"找到 {len(image_pairs)} 张 GT 图片")
    
    pred_dirs = [Path(p) for p in args.pred_dirs]
    pred_names = list(args.pred_names) if args.pred_names else []
    while len(pred_names) < len(pred_dirs):
        pred_names.append(pred_dirs[len(pred_names)].name)
    
    print(f"\n预测目录 ({len(pred_dirs)} 个):")
    for name, pd in zip(pred_names, pred_dirs):
        print(f"  - {name}: {pd}")
    
    beta_list = args.beta_list
    num_classes = args.num_classes
    support_mode = args.support_mode
    
    all_results = []
    per_model_data = {}
    
    for pred_name, pred_dir in zip(pred_names, pred_dirs):
        print(f"\n{'='*50}")
        print(f"处理: {pred_name}")
        print(f"{'='*50}")
        
        pairs = []
        for gt_path, _, name in image_pairs:
            pred_path = find_matching_pred(pred_dir, name)
            if pred_path is not None:
                pairs.append((gt_path, pred_path, name))
        
        if not pairs:
            print(f"  警告: 未找到匹配的预测文件")
            continue
        
        print(f"  找到 {len(pairs)} 对匹配文件")
        
        per_image_metrics = []
        
        for gt_path, pred_path, name in tqdm(pairs, desc="  Computing metrics"):
            gt = load_mask_safe(gt_path, offset=args.gt_offset)
            pred = load_mask_safe(pred_path)
            
            # 计算 unsupported error ratio
            unsup_result = compute_unsupported_error_ratio(
                gt, pred, args.ignore_label, num_classes
            )
            
            # 对每个 beta 计算完整指标（与 real_ranking 一致口径）
            metrics_by_beta = {}
            for beta in beta_list:
                result = evaluate_image_with_context(
                    gt, pred, H, beta, num_classes, args.ignore_label, support_mode
                )
                metrics_by_beta[beta] = {
                    'miou': result['miou'],
                    'hier_base': result['hier_base'],
                    'hier_ctx': result['hier_ctx'],
                    'delta_ctx_base': result['hier_ctx'] - result['hier_base']
                }
            
            per_image_metrics.append({
                'name': name,
                'unsupported_ratio': unsup_result['unsupported_ratio'],
                'total_errors': unsup_result['total_errors'],
                'metrics': metrics_by_beta
            })
        
        per_model_data[pred_name] = per_image_metrics
        
        # 汇总
        unsup_ratios = np.array([m['unsupported_ratio'] for m in per_image_metrics])
        mean_unsup_ratio = float(np.mean(unsup_ratios))
        
        model_result = {
            'model': pred_name,
            'n_images': len(pairs),
            'mean_unsupported_ratio': mean_unsup_ratio,
            'std_unsupported_ratio': float(np.std(unsup_ratios))
        }
        
        # 对每个 beta 计算统计和相关性
        for beta in beta_list:
            mious = np.array([m['metrics'][beta]['miou'] for m in per_image_metrics])
            hier_bases = np.array([m['metrics'][beta]['hier_base'] for m in per_image_metrics])
            hier_ctxs = np.array([m['metrics'][beta]['hier_ctx'] for m in per_image_metrics])
            delta_ctx_bases = np.array([m['metrics'][beta]['delta_ctx_base'] for m in per_image_metrics])
            
            # Bootstrap
            boot_miou = bootstrap_metric(mious, args.bootstrap, seed=args.seed)
            boot_base = bootstrap_metric(hier_bases, args.bootstrap, seed=args.seed)
            boot_ctx = bootstrap_metric(hier_ctxs, args.bootstrap, seed=args.seed)
            
            model_result[f'miou'] = boot_miou['mean']
            model_result[f'hier_base'] = boot_base['mean']
            model_result[f'hier_ctx_beta{beta}'] = boot_ctx['mean']
            model_result[f'hier_ctx_beta{beta}_ci'] = [boot_ctx['ci_low'], boot_ctx['ci_high']]
            model_result[f'delta_ctx_base_beta{beta}'] = float(np.mean(delta_ctx_bases))
            
            # Spearman 相关性（unsupported ratio vs delta_ctx_base）
            if len(unsup_ratios) > 10:
                corr, pval = scipy_stats.spearmanr(unsup_ratios, delta_ctx_bases)
                model_result[f'spearman_delta_beta{beta}'] = float(corr)
                model_result[f'spearman_pval_beta{beta}'] = float(pval)
        
        all_results.append(model_result)
        
        print(f"  Mean Unsupported Ratio: {mean_unsup_ratio:.4f}")
        print(f"  mIoU: {model_result['miou']:.4f}")
        print(f"  hier_base: {model_result['hier_base']:.4f}")
        for beta in beta_list:
            print(f"  hierCtx (β={beta}): {model_result[f'hier_ctx_beta{beta}']:.4f}, "
                  f"Δ={model_result[f'delta_ctx_base_beta{beta}']:.4f}")
    
    # 计算跨模型相关性
    if len(all_results) >= 3:
        for beta in beta_list:
            unsup_ratios = [r['mean_unsupported_ratio'] for r in all_results]
            deltas = [r.get(f'delta_ctx_base_beta{beta}', 0) for r in all_results]
            if len(unsup_ratios) >= 3:
                corr, pval = scipy_stats.spearmanr(unsup_ratios, deltas)
                print(f"\n跨模型相关性 (β={beta}): Spearman(UnsupRatio, Δctx-base) = {corr:.4f} (p={pval:.4f})")
    
    # 保存
    print("\n保存结果...")
    save_csv(all_results, out_dir / "tables" / "hallucination_penalty_unified.csv")
    save_json({
        'results': all_results,
        'support_mode': support_mode,
        'note': '指标口径与 real_ranking 一致'
    }, out_dir / "logs" / "experiment_b_results.json")
    
    # 绘图
    plot_hallucination_unified(all_results, beta_list, out_dir / "plots")
    
    return {'results': all_results, 'per_model': per_model_data}


def plot_hallucination_unified(results: List[Dict], beta_list: List[float], out_dir: Path):
    """绘制统一口径的 Hallucination 分析图"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # 图1: Unsupported Ratio vs Delta(ctx-base)
    fig, axes = plt.subplots(1, len(beta_list), figsize=(5*len(beta_list), 5))
    if len(beta_list) == 1:
        axes = [axes]
    
    for ax, beta in zip(axes, beta_list):
        x = [r['mean_unsupported_ratio'] for r in results]
        y = [r.get(f'delta_ctx_base_beta{beta}', 0) for r in results]
        names = [r['model'] for r in results]
        
        ax.scatter(x, y, s=100, alpha=0.7)
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        ax.set_xlabel('Mean Unsupported Ratio', fontsize=11)
        ax.set_ylabel('Δ(hierCtx - hierBase)', fontsize=11)
        ax.set_title(f'β = {beta}', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'unsup_ratio_vs_delta_ctx_base.png', dpi=150)
    plt.savefig(out_dir / 'unsup_ratio_vs_delta_ctx_base.pdf')
    plt.close()
    
    # 图2: 多指标对比柱状图
    fig, ax = plt.subplots(figsize=(max(10, len(results)*2), 6))
    
    models = [r['model'] for r in results]
    x = np.arange(len(models))
    width = 0.2
    
    max_beta = max(beta_list)
    
    ax.bar(x - width, [r['miou'] for r in results], width, label='mIoU', alpha=0.8)
    ax.bar(x, [r['hier_base'] for r in results], width, label='hierBase', alpha=0.8)
    ax.bar(x + width, [r.get(f'hier_ctx_beta{max_beta}', 0) for r in results], 
           width, label=f'hierCtx (β={max_beta})', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Unified Metrics Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'unified_metrics_comparison.png', dpi=150)
    plt.savefig(out_dir / 'unified_metrics_comparison.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


# ============================================================================
# 实验 C: ctx_real_diffset - 真实预测对比显影
# ============================================================================

def run_ctx_real_diffset_experiment(args) -> Dict:
    """
    实验 C: 真实预测对比显影
    
    输出每个模型的 mIoU, hierBase, hierCtx(beta), UnsupportedRatio
    并计算相关性
    """
    print("=" * 70)
    print("实验 C: Context Real Diffset - 真实预测对比显影")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.pred_dirs:
        print("警告: 未指定预测目录")
        return {'error': 'No prediction directories'}
    
    H = load_similarity_matrix(args.similarity_path)
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"找到 {len(image_pairs)} 张 GT 图片")
    
    pred_dirs = [Path(p) for p in args.pred_dirs]
    pred_names = list(args.pred_names) if args.pred_names else []
    while len(pred_names) < len(pred_dirs):
        pred_names.append(pred_dirs[len(pred_names)].name)
    
    beta_list = args.beta_list
    num_classes = args.num_classes
    support_mode = args.support_mode
    
    all_results = []
    
    for pred_name, pred_dir in zip(pred_names, pred_dirs):
        print(f"\n处理: {pred_name}")
        
        pairs = []
        for gt_path, _, name in image_pairs:
            pred_path = find_matching_pred(pred_dir, name)
            if pred_path is not None:
                pairs.append((gt_path, pred_path, name))
        
        if not pairs:
            print(f"  未找到匹配文件")
            continue
        
        print(f"  找到 {len(pairs)} 对")
        
        # 收集指标
        mious = []
        hier_bases = []
        hier_ctxs_by_beta = {beta: [] for beta in beta_list}
        unsup_ratios = []
        
        for gt_path, pred_path, name in tqdm(pairs, desc="  Evaluating"):
            gt = load_mask_safe(gt_path, offset=args.gt_offset)
            pred = load_mask_safe(pred_path)
            
            unsup = compute_unsupported_error_ratio(gt, pred, args.ignore_label, num_classes)
            unsup_ratios.append(unsup['unsupported_ratio'])
            
            for beta in beta_list:
                result = evaluate_image_with_context(
                    gt, pred, H, beta, num_classes, args.ignore_label, support_mode
                )
                if beta == 0:
                    mious.append(result['miou'])
                    hier_bases.append(result['hier_base'])
                hier_ctxs_by_beta[beta].append(result['hier_ctx'])
        
        # 汇总
        model_result = {
            'model': pred_name,
            'n_images': len(pairs),
            'miou': float(np.mean(mious)),
            'hier_base': float(np.mean(hier_bases)),
            'mean_unsupported_ratio': float(np.mean(unsup_ratios))
        }
        
        for beta in beta_list:
            hier_ctx_mean = float(np.mean(hier_ctxs_by_beta[beta]))
            model_result[f'hier_ctx_beta{beta}'] = hier_ctx_mean
            model_result[f'delta_ctx_base_beta{beta}'] = hier_ctx_mean - model_result['hier_base']
        
        all_results.append(model_result)
        
        print(f"  mIoU: {model_result['miou']:.4f}, hierBase: {model_result['hier_base']:.4f}")
        print(f"  UnsupRatio: {model_result['mean_unsupported_ratio']:.4f}")
        for beta in beta_list:
            print(f"  hierCtx(β={beta}): {model_result[f'hier_ctx_beta{beta}']:.4f}, "
                  f"Δ={model_result[f'delta_ctx_base_beta{beta}']:.4f}")
    
    # 相关性分析
    print("\n" + "=" * 50)
    print("相关性分析")
    print("=" * 50)
    
    correlations = {}
    if len(all_results) >= 3:
        unsup = [r['mean_unsupported_ratio'] for r in all_results]
        for beta in beta_list:
            deltas = [r.get(f'delta_ctx_base_beta{beta}', 0) for r in all_results]
            corr, pval = scipy_stats.spearmanr(unsup, deltas)
            correlations[f'beta_{beta}'] = {'spearman': float(corr), 'pval': float(pval)}
            
            expected = "✓ (负相关符合预期)" if corr < 0 else "✗ (应为负相关)"
            print(f"  β={beta}: Spearman(UnsupRatio, Δctx-base) = {corr:.4f} (p={pval:.4f}) {expected}")
    
    # 保存
    save_csv(all_results, out_dir / "tables" / "ctx_real_diffset.csv")
    save_json({
        'results': all_results,
        'correlations': correlations,
        'support_mode': support_mode
    }, out_dir / "logs" / "experiment_c_results.json")
    
    # 绘图
    plot_ctx_real_diffset(all_results, beta_list, correlations, out_dir / "plots")
    
    return {'results': all_results, 'correlations': correlations}


def plot_ctx_real_diffset(results: List[Dict], beta_list: List[float], 
                          correlations: Dict, out_dir: Path):
    """绘制真实预测对比图"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # 图1: Delta(ctx-base) 随 beta 变化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for r in results:
        betas = beta_list
        deltas = [r.get(f'delta_ctx_base_beta{b}', 0) for b in betas]
        ax.plot(betas, deltas, 'o-', label=r['model'], linewidth=2, markersize=8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('Δ(hierCtx - hierBase)', fontsize=12)
    ax.set_title('Context Effect vs β', fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'delta_ctx_base_vs_beta.png', dpi=150)
    plt.savefig(out_dir / 'delta_ctx_base_vs_beta.pdf')
    plt.close()
    
    # 图2: UnsupportedRatio vs Delta 散点图
    max_beta = max(beta_list)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = [r['mean_unsupported_ratio'] for r in results]
    y = [r.get(f'delta_ctx_base_beta{max_beta}', 0) for r in results]
    names = [r['model'] for r in results]
    
    ax.scatter(x, y, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name, (x[i], y[i]), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    
    # 添加回归线
    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), '--', color='red', alpha=0.5, 
               label=f'Linear fit')
        
        corr_info = correlations.get(f'beta_{max_beta}', {})
        if corr_info:
            ax.text(0.05, 0.95, f"Spearman ρ = {corr_info['spearman']:.3f}\np = {corr_info['pval']:.3f}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Unsupported Ratio', fontsize=12)
    ax.set_ylabel(f'Δ(hierCtx - hierBase) at β={max_beta}', fontsize=12)
    ax.set_title('Hallucination Penalty Correlation', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'unsup_ratio_correlation.png', dpi=150)
    plt.savefig(out_dir / 'unsup_ratio_correlation.pdf')
    plt.close()
    
    print(f"  图表已保存到: {out_dir}")


# ============================================================================
# Beta Sweep 实验
# ============================================================================

def run_beta_sweep_experiment(args) -> Dict:
    """Beta 参数敏感性分析"""
    print("=" * 70)
    print("Beta Sweep 实验")
    print("=" * 70)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.pred_dirs:
        print("警告: 未指定预测目录")
        return {'error': 'No prediction directories'}
    
    H = load_similarity_matrix(args.similarity_path)
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    
    pred_dirs = [Path(p) for p in args.pred_dirs]
    pred_names = list(args.pred_names) if args.pred_names else []
    while len(pred_names) < len(pred_dirs):
        pred_names.append(pred_dirs[len(pred_names)].name)
    
    beta_list = args.beta_list
    support_mode = args.support_mode
    
    all_results = []
    
    for pred_name, pred_dir in zip(pred_names, pred_dirs):
        print(f"\n处理: {pred_name}")
        
        pairs = []
        for gt_path, _, name in image_pairs:
            pred_path = find_matching_pred(pred_dir, name)
            if pred_path is not None:
                pairs.append((gt_path, pred_path, name))
        
        if not pairs:
            continue
        
        for beta in beta_list:
            result = evaluate_dataset_with_context(
                pairs, H, beta, args.num_classes, args.ignore_label, 
                args.gt_offset, support_mode, verbose=False
            )
            
            all_results.append({
                'model': pred_name,
                'beta': beta,
                'support_mode': support_mode,
                'miou': result['mean_miou'],
                'hier_base': result['mean_hier_base'],
                'hier_ctx': result['mean_hier_ctx'],
                'delta_base': result['mean_delta_base'],
                'delta_ctx': result['mean_delta_ctx'],
                'mean_g_supported': result['mean_g_supported'],
                'mean_g_unsupported': result['mean_g_unsupported']
            })
            
            print(f"  β={beta}: ctx={result['mean_hier_ctx']:.4f}, "
                  f"g_sup={result['mean_g_supported']:.4f}, "
                  f"g_unsup={result['mean_g_unsupported']:.4f}")
    
    # 保存
    save_csv(all_results, out_dir / "tables" / "beta_sweep.csv")
    
    # 绘图
    plot_beta_sweep_with_g(all_results, out_dir / "plots")
    
    return {'results': all_results}


def plot_beta_sweep_with_g(results: List[Dict], out_dir: Path):
    """绘制 Beta sweep 图（包含 g 曲线）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models = sorted(set(r['model'] for r in results))
    betas = sorted(set(r['beta'] for r in results))
    
    # 图1: hierCtx vs beta
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        model_results.sort(key=lambda x: x['beta'])
        
        x = [r['beta'] for r in model_results]
        y = [r['hier_ctx'] for r in model_results]
        
        ax.plot(x, y, 'o-', label=model, linewidth=2, markersize=8)
    
    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('hierIoU (ctx)', fontsize=12)
    ax.set_title('Context-hierIoU vs β', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'beta_sweep_hier_ctx.png', dpi=150)
    plt.savefig(out_dir / 'beta_sweep_hier_ctx.pdf')
    plt.close()
    
    # 图2: g 曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        model_results.sort(key=lambda x: x['beta'])
        
        x = [r['beta'] for r in model_results]
        g_sup = [r['mean_g_supported'] for r in model_results]
        g_unsup = [r['mean_g_unsupported'] for r in model_results]
        
        ax.plot(x, g_sup, 'o-', label=f'{model} (sup)', linewidth=2, markersize=6)
        ax.plot(x, g_unsup, 's--', label=f'{model} (unsup)', linewidth=2, markersize=6)
    
    ax.set_xlabel('β', fontsize=12)
    ax.set_ylabel('Mean g(j|x)', fontsize=12)
    ax.set_title('Scaling Factor g vs β', fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(out_dir / 'beta_sweep_g_curves.png', dpi=150)
    plt.savefig(out_dir / 'beta_sweep_g_curves.pdf')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="情景条件化指标验证实验 V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行命令示例:

  # 单元测试
  python run_context_validation.py --mode unit_test \\
      --gt_dir /path/to/gt \\
      --similarity_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pred_dirs /path/to/pred1 \\
      --out_dir output/unit_test

  # 实验 A: Supported vs Unsupported (带公平性检查)
  python run_context_validation.py --mode sup_unsup \\
      --gt_dir /path/to/gt \\
      --similarity_path similarity_output/ade150_hybrid_inv/S.npy \\
      --out_dir output/exp_a \\
      --beta_list 0 0.25 0.5 0.75 1.0 \\
      --support_mode penalty_only \\
      --fair_matching

  # 实验 B: Hallucination Penalty (统一口径)
  python run_context_validation.py --mode hallucination \\
      --gt_dir /path/to/gt \\
      --similarity_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pred_dirs /path/to/pred1 /path/to/pred2 \\
      --pred_names "Model-A" "Model-B" \\
      --out_dir output/exp_b \\
      --support_mode penalty_only

  # 实验 C: 真实预测对比显影
  python run_context_validation.py --mode ctx_real_diffset \\
      --gt_dir /path/to/gt \\
      --similarity_path similarity_output/ade150_hybrid_inv/S.npy \\
      --pred_dirs /path/to/pred1 /path/to/pred2 /path/to/pred3 \\
      --pred_names "Model-A" "Model-B" "Model-C" \\
      --out_dir output/exp_c \\
      --beta_list 0 0.25 0.5 0.75 1.0
        """
    )
    
    # 模式
    parser.add_argument("--mode", type=str, required=True,
                        choices=['unit_test', 'sup_unsup', 'hallucination', 
                                'ctx_real_diffset', 'beta_sweep', 'all'],
                        help="运行模式")
    
    # 输入路径
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--similarity_path", type=Path, required=True)
    parser.add_argument("--pred_dirs", type=Path, nargs='*', default=[])
    parser.add_argument("--pred_names", type=str, nargs='*', default=[])
    
    # 输出
    parser.add_argument("--out_dir", type=Path, default=Path("./output/context_validation"))
    
    # 参数
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--gt_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Beta 和 support_mode
    parser.add_argument("--beta_list", type=float, nargs='+',
                        default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--support_mode", type=str, default='penalty_only',
                        choices=['penalty_only', 'exclude_diag'],
                        help="Support 计算模式")
    
    # Corruption 参数
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.05, 0.10, 0.20])
    parser.add_argument("--sim_low", type=float, default=0.3)
    parser.add_argument("--sim_high", type=float, default=0.7)
    parser.add_argument("--fair_matching", action="store_true",
                        help="使用公平匹配策略（按相似度分桶对齐）")
    
    # 其他
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=1000)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(get_experiment_meta(args), args.out_dir / "logs" / "experiment_meta.json")
    
    # 运行实验
    if args.mode == 'unit_test':
        run_unit_tests(args)
    
    elif args.mode == 'sup_unsup':
        run_supported_unsupported_experiment(args)
    
    elif args.mode == 'hallucination':
        run_hallucination_penalty_experiment(args)
    
    elif args.mode == 'ctx_real_diffset':
        run_ctx_real_diffset_experiment(args)
    
    elif args.mode == 'beta_sweep':
        run_beta_sweep_experiment(args)
    
    elif args.mode == 'all':
        if args.pred_dirs:
            run_unit_tests(args)
        run_supported_unsupported_experiment(args)
        if args.pred_dirs:
            run_hallucination_penalty_experiment(args)
            run_ctx_real_diffset_experiment(args)
            run_beta_sweep_experiment(args)
    
    print("\n" + "=" * 70)
    print(f"实验完成，结果保存到: {args.out_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())