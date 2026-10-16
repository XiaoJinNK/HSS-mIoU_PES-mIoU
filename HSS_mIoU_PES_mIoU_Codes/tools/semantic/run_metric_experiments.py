#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标有效性验证实验套件
"""

import argparse
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# 添加当前目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

# 本地模块
from semantic_metrics import (
    compute_confusion_matrix,
    compute_miou_from_confusion,
    compute_hier_soft_iou_from_confusion,
    evaluate_dataset
)
from corrupt import (
    compute_corruption_targets,
    compute_balanced_corruption_targets,  # 新增
    compute_class_pixel_counts,           # 新增
    generate_synthetic_predictions,
    verify_corruption_ratio
)
from bootstrap import bootstrap_comparison
from io_utils import (
    load_Hij, find_image_pairs, save_json, save_csv, save_latex_table,
    get_experiment_meta, load_mask
)


def run_sanity_check(args) -> dict:
    """
    运行 Sanity Check
    
    检查：
    1. Hij 属性（对称、对角线、值域）
    2. 指标值域 [0, 1]
    3. TP/FP/FN 非负
    4. 退化一致性：Hij=I 时 Hier-mIoU == mIoU
    """
    print("=" * 60)
    print("Sanity Check")
    print("=" * 60)
    
    # 加载数据
    Hij = load_Hij(args.Hij_path)
    num_classes = args.num_classes
    
    results = {
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # 1. 检查 Hij 属性
    print("\n[1] 检查 Hij 属性...")
    
    # 对称性
    is_symmetric = np.allclose(Hij, Hij.T, rtol=1e-5)
    results['checks']['Hij_symmetric'] = is_symmetric
    if not is_symmetric:
        results['warnings'].append("Hij 不对称")
    print(f"  对称性: {'✓' if is_symmetric else '✗'}")
    
    # 对角线
    diag_ok = np.allclose(np.diag(Hij), 1.0, rtol=1e-5)
    results['checks']['Hij_diagonal_ones'] = diag_ok
    if not diag_ok:
        results['warnings'].append("Hij 对角线不全为 1")
    print(f"  对角线为 1: {'✓' if diag_ok else '✗'}")
    
    # 值域
    range_ok = (Hij.min() >= 0) and (Hij.max() <= 1)
    results['checks']['Hij_range_01'] = range_ok
    if not range_ok:
        results['errors'].append(f"Hij 值域异常: [{Hij.min():.4f}, {Hij.max():.4f}]")
    print(f"  值域 [0,1]: {'✓' if range_ok else '✗'} (范围: [{Hij.min():.4f}, {Hij.max():.4f}])")
    
    # 2. 使用 GT=Pred 测试指标
    print("\n[2] 计算指标（GT=Pred）...")
    
    # 调试：打印路径信息
    gt_dir = Path(args.gt_dir)
    print(f"GT 目录: {gt_dir}")
    print(f"GT 目录存在: {gt_dir.exists()}")
    
    if gt_dir.exists():
        all_files = list(gt_dir.iterdir())
        print(f"目录内文件数: {len(all_files)}")
        # 检查文件类型
        extensions = {}
        for f in all_files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        print(f"文件扩展名统计: {extensions}")
    
    image_pairs = find_image_pairs(gt_dir, pattern="*.png")
    
    if not image_pairs:
        print("  警告: 未找到 .png 文件，尝试其他格式...")
        # 直接列出目录内容
        if gt_dir.exists():
            for ext in ["*.png", "*.PNG", "*.jpg", "*.tif"]:
                files = list(gt_dir.glob(ext))
                if files:
                    print(f"  找到 {ext}: {len(files)} 个")
        results['errors'].append("未找到图片")
        return results
    
    print(f"  找到 {len(image_pairs)} 张图片")
    
    # 评估
    eval_result = evaluate_dataset(
        image_pairs, Hij, num_classes,
        ignore_label=args.ignore_label,
        gt_offset=args.gt_offset,
        use_gt_as_pred=True,
        verbose=True
    )
    
    # 3. 检查指标值域
    print("\n[3] 检查指标值域...")
    
    miou_ok = 0 <= eval_result['miou'] <= 1
    hier_miou_ok = 0 <= eval_result['hier_miou'] <= 1
    
    results['checks']['miou_range'] = miou_ok
    results['checks']['hier_miou_range'] = hier_miou_ok
    
    print(f"  mIoU: {eval_result['miou']:.4f} ({'✓' if miou_ok else '✗'})")
    print(f"  Hier-mIoU: {eval_result['hier_miou']:.4f} ({'✓' if hier_miou_ok else '✗'})")
    
    # 4. 检查 TP/FP/FN 非负
    print("\n[4] 检查 TP/FP/FN 非负...")
    
    tp_ok = np.all(eval_result['tp'] >= 0)
    fp_ok = np.all(eval_result['fp'] >= 0)
    fn_ok = np.all(eval_result['fn'] >= 0)
    soft_tp_ok = np.all(eval_result['soft_tp'] >= 0)
    soft_fp_ok = np.all(eval_result['soft_fp'] >= 0)
    soft_fn_ok = np.all(eval_result['soft_fn'] >= 0)
    
    results['checks']['tp_non_negative'] = tp_ok
    results['checks']['fp_non_negative'] = fp_ok
    results['checks']['fn_non_negative'] = fn_ok
    results['checks']['soft_tp_non_negative'] = soft_tp_ok
    results['checks']['soft_fp_non_negative'] = soft_fp_ok
    results['checks']['soft_fn_non_negative'] = soft_fn_ok
    
    all_non_neg = tp_ok and fp_ok and fn_ok and soft_tp_ok and soft_fp_ok and soft_fn_ok
    print(f"  所有 TP/FP/FN 非负: {'✓' if all_non_neg else '✗'}")
    
    # 5. 退化一致性测试
    print("\n[5] 退化一致性测试（Hij=I）...")
    
    Hij_identity = np.eye(num_classes)
    
    # 重新累计混淆矩阵
    confusion_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt_path, _, name in image_pairs:
        gt = load_mask(gt_path, offset=args.gt_offset)
        confusion = compute_confusion_matrix(gt, gt, num_classes, args.ignore_label)
        confusion_total += confusion
    
    miou_result = compute_miou_from_confusion(confusion_total)
    hier_result_identity = compute_hier_soft_iou_from_confusion(confusion_total, Hij_identity)
    
    degrade_diff = abs(miou_result['miou'] - hier_result_identity['hier_miou'])
    degrade_ok = degrade_diff < 1e-6
    
    results['checks']['degradation_consistency'] = degrade_ok
    results['degradation_diff'] = float(degrade_diff)
    
    print(f"  mIoU: {miou_result['miou']:.6f}")
    print(f"  Hier-mIoU (Hij=I): {hier_result_identity['hier_miou']:.6f}")
    print(f"  差异: {degrade_diff:.2e} ({'✓ < 1e-6' if degrade_ok else '✗ >= 1e-6'})")
    
    # 6. 输出汇总表
    results['summary'] = {
        'miou_with_Hij': eval_result['miou'],
        'hier_miou_with_Hij': eval_result['hier_miou'],
        'miou_with_I': miou_result['miou'],
        'hier_miou_with_I': hier_result_identity['hier_miou'],
        'num_images': len(image_pairs),
        'num_classes': num_classes
    }
    
    # 保存结果
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(results, out_dir / "logs" / "sanity.json")
    
    # 保存表格
    table_data = [
        {'Setting': 'Hij=Original', 'mIoU': eval_result['miou'], 'Hier-mIoU': eval_result['hier_miou']},
        {'Setting': 'Hij=Identity', 'mIoU': miou_result['miou'], 'Hier-mIoU': hier_result_identity['hier_miou']},
    ]
    save_csv(table_data, out_dir / "tables" / "sanity_summary.csv")
    save_latex_table(
        table_data, out_dir / "tables" / "sanity_summary.tex",
        caption="Sanity Check: mIoU vs Hier-mIoU under different Hij",
        label="tab:sanity"
    )
    
    print("\n" + "=" * 60)
    print(f"Sanity Check 完成，结果保存到: {out_dir}")
    print("=" * 60)
    
    return results


def run_corruption_experiment(args) -> dict:
    """
    运行 Corruption 实验
    """
    print("=" * 60)
    print("Corruption Experiment")
    print("=" * 60)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    Hij = load_Hij(args.Hij_path)
    num_classes = args.num_classes
    
    gt_dir = Path(args.gt_dir)
    print(f"GT 目录: {gt_dir}")
    print(f"GT 目录存在: {gt_dir.exists()}")
    
    if gt_dir.exists():
        all_files = list(gt_dir.iterdir())
        print(f"目录内文件数: {len(all_files)}")
        extensions = {}
        for f in all_files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        print(f"文件扩展名统计: {extensions}")
    
    image_pairs = find_image_pairs(gt_dir, pattern="*.png")
    
    print(f"找到 {len(image_pairs)} 张图片")
    
    if not image_pairs:
        print("错误: 未找到图片，请检查路径")
        return {'error': 'No images found'}
    
    print(f"替换比例: {args.ratios}")
    
    synth_dir = out_dir / "synth_preds"
    
    # 检查是否跳过生成步骤
    skip_generate = getattr(args, 'skip_generate', False)
    use_balanced = getattr(args, 'balanced', False)
    
    if skip_generate:
        print("\n[1-2] 跳过生成步骤，使用已有的合成预测...")
        for r in args.ratios:
            r_str = f"r{int(r*100):02d}"
            for corrupt_type in ['fine', 'coarse', 'random']:
                pred_dir = synth_dir / corrupt_type / r_str
                if not pred_dir.exists():
                    print(f"错误: 找不到预测目录 {pred_dir}")
                    return {'error': f'Missing prediction dir: {pred_dir}'}
                n_files = len(list(pred_dir.glob("*.png")))
                print(f"  {corrupt_type}/{r_str}: {n_files} 个文件")
    else:
        # 1. 计算替换目标
        print("\n[1] 计算替换目标...")
        
        if use_balanced:
            print("  使用平衡模式（考虑类别像素分布）...")
            # 先统计类别像素分布
            class_counts = compute_class_pixel_counts(
                image_pairs, num_classes,
                ignore_label=args.ignore_label,
                gt_offset=args.gt_offset,
                max_images=200,  # 用前 200 张图统计
                verbose=True
            )
            targets = compute_balanced_corruption_targets(
                Hij, class_counts, k=args.target_k, seed=args.seed
            )
        else:
            print("  使用标准模式...")
            targets = compute_corruption_targets(Hij, k=args.target_k, seed=args.seed)
        
        # 保存目标
        targets_json = {
            'fine': targets['fine'].tolist(),
            'coarse': targets['coarse'].tolist(),
            'random': targets['random'].tolist(),
            'k': args.target_k,
            'seed': args.seed,
            'balanced': use_balanced
        }
        if 'fine_sims' in targets:
            targets_json['fine_sims_mean'] = float(targets['fine_sims'].mean())
            targets_json['coarse_sims_mean'] = float(targets['coarse_sims'].mean())
        
        save_json(targets_json, out_dir / "logs" / "targets.json")
        
        # 2. 生成合成预测
        print("\n[2] 生成合成预测...")
        
        generate_synthetic_predictions(
            image_pairs, targets, args.ratios, synth_dir,
            num_classes, args.ignore_label, args.gt_offset,
            seed=args.seed, save_format='png', verbose=True
        )
    
    # 3. 评测
    print("\n[3] 评测所有合成预测...")
    
    all_results = []
    per_image_data = {r: {t: [] for t in ['fine', 'coarse', 'random']} for r in args.ratios}
    
    for r in args.ratios:
        r_str = f"r{int(r*100):02d}"
        print(f"\n  替换比例 r={r}:")
        
        for corrupt_type in ['fine', 'coarse', 'random']:
            pred_dir = synth_dir / corrupt_type / r_str
            pairs = [(gt_path, pred_dir / f"{name}.png", name) 
                     for gt_path, _, name in image_pairs]
            
            eval_result = evaluate_dataset(
                pairs, Hij, num_classes,
                ignore_label=args.ignore_label,
                gt_offset=args.gt_offset,
                verbose=False
            )
            
            # 计算像素准确率（作为额外参考）
            pixel_acc = np.diag(eval_result['confusion']).sum() / max(eval_result['confusion'].sum(), 1)
            
            delta = eval_result['hier_miou'] - eval_result['miou']
            
            all_results.append({
                'r': r,
                'type': corrupt_type,
                'mIoU': eval_result['miou'],
                'hierIoU': eval_result['hier_miou'],
                'delta': delta,
                'pixel_acc': pixel_acc
            })
            
            per_image_data[r][corrupt_type] = eval_result['per_image_results']
            
            print(f"    {corrupt_type}: mIoU={eval_result['miou']:.4f}, "
                  f"hierIoU={eval_result['hier_miou']:.4f}, delta={delta:.4f}, "
                  f"pxAcc={pixel_acc:.4f}")
    
    # 4. 保存结果
    save_csv(all_results, out_dir / "tables" / "corruption_summary.csv")
    save_latex_table(
        all_results, out_dir / "tables" / "corruption_summary.tex",
        caption="Corruption Experiment Results",
        label="tab:corruption"
    )
    
    # 5. 绘图
    print("\n[4] 绘制图表...")
    plot_corruption_curves(all_results, out_dir / "plots")
    
    # 6. Bootstrap 统计
    print("\n[5] Bootstrap 统计显著性检验...")
    
    bootstrap_results = {}
    for r in args.ratios:
        fine_results = per_image_data[r]['fine']
        coarse_results = per_image_data[r]['coarse']
        
        miou_boot = bootstrap_comparison(
            fine_results, coarse_results, 
            metric_key='miou', n_bootstrap=1000, seed=args.seed
        )
        
        hier_boot = bootstrap_comparison(
            fine_results, coarse_results,
            metric_key='hier_miou', n_bootstrap=1000, seed=args.seed
        )
        
        bootstrap_results[f'r{int(r*100):02d}'] = {
            'miou_diff': miou_boot,
            'hier_miou_diff': hier_boot
        }
        
        print(f"  r={r}:")
        print(f"    mIoU diff (fine-coarse): {miou_boot['mean_diff']:.4f} "
              f"[{miou_boot['ci_low']:.4f}, {miou_boot['ci_high']:.4f}], "
              f"p={miou_boot['p_value']:.4f}")
        print(f"    Hier-mIoU diff (fine-coarse): {hier_boot['mean_diff']:.4f} "
              f"[{hier_boot['ci_low']:.4f}, {hier_boot['ci_high']:.4f}], "
              f"p={hier_boot['p_value']:.4f}")
    
    save_json(bootstrap_results, out_dir / "logs" / "corruption_stats.json")
    
    # 7. 打印关键结论
    print("\n" + "=" * 60)
    print("关键结论")
    print("=" * 60)
    print("\n核心指标 delta = hierIoU - mIoU:")
    for r in args.ratios:
        deltas = {row['type']: row['delta'] for row in all_results if row['r'] == r}
        print(f"  r={r}: fine={deltas['fine']:.4f}, random={deltas['random']:.4f}, coarse={deltas['coarse']:.4f}")
        if deltas['fine'] > deltas['random'] > deltas['coarse']:
            print(f"         ✓ fine > random > coarse (符合预期)")
        else:
            print(f"         ✗ 顺序异常")
    
    print("\n" + "=" * 60)
    print(f"Corruption 实验完成，结果保存到: {out_dir}")
    print("=" * 60)
    
    return {
        'summary': all_results,
        'bootstrap': bootstrap_results
    }


def plot_corruption_curves(results: list, plot_dir: Path):
    """绘制 Corruption 实验曲线"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 整理数据
    data = {'fine': {'r': [], 'miou': [], 'hier': []},
            'coarse': {'r': [], 'miou': [], 'hier': []},
            'random': {'r': [], 'miou': [], 'hier': []}}
    
    for row in results:
        t = row['type']
        data[t]['r'].append(row['r'])
        data[t]['miou'].append(row['mIoU'])
        data[t]['hier'].append(row['hierIoU'])
    
    # 图 1: mIoU 曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    for t, style in [('fine', '-o'), ('random', '-s'), ('coarse', '-^')]:
        ax.plot(data[t]['r'], data[t]['miou'], style, label=t, markersize=8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('mIoU under Different Corruption Types', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'miou_curves.png', dpi=150)
    plt.savefig(plot_dir / 'miou_curves.pdf')
    plt.close()
    
    # 图 2: Hier-mIoU 曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    for t, style in [('fine', '-o'), ('random', '-s'), ('coarse', '-^')]:
        ax.plot(data[t]['r'], data[t]['hier'], style, label=t, markersize=8)
    
    ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
    ax.set_ylabel('Hier-mIoU', fontsize=12)
    ax.set_title('Hier-mIoU under Different Corruption Types', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'hier_miou_curves.png', dpi=150)
    plt.savefig(plot_dir / 'hier_miou_curves.pdf')
    plt.close()
    
    # 图 3: 两个指标对比（并排）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (metric, title) in enumerate([('miou', 'mIoU'), ('hier', 'Hier-mIoU')]):
        ax = axes[idx]
        for t, style in [('fine', '-o'), ('random', '-s'), ('coarse', '-^')]:
            ax.plot(data[t]['r'], data[t][metric], style, label=t, markersize=8)
        
        ax.set_xlabel('Corruption Ratio (r)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} under Different Corruption Types', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'combined_curves.png', dpi=150)
    plt.savefig(plot_dir / 'combined_curves.pdf')
    plt.close()
    
    print(f"  图表已保存到: {plot_dir}")


def run_coarsen_experiment(args) -> dict:
    """运行粒度上卷实验"""
    print("=" * 60)
    print("Coarsening Experiment")
    print("=" * 60)
    
    from coarsen import load_class_paths, run_coarsening_experiment
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    Hij = load_Hij(args.Hij_path)
    class_paths = load_class_paths(Path(args.class_paths))
    image_pairs = find_image_pairs(args.gt_dir, args.pred_dir, pattern="*.png")
    if args.pred_dir is not None:
        missing_preds = [name for _, pred_path, name in image_pairs if pred_path is None]
        if missing_preds:
            preview = ", ".join(missing_preds[:5])
            if len(missing_preds) > 5:
                preview += ", ..."
            raise FileNotFoundError(
                f"在 pred_dir 中缺少 {len(missing_preds)} 张预测图，示例: {preview}"
            )
    
    print(f"找到 {len(image_pairs)} 张图片")
    print(f"深度级别: {args.depths}")
    
    # 运行实验
    results = run_coarsening_experiment(
        image_pairs, class_paths, Hij, args.depths,
        args.num_classes, args.ignore_label, args.gt_offset,
        verbose=True
    )
    
    # 保存结果
    table_data = []
    for depth, data in sorted(results.items()):
        table_data.append({
            'Depth': depth,
            'Num_Classes': data['num_coarse_classes'],
            'mIoU': data['miou'],
            'Hier_mIoU': data['hier_miou']
        })
    
    save_csv(table_data, out_dir / "tables" / "coarsen_summary.csv")
    save_latex_table(
        table_data, out_dir / "tables" / "coarsen_summary.tex",
        caption="Coarsening Experiment: mIoU and Hier-mIoU at Different Depths",
        label="tab:coarsen"
    )
    
    # 绘图
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    depths = [d['Depth'] for d in table_data]
    mious = [d['mIoU'] for d in table_data]
    hier_mious = [d['Hier_mIoU'] for d in table_data]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(depths, mious, '-o', label='mIoU', markersize=8)
    ax.plot(depths, hier_mious, '-s', label='Hier-mIoU', markersize=8)
    
    ax.set_xlabel('Depth (coarser → finer)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics at Different Granularity Levels', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / 'coarsen_curves.png', dpi=150)
    plt.savefig(out_dir / "plots" / 'coarsen_curves.pdf')
    plt.close()
    
    save_json(results, out_dir / "logs" / "coarsen_results.json")
    
    print("\n" + "=" * 60)
    print(f"Coarsening 实验完成，结果保存到: {out_dir}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="指标有效性验证实验套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只跑 sanity check
  python run_metric_experiments.py --mode sanity \\
      --gt_dir /path/to/ade150_val_gt \\
      --Hij_path /path/to/Hij.npy \\
      --out_dir ./output

  # 跑 corruption 全套
  python run_metric_experiments.py --mode corruption \\
      --gt_dir /path/to/ade150_val_gt \\
      --Hij_path /path/to/Hij.npy \\
      --out_dir ./output \\
      --ratios 0.05 0.10 0.20 0.30

  # 跑粒度上卷
  python run_metric_experiments.py --mode coarsen \\
      --gt_dir /path/to/ade150_val_gt \\
      --Hij_path /path/to/Hij.npy \\
      --class_paths /path/to/class_paths.json \\
      --out_dir ./output
        """
    )
    
    # 模式
    parser.add_argument("--mode", type=str, required=True,
                        choices=['sanity', 'corruption', 'coarsen', 'all'],
                        help="运行模式")
    
    # 输入路径
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="GT masks 目录")
    parser.add_argument("--pred_dir", type=Path, default=None,
                        help="Pred masks 目录（可选）")
    parser.add_argument("--Hij_path", type=Path, required=True,
                        help="Hij.npy 路径")
    parser.add_argument("--class_paths", type=Path, default=None,
                        help="类别 WordNet 路径 JSON（coarsen 模式需要）")
    
    # 输出
    parser.add_argument("--out_dir", type=Path, default=Path("./output"),
                        help="输出目录")
    
    # 参数
    parser.add_argument("--num_classes", type=int, default=150,
                        help="类别数量")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="忽略标签")
    parser.add_argument("--gt_offset", type=int, default=0,
                        help="GT 类别偏移量（如 GT 是 1-150，则设 -1）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # Corruption 参数
    parser.add_argument("--ratios", type=float, nargs='+',
                        default=[0.05, 0.10, 0.20, 0.30],
                        help="替换比例列表")
    parser.add_argument("--target_k", type=int, default=5,
                        help="Top-k/Bottom-k 的 k 值")
    parser.add_argument("--skip_generate", action="store_true",
                        help="跳过生成步骤，直接使用已有的合成预测进行评测")
    parser.add_argument("--balanced", action="store_true",
                        help="使用平衡模式，确保目标类具有相似的像素分布")
    
    # Coarsen 参数
    parser.add_argument("--depths", type=int, nargs='+',
                        default=[2, 3, 4, 5, 6],
                        help="粒度深度列表")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 保存实验元数据
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(get_experiment_meta(args), args.out_dir / "logs" / "experiment_meta.json")
    
    # 运行实验
    if args.mode == 'sanity':
        run_sanity_check(args)
    elif args.mode == 'corruption':
        run_corruption_experiment(args)
    elif args.mode == 'coarsen':
        if args.class_paths is None:
            print("错误: coarsen 模式需要 --class_paths 参数")
            return 1
        run_coarsen_experiment(args)
    elif args.mode == 'all':
        run_sanity_check(args)
        run_corruption_experiment(args)
        if args.class_paths is not None:
            run_coarsen_experiment(args)
    
    return 0


if __name__ == "__main__":
    exit(main())
