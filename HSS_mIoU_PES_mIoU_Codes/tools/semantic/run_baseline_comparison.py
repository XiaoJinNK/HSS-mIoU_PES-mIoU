#!/usr/bin/env python3
"""
对比 Hybrid vs 算术平均/几何平均基线
证明 Hybrid 不是简单的 PathSim 和 WuP 的插值
"""
import numpy as np
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

from semantic_metrics import (
    compute_confusion_matrix,
    compute_miou_from_confusion,
    compute_hier_soft_iou_from_confusion
)
from corrupt import (
    compute_corruption_targets,
    generate_synthetic_predictions,
    load_mask
)
from io_utils import find_image_pairs, save_json


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid vs 插值基线对比")
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--hybrid_path", type=Path, required=True)
    parser.add_argument("--pathsim_path", type=Path, required=True)
    parser.add_argument("--wup_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("output/baseline_comparison"))
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--num_classes", type=int, default=150)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_k", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始矩阵
    S_hybrid = np.load(args.hybrid_path).astype(np.float64)
    S_pathsim = np.load(args.pathsim_path).astype(np.float64)
    S_wup = np.load(args.wup_path).astype(np.float64)

    # 构造基线
    S_arith = (S_pathsim + S_wup) / 2.0
    S_geom = np.sqrt(S_pathsim * S_wup)

    # 验证 Hybrid 是否严格等于几何平均
    diff = np.abs(S_hybrid - S_geom)
    print(f"Hybrid vs Geom: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    print(f"  如果 max_diff ≈ 0，说明 Hybrid 确实是几何平均")
    print(f"  如果 max_diff > 0，说明 Hybrid 有额外处理")

    matrices = {
        'hybrid': S_hybrid,
        'pathsim': S_pathsim,
        'wup': S_wup,
        'arith_mean': S_arith,
        'geom_mean': S_geom
    }

    # 分析矩阵统计特性
    print("\n=== 矩阵统计对比 ===")
    print(f"{'Matrix':<15} {'Mean':>8} {'Std':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
    for name, S in matrices.items():
        off_diag = S[np.triu_indices(S.shape[0], k=1)]
        print(f"{name:<15} {off_diag.mean():>8.4f} {off_diag.std():>8.4f} "
              f"{np.percentile(off_diag, 10):>8.4f} {np.percentile(off_diag, 50):>8.4f} "
              f"{np.percentile(off_diag, 90):>8.4f}")

    # 加载图片
    image_pairs = find_image_pairs(args.gt_dir, pattern="*.png")
    print(f"\n找到 {len(image_pairs)} 张图片")

    all_results = []

    for sim_name, S in matrices.items():
        print(f"\n{'='*60}")
        print(f"处理矩阵: {sim_name}")
        print(f"{'='*60}")

        # 用该矩阵自己选 targets
        targets = compute_corruption_targets(S, k=args.target_k, seed=args.seed)

        for r in args.ratios:
            synth_dir = out_dir / "synth_preds" / sim_name
            generate_synthetic_predictions(
                image_pairs, targets, [r], synth_dir,
                args.num_classes, args.ignore_label, 0, args.seed
            )

            for corrupt_type in ['fine', 'coarse', 'random']:
                r_str = f"r{int(r*100):02d}"
                pred_dir = synth_dir / corrupt_type / r_str

                if not pred_dir.exists():
                    continue

                # 计算 mIoU 和 hierIoU
                total_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
                for gt_path, _, name in image_pairs:
                    pred_path = pred_dir / f"{name}.png"
                    if not pred_path.exists():
                        continue
                    gt = load_mask(gt_path)
                    pred = load_mask(pred_path)
                    total_confusion += compute_confusion_matrix(gt, pred, args.num_classes, args.ignore_label)

                miou_result = compute_miou_from_confusion(total_confusion)
                hier_result = compute_hier_soft_iou_from_confusion(total_confusion, S)

                result = {
                    'similarity': sim_name,
                    'r': r,
                    'type': corrupt_type,
                    'mIoU': miou_result['miou'],
                    'hierIoU': hier_result['hier_miou'],
                    'delta': hier_result['hier_miou'] - miou_result['miou']
                }
                all_results.append(result)
                print(f"  {sim_name} r={r} {corrupt_type}: "
                      f"mIoU={result['mIoU']:.4f}, hierIoU={result['hierIoU']:.4f}, "
                      f"Δ={result['delta']:.4f}")

    # Far-credit 分析
    print("\n=== Far-credit 对比 ===")
    far_credit_results = []
    q = 0.1  # 10th percentile

    for sim_name, S in matrices.items():
        targets = compute_corruption_targets(S, k=args.target_k, seed=args.seed)
        for r in args.ratios:
            r_str = f"r{int(r*100):02d}"
            for corrupt_type in ['fine', 'coarse']:
                synth_dir = out_dir / "synth_preds" / sim_name
                pred_dir = synth_dir / corrupt_type / r_str
                if not pred_dir.exists():
                    continue

                all_sims = []
                for gt_path, _, name in image_pairs:
                    pred_path = pred_dir / f"{name}.png"
                    if not pred_path.exists():
                        continue
                    gt = load_mask(gt_path)
                    pred = load_mask(pred_path)
                    valid = (gt != args.ignore_label) & (gt >= 0) & (gt < args.num_classes)
                    changed = (gt != pred) & valid
                    if changed.sum() == 0:
                        continue
                    gt_c = gt[changed]
                    pred_c = pred[changed]
                    for g, p in zip(gt_c, pred_c):
                        if 0 <= g < args.num_classes and 0 <= p < args.num_classes:
                            all_sims.append(S[g, p])

                if not all_sims:
                    continue
                all_sims = np.array(all_sims)
                far_threshold = np.percentile(all_sims, q * 100)
                near_threshold = np.percentile(all_sims, (1 - q) * 100)
                far_mask = all_sims <= far_threshold
                near_mask = all_sims >= near_threshold
                far_credit = float(all_sims[far_mask].mean()) if far_mask.sum() > 0 else 0
                near_credit = float(all_sims[near_mask].mean()) if near_mask.sum() > 0 else 0

                fc_result = {
                    'similarity': sim_name,
                    'r': r,
                    'type': corrupt_type,
                    'far_credit': far_credit,
                    'near_credit': near_credit,
                    'near_far_gap': near_credit - far_credit,
                    'mean_sim': float(all_sims.mean())
                }
                far_credit_results.append(fc_result)

    # 保存
    save_json(all_results, out_dir / "logs" / "corruption_results.json")
    save_json(far_credit_results, out_dir / "logs" / "far_credit_results.json")

    # 打印关键对比表
    print("\n" + "=" * 80)
    print("关键对比: r=0.1, fine corruption")
    print("=" * 80)
    print(f"{'Matrix':<15} {'ΔGap':>8} {'FarCredit':>10} {'NearCredit':>11} {'NF-Gap':>8}")
    for sim_name in ['pathsim', 'geom_mean', 'hybrid', 'arith_mean', 'wup']:
        # ΔGap
        fine_delta = next((r['delta'] for r in all_results
                          if r['similarity'] == sim_name and r['r'] == 0.1 and r['type'] == 'fine'), 0)
        coarse_delta = next((r['delta'] for r in all_results
                            if r['similarity'] == sim_name and r['r'] == 0.1 and r['type'] == 'coarse'), 0)
        delta_gap = fine_delta - coarse_delta

        # Far-credit
        fc_fine = next((r for r in far_credit_results
                       if r['similarity'] == sim_name and r['r'] == 0.1 and r['type'] == 'fine'), None)
        if fc_fine:
            print(f"{sim_name:<15} {delta_gap:>8.4f} {fc_fine['far_credit']:>10.4f} "
                  f"{fc_fine['near_credit']:>11.4f} {fc_fine['near_far_gap']:>8.4f}")
        else:
            print(f"{sim_name:<15} {delta_gap:>8.4f} {'N/A':>10} {'N/A':>11} {'N/A':>8}")

    print("\n完成！结果保存到:", out_dir)


if __name__ == "__main__":
    main()