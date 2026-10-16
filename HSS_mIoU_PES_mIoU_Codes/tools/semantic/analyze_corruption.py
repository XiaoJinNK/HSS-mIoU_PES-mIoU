#!/usr/bin/env python3
"""
分析 corruption 实验结果，诊断问题
"""

import numpy as np
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

from io_utils import load_Hij, load_mask, find_image_pairs


def analyze_targets(Hij_path: Path, k: int = 5):
    """分析替换目标的合理性"""
    Hij = load_Hij(Hij_path)
    num_classes = Hij.shape[0]
    
    print("=" * 60)
    print("分析替换目标")
    print("=" * 60)
    
    # 计算每个类的 fine/coarse target
    for i in range(min(10, num_classes)):  # 只看前 10 个类
        other_classes = np.array([j for j in range(num_classes) if j != i])
        other_sims = Hij[i, other_classes]
        
        # Top-k (fine)
        top_k_idx = np.argsort(other_sims)[-k:]
        fine_candidates = other_classes[top_k_idx]
        fine_sims = other_sims[top_k_idx]
        
        # Bottom-k (coarse)
        bottom_k_idx = np.argsort(other_sims)[:k]
        coarse_candidates = other_classes[bottom_k_idx]
        coarse_sims = other_sims[bottom_k_idx]
        
        print(f"\n类别 {i}:")
        print(f"  Fine candidates (最相似):  {fine_candidates} → H = {fine_sims}")
        print(f"  Coarse candidates (最不相似): {coarse_candidates} → H = {coarse_sims}")


def analyze_class_distribution(gt_dir: Path, num_images: int = 100):
    """分析类别分布"""
    print("\n" + "=" * 60)
    print("分析类别分布")
    print("=" * 60)
    
    image_pairs = find_image_pairs(gt_dir, pattern="*.png")[:num_images]
    
    total_pixels = np.zeros(150, dtype=np.int64)
    
    for gt_path, _, name in image_pairs:
        gt = load_mask(gt_path)
        for c in range(150):
            total_pixels[c] += (gt == c).sum()
    
    # 排序
    sorted_idx = np.argsort(total_pixels)[::-1]
    
    print("\nTop-20 类别（按像素数）:")
    total = total_pixels.sum()
    cumsum = 0
    for rank, c in enumerate(sorted_idx[:20]):
        pct = total_pixels[c] / total * 100
        cumsum += pct
        print(f"  {rank+1}. 类别 {c}: {total_pixels[c]:,} 像素 ({pct:.2f}%, 累计 {cumsum:.1f}%)")
    
    print("\nBottom-20 类别（按像素数）:")
    for rank, c in enumerate(sorted_idx[-20:][::-1]):
        pct = total_pixels[c] / total * 100
        print(f"  {150-rank}. 类别 {c}: {total_pixels[c]:,} 像素 ({pct:.4f}%)")
    
    return total_pixels, sorted_idx


def verify_actual_corruption(
    gt_dir: Path, 
    synth_dir: Path, 
    ratio: float = 0.1,
    num_images: int = 10
):
    """验证实际替换比例"""
    print("\n" + "=" * 60)
    print(f"验证实际替换比例 (r={ratio})")
    print("=" * 60)
    
    r_str = f"r{int(ratio*100):02d}"
    image_pairs = find_image_pairs(gt_dir, pattern="*.png")[:num_images]
    
    for corrupt_type in ['fine', 'coarse', 'random']:
        pred_dir = synth_dir / corrupt_type / r_str
        
        total_gt_pixels = 0
        total_changed = 0
        per_class_changes = np.zeros(150, dtype=np.int64)
        per_class_totals = np.zeros(150, dtype=np.int64)
        
        for gt_path, _, name in image_pairs:
            gt = load_mask(gt_path)
            pred = load_mask(pred_dir / f"{name}.png")
            
            valid = (gt != 255) & (gt >= 0) & (gt < 150)
            total_gt_pixels += valid.sum()
            
            changed = (gt != pred) & valid
            total_changed += changed.sum()
            
            for c in range(150):
                mask_c = (gt == c) & valid
                per_class_totals[c] += mask_c.sum()
                per_class_changes[c] += (changed & mask_c).sum()
        
        actual_ratio = total_changed / total_gt_pixels
        print(f"\n{corrupt_type}:")
        print(f"  期望比例: {ratio:.2%}")
        print(f"  实际比例: {actual_ratio:.2%}")
        print(f"  总像素: {total_gt_pixels:,}, 改变: {total_changed:,}")
        
        # 检查每类的替换情况
        print(f"  每类替换比例 (前10类):")
        for c in range(10):
            if per_class_totals[c] > 0:
                class_ratio = per_class_changes[c] / per_class_totals[c]
                print(f"    类别 {c}: {class_ratio:.2%} ({per_class_changes[c]}/{per_class_totals[c]})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Hij_path", type=Path, default=Path("similarity_output/ade150_hybrid_inv/S.npy"))
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--synth_dir", type=Path, default=Path("output/metric_experiments/synth_preds"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    analyze_targets(args.Hij_path, args.k)
    analyze_class_distribution(args.gt_dir)
    
    if args.synth_dir.exists():
        verify_actual_corruption(args.gt_dir, args.synth_dir, ratio=0.1)


if __name__ == "__main__":
    main()