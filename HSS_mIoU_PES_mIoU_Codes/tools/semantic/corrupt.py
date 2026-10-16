"""
合成预测生成模块

生成 fine/coarse/random 三类人工预测（controlled corruption 版）

关键保证：
- 对同一张图、同一替换比例 r：fine/coarse/random 复用完全相同的被替换像素位置
  仅改变替换目标类别（target），从而让对比只反映"语义粗细"的影响。
  
改进 v2：
- 添加 balanced 模式，确保 fine/coarse/random 的目标类具有相似的像素分布
- 这样 mIoU 在三种 corruption 下会更接近，更公平地对比 hierIoU
"""

import zlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image


def _stable_int_seed(base_seed: int, name: str, r: float) -> int:
    """
    生成一个可复现的整型 seed，避免 Python 内置 hash() 的随机化。
    """
    h = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    rr = int(round(r * 1_000_000))
    seed = (base_seed * 1000003 + h * 9176 + rr * 131) & 0x7FFFFFFF
    return int(seed)


def load_mask(path: Path, offset: int = 0, ignore_label: int = 255) -> np.ndarray:
    """
    加载分割 mask
    """
    img = Image.open(path)
    mask = np.array(img, dtype=np.int32)
    if offset != 0:
        valid = (mask != ignore_label)
        mask[valid] = mask[valid] + offset
    return mask


def save_mask(mask: np.ndarray, path: Path):
    """保存 mask 为 PNG"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(path)


def compute_corruption_targets(
    Hij: np.ndarray,
    k: int = 5,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    预计算替换目标（全局固定版）
    
    对每个类 i：
    - fine_target[i]: Hij 最高的 k 个类中随机选一个（相似类）
    - coarse_target[i]: Hij 最低的 k 个类中随机选一个（不相似类）
    - random_target[i]: 均匀随机选一个其他类
    """
    rng = np.random.default_rng(seed)
    num_classes = Hij.shape[0]

    fine_targets = np.zeros(num_classes, dtype=np.int32)
    coarse_targets = np.zeros(num_classes, dtype=np.int32)
    random_targets = np.zeros(num_classes, dtype=np.int32)
    
    # 记录相似度用于诊断
    fine_sims = np.zeros(num_classes, dtype=np.float32)
    coarse_sims = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        other_classes = np.array([j for j in range(num_classes) if j != i], dtype=np.int32)
        other_sims = Hij[i, other_classes]

        kk = min(k, other_classes.size)

        # Fine: 最相似的 k 个
        top_k_idx = np.argsort(other_sims)[-kk:]
        chosen_fine = rng.choice(top_k_idx)
        fine_targets[i] = other_classes[chosen_fine]
        fine_sims[i] = other_sims[chosen_fine]

        # Coarse: 最不相似的 k 个
        bottom_k_idx = np.argsort(other_sims)[:kk]
        chosen_coarse = rng.choice(bottom_k_idx)
        coarse_targets[i] = other_classes[chosen_coarse]
        coarse_sims[i] = other_sims[chosen_coarse]

        # Random
        random_targets[i] = rng.choice(other_classes)

    # 打印诊断信息
    print(f"\n替换目标统计:")
    print(f"  Fine targets 平均相似度: {fine_sims.mean():.4f} (期望高)")
    print(f"  Coarse targets 平均相似度: {coarse_sims.mean():.4f} (期望低)")
    print(f"  差异: {fine_sims.mean() - coarse_sims.mean():.4f}")

    return {
        "fine": fine_targets,
        "coarse": coarse_targets,
        "random": random_targets,
        "fine_sims": fine_sims,
        "coarse_sims": coarse_sims
    }


def compute_balanced_corruption_targets(
    Hij: np.ndarray,
    class_pixel_counts: np.ndarray,
    k: int = 5,
    size_ratio_range: Tuple[float, float] = (0.2, 5.0),
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    计算平衡的替换目标（考虑类别像素分布）
    
    改进：确保 fine/coarse/random 的目标类具有相似的像素分布，
    这样 mIoU 在三种 corruption 下会更接近。
    
    Args:
        Hij: (C, C) 相似度矩阵
        class_pixel_counts: (C,) 每个类的总像素数
        k: top-k / bottom-k 的 k 值
        size_ratio_range: (min_ratio, max_ratio)，目标类像素数应在源类的这个范围内
        seed: 随机种子
    
    Returns:
        dict with 'fine', 'coarse', 'random' targets
    """
    rng = np.random.default_rng(seed)
    num_classes = Hij.shape[0]
    
    fine_targets = np.zeros(num_classes, dtype=np.int32)
    coarse_targets = np.zeros(num_classes, dtype=np.int32)
    random_targets = np.zeros(num_classes, dtype=np.int32)
    
    fine_sims = np.zeros(num_classes, dtype=np.float32)
    coarse_sims = np.zeros(num_classes, dtype=np.float32)
    
    min_ratio, max_ratio = size_ratio_range
    
    for i in range(num_classes):
        other_classes = np.array([j for j in range(num_classes) if j != i], dtype=np.int32)
        other_sims = Hij[i, other_classes]
        other_counts = class_pixel_counts[other_classes]
        
        # 找到与 i 像素数相近的类
        i_count = max(class_pixel_counts[i], 1)
        size_similar = (other_counts >= i_count * min_ratio) & (other_counts <= i_count * max_ratio)
        
        # 如果相似大小的类太少，逐步放宽限制
        if size_similar.sum() < k:
            # 放宽到 0.1x ~ 10x
            size_similar = (other_counts >= i_count * 0.1) & (other_counts <= i_count * 10.0)
        if size_similar.sum() < k:
            # 如果还是太少，使用所有类
            size_similar = np.ones(len(other_classes), dtype=bool)
        
        valid_idx = np.where(size_similar)[0]
        valid_classes = other_classes[valid_idx]
        valid_sims = other_sims[valid_idx]
        
        kk = min(k, len(valid_classes))
        
        if kk > 0:
            # Fine: 在相似大小的类中选最相似的
            top_k_idx = np.argsort(valid_sims)[-kk:]
            chosen = rng.choice(top_k_idx)
            fine_targets[i] = valid_classes[chosen]
            fine_sims[i] = valid_sims[chosen]
            
            # Coarse: 在相似大小的类中选最不相似的
            bottom_k_idx = np.argsort(valid_sims)[:kk]
            chosen = rng.choice(bottom_k_idx)
            coarse_targets[i] = valid_classes[chosen]
            coarse_sims[i] = valid_sims[chosen]
            
            # Random: 在相似大小的类中随机选
            random_targets[i] = rng.choice(valid_classes)
        else:
            # Fallback: 使用全局
            fine_targets[i] = other_classes[np.argmax(other_sims)]
            fine_sims[i] = other_sims.max()
            coarse_targets[i] = other_classes[np.argmin(other_sims)]
            coarse_sims[i] = other_sims.min()
            random_targets[i] = rng.choice(other_classes)
    
    # 打印诊断信息
    print(f"\n平衡替换目标统计:")
    print(f"  Fine targets 平均相似度: {fine_sims.mean():.4f} (期望高)")
    print(f"  Coarse targets 平均相似度: {coarse_sims.mean():.4f} (期望低)")
    print(f"  差异: {fine_sims.mean() - coarse_sims.mean():.4f}")
    
    # 验证目标类大小分布
    fine_target_counts = class_pixel_counts[fine_targets]
    coarse_target_counts = class_pixel_counts[coarse_targets]
    print(f"  Fine target 平均像素数: {fine_target_counts.mean():.0f}")
    print(f"  Coarse target 平均像素数: {coarse_target_counts.mean():.0f}")
    
    return {
        "fine": fine_targets,
        "coarse": coarse_targets,
        "random": random_targets,
        "fine_sims": fine_sims,
        "coarse_sims": coarse_sims
    }


def compute_class_pixel_counts(
    image_pairs: List[Tuple],
    num_classes: int,
    ignore_label: int = 255,
    gt_offset: int = 0,
    max_images: int = 0,
    verbose: bool = True
) -> np.ndarray:
    """
    统计每个类的总像素数
    
    Args:
        image_pairs: [(gt_path, _, name), ...]
        num_classes: 类别数量
        ignore_label: 忽略标签
        gt_offset: GT 偏移量
        max_images: 最多统计多少张图（0 = 全部）
        verbose: 是否打印进度
    
    Returns:
        (num_classes,) 每个类的总像素数
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    
    if max_images > 0:
        image_pairs = image_pairs[:max_images]
    
    iterator = tqdm(image_pairs, desc="Counting class pixels") if verbose else image_pairs
    
    for gt_path, _, name in iterator:
        gt = load_mask(gt_path, offset=gt_offset, ignore_label=ignore_label)
        
        for c in range(num_classes):
            counts[c] += ((gt == c) & (gt != ignore_label)).sum()
    
    return counts


def sample_replace_map_per_class(
    gt: np.ndarray,
    r: float,
    num_classes: int,
    ignore_label: int = 255,
    seed: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """
    采样替换像素位置（按类采样）
    
    对每个类 i，随机选择 ceil(r * count_i) 个像素进行替换
    """
    rng = np.random.default_rng(seed)
    flat = gt.ravel()

    valid_mask = (flat != ignore_label) & (flat >= 0) & (flat < num_classes)
    replace_map: Dict[int, np.ndarray] = {}

    for i in range(num_classes):
        idx = np.where(valid_mask & (flat == i))[0]
        if idx.size == 0:
            continue

        n = int(np.ceil(r * idx.size))
        n = min(n, idx.size)
        if n <= 0:
            continue

        replace_map[i] = rng.choice(idx, size=n, replace=False)

    return replace_map


def apply_replace_map(
    gt: np.ndarray,
    replace_map: Dict[int, np.ndarray],
    targets: np.ndarray
) -> np.ndarray:
    """
    复用 replace_map，将对应类像素替换为 targets[i]，生成 pred。
    """
    pred = gt.copy()
    flat_pred = pred.ravel()
    for i, ridx in replace_map.items():
        flat_pred[ridx] = targets[i]
    return flat_pred.reshape(gt.shape)


def verify_corruption_ratio(
    gt: np.ndarray,
    pred: np.ndarray,
    expected_ratio: float,
    num_classes: int,
    ignore_label: int = 255,
    tolerance: float = 0.05
) -> Tuple[bool, float]:
    """
    验证替换比例是否接近期望值
    """
    valid_mask = (gt != ignore_label) & (gt >= 0) & (gt < num_classes)
    total_valid = valid_mask.sum()

    if total_valid == 0:
        return True, 0.0

    diff_mask = (gt != pred) & valid_mask
    actual_ratio = float(diff_mask.sum() / total_valid)

    is_ok = abs(actual_ratio - expected_ratio) <= tolerance
    return is_ok, actual_ratio


def corrupt_single_image(
    gt: np.ndarray,
    r: float,
    targets: np.ndarray,
    num_classes: int,
    ignore_label: int = 255,
    seed: int = None
) -> np.ndarray:
    """
    对单张图生成合成预测（简化版，用于测试）
    """
    replace_map = sample_replace_map_per_class(
        gt, r, num_classes, ignore_label, seed
    )
    return apply_replace_map(gt, replace_map, targets)


def generate_synthetic_predictions(
    image_pairs: List[Tuple],
    targets: Dict[str, np.ndarray],
    ratios: List[float],
    output_dir: Path,
    num_classes: int,
    ignore_label: int = 255,
    gt_offset: int = 0,
    seed: int = 42,
    save_format: str = "png",
    verbose: bool = True,
    controlled_same_positions: bool = True,
    ratio_tolerance: float = 0.05
) -> Dict:
    """
    生成所有合成预测（controlled corruption）

    关键：对同一张图、同一替换比例 r，fine/coarse/random 复用完全相同的被替换像素位置，
    仅改变替换目标类别（target），从而让对比只反映"语义粗细"的影响。

    Args:
        image_pairs: [(gt_path, _, name), ...]
        targets: {'fine': array, 'coarse': array, 'random': array}
        ratios: 替换比例列表
        output_dir: 输出目录
        num_classes: 类别数量
        ignore_label: 忽略标签
        gt_offset: GT 偏移量
        seed: 随机种子
        save_format: 'png' 或 'npy'
        verbose: 是否打印进度
        controlled_same_positions: True 表示三种 corruption 复用相同替换位置
        ratio_tolerance: verify_corruption_ratio 的容忍度

    Returns:
        统计信息 dict
    """
    stats = {
        "num_images": len(image_pairs),
        "ratios": ratios,
        "types": ["fine", "coarse", "random"],
        "controlled_same_positions": controlled_same_positions,
        "ratio_check": []
    }

    total_ops = len(image_pairs) * len(ratios) * 3
    pbar = tqdm(total=total_ops, desc="Generating corrupted predictions") if verbose else None

    # 预建输出目录
    for corrupt_type in ["fine", "coarse", "random"]:
        for r in ratios:
            r_str = f"r{int(r*100):02d}"
            (output_dir / corrupt_type / r_str).mkdir(parents=True, exist_ok=True)

    for r in ratios:
        r_str = f"r{int(r*100):02d}"

        for gt_path, _, name in image_pairs:
            gt = load_mask(gt_path, offset=gt_offset, ignore_label=ignore_label)

            # 同一张图同一 r 的基础 seed
            img_seed = _stable_int_seed(seed, name, r)

            # 采样替换位置：同一张图同一 r 只做一次
            if controlled_same_positions:
                replace_map = sample_replace_map_per_class(
                    gt, r, num_classes=num_classes, ignore_label=ignore_label, seed=img_seed
                )

            # 生成三种 pred：只改变 targets，不改变替换位置
            for corrupt_type in ["fine", "coarse", "random"]:
                out_subdir = output_dir / corrupt_type / r_str
                target_arr = targets[corrupt_type]

                if controlled_same_positions:
                    # 复用相同的替换位置
                    pred = apply_replace_map(gt, replace_map, target_arr)
                else:
                    # 每种类型独立采样（不推荐，仅用于对比）
                    type_seed = (img_seed + {"fine": 0, "coarse": 1, "random": 2}[corrupt_type]) & 0x7FFFFFFF
                    rmap = sample_replace_map_per_class(
                        gt, r, num_classes=num_classes, ignore_label=ignore_label, seed=type_seed
                    )
                    pred = apply_replace_map(gt, rmap, target_arr)

                # ratio check
                ok, actual = verify_corruption_ratio(
                    gt, pred, expected_ratio=r, num_classes=num_classes,
                    ignore_label=ignore_label, tolerance=ratio_tolerance
                )
                stats["ratio_check"].append({
                    "name": name, "r": r, "type": corrupt_type, "ok": ok, "actual_ratio": actual
                })

                # 保存
                if save_format == "npy":
                    np.save(out_subdir / f"{name}.npy", pred)
                else:
                    save_mask(pred, out_subdir / f"{name}.png")

                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    return stats
