#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context-conditioned Hierarchical Metrics

核心原则：所有 mIoU / hierIoU 计算统一调用 metrics.py 的标准函数，
确保口径与 real_ranking 完全一致。
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))

from semantic_metrics import (
    compute_confusion_matrix,
    compute_miou_from_confusion,
    compute_hier_soft_iou_from_confusion
)


# ============================================================================
# 基础工具
# ============================================================================
def load_mask_safe(path, offset: int = 0) -> np.ndarray:
    """安全加载 mask，处理多通道情况"""
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    if offset != 0:
        mask = img != 255
        img = img.astype(np.int32)
        img[mask] = img[mask] - offset
    return img.astype(np.int32)


def get_context_set(gt: np.ndarray, ignore_label: int = 255, num_classes: int = 150) -> Set[int]:
    """获取图像的上下文类别集合 C(x)"""
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    unique_classes = np.unique(gt)
    context_set = set()
    for c in unique_classes:
        if c != ignore_label and 0 <= c < num_classes:
            context_set.add(int(c))
    return context_set


# ============================================================================
# Context 支持度与缩放因子
# ============================================================================
def compute_context_support(
    H: np.ndarray,
    context_set: Set[int],
    num_classes: int = 150,
    support_mode: str = 'penalty_only'
) -> np.ndarray:
    """计算每个类的情景支持度 sup(j|x)"""
    sup = np.zeros(num_classes, dtype=np.float64)
    if not context_set:
        return sup
    context_list = list(context_set)
    for j in range(num_classes):
        if support_mode == 'exclude_diag':
            valid_k = [k for k in context_list if k != j]
            if valid_k:
                sup[j] = np.max(H[j, valid_k])
            else:
                sup[j] = 0.0
        else:  # penalty_only
            sup[j] = np.max(H[j, context_list])
    return sup


def compute_context_scaling(sup: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """计算情景缩放因子 g(j|x) = (1 - beta) + beta * sup(j|x)"""
    return (1.0 - beta) + beta * sup


def compute_context_similarity_matrix(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """计算情景条件化相似度矩阵 H^{ctx}_{ij}(x) = H_{ij} * g(j|x)"""
    return H * g[np.newaxis, :]


# ============================================================================
# 核心评估函数
# ============================================================================
def evaluate_image_with_context(
    gt: np.ndarray,
    pred: np.ndarray,
    H: np.ndarray,
    beta: float = 0.5,
    num_classes: int = 150,
    ignore_label: int = 255,
    support_mode: str = 'penalty_only'
) -> Dict:
    """
    对单张图像计算所有指标（mIoU, hierIoU_base, hierIoU_ctx）
    全部调用 metrics.py 的标准函数，保证口径一致。
    """
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    # 1. 计算混淆矩阵
    confusion = compute_confusion_matrix(gt, pred, num_classes, ignore_label)

    # 2. 获取上下文集合
    context_set = get_context_set(gt, ignore_label, num_classes)

    # 3. mIoU（metrics.py 标准函数）
    miou_result = compute_miou_from_confusion(confusion)
    miou = miou_result['miou']

    # 4. base hierIoU（metrics.py 标准函数，传入原始 H）
    hier_base_result = compute_hier_soft_iou_from_confusion(confusion, H)
    hier_base = hier_base_result['hier_miou']

    # 5. ctx hierIoU
    if beta == 0:
        hier_ctx = hier_base
        sup = np.ones(num_classes)
        g = np.ones(num_classes)
    else:
        sup = compute_context_support(H, context_set, num_classes, support_mode)
        g = compute_context_scaling(sup, beta)
        H_ctx = compute_context_similarity_matrix(H, g)
        hier_ctx_result = compute_hier_soft_iou_from_confusion(confusion, H_ctx)
        hier_ctx = hier_ctx_result['hier_miou']

    return {
        'miou': miou,
        'hier_base': hier_base,
        'hier_ctx': hier_ctx,
        'delta_base': hier_base - miou,
        'delta_ctx': hier_ctx - miou,
        'context_size': len(context_set),
        'beta': beta,
        'support_mode': support_mode,
        'mean_g_supported': float(np.mean([g[j] for j in context_set])) if context_set else 1.0,
        'mean_g_unsupported': float(np.mean([g[j] for j in range(num_classes) if j not in context_set])) if len(context_set) < num_classes else 0.0
    }


def compute_context_hier_iou_from_confusion(
    confusion: np.ndarray,
    H: np.ndarray,
    context_set: Set[int],
    beta: float = 0.5,
    support_mode: str = 'penalty_only'
) -> Dict:
    """计算情景条件化的层级 IoU（使用 metrics.py 标准函数）"""
    num_classes = confusion.shape[0]
    sup = compute_context_support(H, context_set, num_classes, support_mode)
    g = compute_context_scaling(sup, beta)
    H_ctx = compute_context_similarity_matrix(H, g)
    result = compute_hier_soft_iou_from_confusion(confusion, H_ctx)
    return {
        'hier_miou_ctx': result['hier_miou'],
        'per_class_iou_ctx': result['per_class_iou'],
        'support': sup,
        'scaling': g,
        'beta': beta,
        'context_size': len(context_set),
        'support_mode': support_mode
    }


def evaluate_dataset_with_context(
    image_pairs: List[Tuple],
    H: np.ndarray,
    beta: float = 0.5,
    num_classes: int = 150,
    ignore_label: int = 255,
    gt_offset: int = 0,
    support_mode: str = 'penalty_only',
    verbose: bool = True
) -> Dict:
    """对整个数据集计算情景条件化指标"""
    from tqdm import tqdm

    per_image_results = []
    iterator = tqdm(image_pairs, desc=f"Evaluating") if verbose else image_pairs

    for gt_path, pred_path, name in iterator:
        gt = load_mask_safe(gt_path, offset=gt_offset)
        pred = load_mask_safe(pred_path)
        result = evaluate_image_with_context(
            gt, pred, H, beta, num_classes, ignore_label, support_mode
        )
        result['name'] = name
        per_image_results.append(result)

    mious = [r['miou'] for r in per_image_results]
    hier_bases = [r['hier_base'] for r in per_image_results]
    hier_ctxs = [r['hier_ctx'] for r in per_image_results]
    mean_g_sup = [r['mean_g_supported'] for r in per_image_results]
    mean_g_unsup = [r['mean_g_unsupported'] for r in per_image_results]

    return {
        'mean_miou': float(np.mean(mious)),
        'mean_hier_base': float(np.mean(hier_bases)),
        'mean_hier_ctx': float(np.mean(hier_ctxs)),
        'mean_delta_base': float(np.mean(hier_bases)) - float(np.mean(mious)),
        'mean_delta_ctx': float(np.mean(hier_ctxs)) - float(np.mean(mious)),
        'mean_g_supported': float(np.mean(mean_g_sup)),
        'mean_g_unsupported': float(np.mean(mean_g_unsup)),
        'per_image': per_image_results,
        'beta': beta,
        'support_mode': support_mode
    }


# ============================================================================
# Supported vs Unsupported Corruption
# ============================================================================
def compute_supported_unsupported_targets(
    H: np.ndarray,
    context_set: Set[int],
    sim_range: Tuple[float, float] = (0.3, 0.7),
    seed: int = 42
) -> Dict[int, Dict[str, Optional[int]]]:
    """为每个类计算 supported 和 unsupported 替换目标"""
    rng = np.random.default_rng(seed)
    num_classes = H.shape[0]
    low, high = sim_range
    targets = {}
    for i in range(num_classes):
        targets[i] = {'supported': None, 'unsupported': None,
                      'supported_sim': None, 'unsupported_sim': None}
        supported_candidates = []
        for j in context_set:
            if j != i and low <= H[i, j] <= high:
                supported_candidates.append((j, H[i, j]))
        if supported_candidates:
            idx = rng.choice(len(supported_candidates))
            targets[i]['supported'] = int(supported_candidates[idx][0])
            targets[i]['supported_sim'] = float(supported_candidates[idx][1])
        unsupported_candidates = []
        for j in range(num_classes):
            if j not in context_set and j != i and low <= H[i, j] <= high:
                unsupported_candidates.append((j, H[i, j]))
        if unsupported_candidates:
            idx = rng.choice(len(unsupported_candidates))
            targets[i]['unsupported'] = int(unsupported_candidates[idx][0])
            targets[i]['unsupported_sim'] = float(unsupported_candidates[idx][1])
    return targets


def compute_fairness_matched_targets(
    H: np.ndarray,
    context_set: Set[int],
    sim_range: Tuple[float, float] = (0.3, 0.7),
    num_bins: int = 5,
    seed: int = 42
) -> Dict[int, Dict[str, Optional[int]]]:
    """公平匹配版：确保 supported 和 unsupported 的相似度分布接近"""
    rng = np.random.default_rng(seed)
    num_classes = H.shape[0]
    low, high = sim_range
    targets = {}
    for i in range(num_classes):
        targets[i] = {'supported': None, 'unsupported': None,
                      'supported_sim': None, 'unsupported_sim': None}
        sup_by_bin = [[] for _ in range(num_bins)]
        unsup_by_bin = [[] for _ in range(num_bins)]
        for j in range(num_classes):
            if j == i:
                continue
            sim = H[i, j]
            if sim < low or sim > high:
                continue
            bin_idx = min(int((sim - low) / (high - low) * num_bins), num_bins - 1)
            if j in context_set:
                sup_by_bin[bin_idx].append((j, sim))
            else:
                unsup_by_bin[bin_idx].append((j, sim))
        for bin_idx in range(num_bins):
            if sup_by_bin[bin_idx] and unsup_by_bin[bin_idx]:
                sup_choice = sup_by_bin[bin_idx][rng.choice(len(sup_by_bin[bin_idx]))]
                unsup_choice = unsup_by_bin[bin_idx][rng.choice(len(unsup_by_bin[bin_idx]))]
                targets[i]['supported'] = int(sup_choice[0])
                targets[i]['supported_sim'] = float(sup_choice[1])
                targets[i]['unsupported'] = int(unsup_choice[0])
                targets[i]['unsupported_sim'] = float(unsup_choice[1])
                break
    return targets


def generate_context_corruption(
    gt: np.ndarray,
    targets: Dict[int, Dict[str, Optional[int]]],
    corruption_type: str,
    ratio: float,
    seed: int = 42,
    ignore_label: int = 255,
    num_classes: int = 150
) -> Tuple[np.ndarray, Dict]:
    """生成情景条件化的 corruption 预测"""
    rng = np.random.default_rng(seed)
    pred = gt.copy()
    stats = {
        'total_replaced': 0,
        'skipped_no_target': 0,
        'per_class_replaced': defaultdict(int),
        'new_classes_introduced': set(),
        'replacement_sims': []
    }
    for c in range(num_classes):
        target_info = targets.get(c, {})
        target = target_info.get(corruption_type)
        target_sim = target_info.get(f'{corruption_type}_sim')
        if target is None:
            stats['skipped_no_target'] += 1
            continue
        class_mask = (gt == c)
        class_pixels = np.where(class_mask.ravel())[0]
        if len(class_pixels) == 0:
            continue
        n_replace = max(1, int(len(class_pixels) * ratio))
        n_replace = min(n_replace, len(class_pixels))
        replace_indices = rng.choice(class_pixels, size=n_replace, replace=False)
        pred_flat = pred.ravel()
        pred_flat[replace_indices] = target
        pred = pred_flat.reshape(gt.shape)
        stats['total_replaced'] += n_replace
        stats['per_class_replaced'][c] = n_replace
        stats['new_classes_introduced'].add(target)
        if target_sim is not None:
            stats['replacement_sims'].extend([target_sim] * n_replace)
    stats['new_classes_introduced'] = list(stats['new_classes_introduced'])
    stats['num_new_classes'] = len(stats['new_classes_introduced'])
    stats['mean_replacement_sim'] = float(np.mean(stats['replacement_sims'])) if stats['replacement_sims'] else 0.0
    return pred, stats


# ============================================================================
# Unsupported Error Ratio
# ============================================================================
def compute_unsupported_error_ratio(
    gt: np.ndarray,
    pred: np.ndarray,
    ignore_label: int = 255,
    num_classes: int = 150
) -> Dict:
    """计算 unsupported error ratio"""
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    context_set = get_context_set(gt, ignore_label, num_classes)
    valid = (gt != ignore_label) & (gt >= 0) & (gt < num_classes)
    valid &= (pred >= 0) & (pred < num_classes)
    error_mask = (gt != pred) & valid
    total_errors = error_mask.sum()
    if total_errors == 0:
        return {
            'unsupported_ratio': 0.0,
            'total_errors': 0,
            'unsupported_errors': 0,
            'context_size': len(context_set)
        }
    pred_at_errors = pred[error_mask]
    unsupported_errors = sum(1 for p in pred_at_errors if p not in context_set)
    return {
        'unsupported_ratio': float(unsupported_errors / total_errors),
        'total_errors': int(total_errors),
        'unsupported_errors': int(unsupported_errors),
        'context_size': len(context_set)
    }


# ============================================================================
# 单元测试函数
# ============================================================================
def assert_ctx_reduces_to_base(
    image_pairs: List[Tuple],
    H: np.ndarray,
    num_samples: int = 20,
    num_classes: int = 150,
    ignore_label: int = 255,
    gt_offset: int = 0,
    tolerance: float = 1e-10
) -> bool:
    """验证 beta=0 时 ctx 严格退化为 base"""
    import random
    samples = random.sample(image_pairs, min(num_samples, len(image_pairs)))
    max_diff = 0.0
    max_diff_name = None
    for gt_path, pred_path, name in samples:
        gt = load_mask_safe(gt_path, offset=gt_offset)
        pred = load_mask_safe(pred_path)
        result = evaluate_image_with_context(
            gt, pred, H, beta=0.0, num_classes=num_classes,
            ignore_label=ignore_label, support_mode='penalty_only'
        )
        diff = abs(result['hier_ctx'] - result['hier_base'])
        if diff > max_diff:
            max_diff = diff
            max_diff_name = name
    if max_diff > tolerance:
        raise AssertionError(
            f"beta=0 退化测试失败！最大差异: {max_diff:.2e} (图像: {max_diff_name})"
        )
    print(f"✓ beta=0 退化测试通过，最大差异: {max_diff:.2e}")
    return True


def assert_support_mode_difference(
    image_pairs: List[Tuple],
    H: np.ndarray,
    beta: float = 0.5,
    num_samples: int = 20,
    num_classes: int = 150,
    ignore_label: int = 255,
    gt_offset: int = 0
) -> Dict:
    """验证两种 support_mode 的差异"""
    import random
    samples = random.sample(image_pairs, min(num_samples, len(image_pairs)))
    results = {
        'penalty_only': {'hier_ctx': [], 'mean_g_sup': [], 'mean_g_unsup': []},
        'exclude_diag': {'hier_ctx': [], 'mean_g_sup': [], 'mean_g_unsup': []}
    }
    for gt_path, pred_path, name in samples:
        gt = load_mask_safe(gt_path, offset=gt_offset)
        pred = load_mask_safe(pred_path)
        for mode in ['penalty_only', 'exclude_diag']:
            result = evaluate_image_with_context(
                gt, pred, H, beta=beta, num_classes=num_classes,
                ignore_label=ignore_label, support_mode=mode
            )
            results[mode]['hier_ctx'].append(result['hier_ctx'])
            results[mode]['mean_g_sup'].append(result['mean_g_supported'])
            results[mode]['mean_g_unsup'].append(result['mean_g_unsupported'])
    summary = {}
    for mode in ['penalty_only', 'exclude_diag']:
        summary[mode] = {
            'mean_hier_ctx': float(np.mean(results[mode]['hier_ctx'])),
            'mean_g_supported': float(np.mean(results[mode]['mean_g_sup'])),
            'mean_g_unsupported': float(np.mean(results[mode]['mean_g_unsup']))
        }
    print(f"\nSupport mode 对比 (beta={beta}):")
    print(f"  penalty_only: hier_ctx={summary['penalty_only']['mean_hier_ctx']:.4f}, "
          f"g_sup={summary['penalty_only']['mean_g_supported']:.4f}, "
          f"g_unsup={summary['penalty_only']['mean_g_unsupported']:.4f}")
    print(f"  exclude_diag: hier_ctx={summary['exclude_diag']['mean_hier_ctx']:.4f}, "
          f"g_sup={summary['exclude_diag']['mean_g_supported']:.4f}, "
          f"g_unsup={summary['exclude_diag']['mean_g_unsupported']:.4f}")
    return summary
