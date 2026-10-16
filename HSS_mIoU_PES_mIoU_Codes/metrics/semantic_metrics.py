"""
指标实现模块

包含：
1. mIoU（标准 IoU）
2. Hier-Soft IoU（基于 Hij 的软混淆 IoU）
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image


def load_mask(path: Path, offset: int = 0) -> np.ndarray:
    """
    加载分割 mask
    
    Args:
        path: 图片路径
        offset: 类别偏移量（如 GT 是 1-150，则 offset=-1 转为 0-149）
    
    Returns:
        H×W 的 numpy 数组，dtype=int32
    """
    img = Image.open(path)
    mask = np.array(img, dtype=np.int32)
    if offset != 0:
        # 只对非 ignore 区域做偏移
        valid = mask < 255
        mask[valid] = mask[valid] + offset
    return mask


def compute_confusion_matrix(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_label: int = 255
) -> np.ndarray:
    """
    计算混淆矩阵（高效实现）
    
    Args:
        gt: Ground truth mask (H, W)
        pred: Prediction mask (H, W)
        num_classes: 类别数量
        ignore_label: 忽略标签
    
    Returns:
        confusion_matrix: (num_classes, num_classes) 的混淆矩阵
        confusion[i, j] = 预测为 j 但真实为 i 的像素数
    """
    # 过滤 ignore 区域
    valid_mask = (gt != ignore_label) & (pred != ignore_label)
    valid_mask &= (gt >= 0) & (gt < num_classes)
    valid_mask &= (pred >= 0) & (pred < num_classes)
    
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    # 使用 bincount 高效计算
    # index = gt * num_classes + pred
    index = gt_valid.astype(np.int64) * num_classes + pred_valid.astype(np.int64)
    counts = np.bincount(index, minlength=num_classes * num_classes)
    confusion = counts.reshape(num_classes, num_classes)
    
    return confusion


def compute_miou_from_confusion(confusion: np.ndarray) -> Dict:
    """
    从混淆矩阵计算 mIoU
    
    Args:
        confusion: (C, C) 混淆矩阵，confusion[i,j] = GT=i, Pred=j 的像素数
    
    Returns:
        dict with 'per_class_iou', 'miou', 'tp', 'fp', 'fn'
    """
    num_classes = confusion.shape[0]
    
    # TP[i] = confusion[i, i]
    tp = np.diag(confusion).astype(np.float64)
    
    # FP[i] = sum_j confusion[j, i] - confusion[i, i] = 预测为 i 但实际不是 i
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    
    # FN[i] = sum_j confusion[i, j] - confusion[i, i] = 实际为 i 但预测不是 i
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    
    # IoU[i] = TP[i] / (TP[i] + FP[i] + FN[i])
    denominator = tp + fp + fn
    
    per_class_iou = np.zeros(num_classes, dtype=np.float64)
    valid_classes = denominator > 0
    per_class_iou[valid_classes] = tp[valid_classes] / denominator[valid_classes]
    
    # mIoU（只对出现的类计算）
    if valid_classes.sum() > 0:
        miou = per_class_iou[valid_classes].mean()
    else:
        miou = 0.0
    
    return {
        'per_class_iou': per_class_iou,
        'miou': miou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'valid_classes': valid_classes,
        'num_valid_classes': int(valid_classes.sum())
    }


def compute_hier_soft_iou_from_confusion(
    confusion: np.ndarray,
    Hij: np.ndarray
) -> Dict:
    """
    从混淆矩阵计算 Hier-Soft IoU
    
    核心思想：用 Hij 对混淆计数进行软加权
    
    对于类别 i 的 Soft IoU：
    - Soft_TP[i] = sum_j H[i,j] * confusion[i,j]  (GT=i 的像素的软匹配分数)
    - Soft_FP[i] = sum_j (1 - H[i,j]) * confusion[j,i]  (预测为 i 但不够匹配)
    - Soft_FN[i] = sum_j (1 - H[i,j]) * confusion[i,j]  (GT=i 但预测不够匹配)
    
    这样设计保证：
    1. 当 Hij = I 时，退化为标准 mIoU
    2. TP/FP/FN 非负
    3. IoU 值域 [0, 1]
    
    Args:
        confusion: (C, C) 混淆矩阵
        Hij: (C, C) 层级相似度矩阵，Hij[i,j] 表示类 i 和类 j 的相似度
    
    Returns:
        dict with 'per_class_iou', 'hier_miou', 'soft_tp', 'soft_fp', 'soft_fn'
    """
    num_classes = confusion.shape[0]
    confusion = confusion.astype(np.float64)
    Hij = Hij.astype(np.float64)
    
    # Soft TP: 对于 GT=i 的像素，根据预测 j 与 i 的相似度给予奖励
    # soft_tp[i] = sum_j H[i,j] * confusion[i,j]
    soft_tp = np.einsum('ij,ij->i', Hij, confusion)
    
    # 总像素数
    # gt_count[i] = sum_j confusion[i,j] = GT=i 的总像素数
    gt_count = confusion.sum(axis=1)
    # pred_count[i] = sum_j confusion[j,i] = 预测为 i 的总像素数
    pred_count = confusion.sum(axis=0)
    
    # Soft FP: 预测为 i 的像素中，根据 GT j 与 i 的不相似度计算惩罚
    # soft_fp[i] = sum_j (1 - H[j,i]) * confusion[j,i]
    # 即：预测为 i 但实际 GT=j，惩罚 = 1 - H[j,i]
    soft_fp = np.einsum('ji,ji->i', 1.0 - Hij, confusion)
    
    # Soft FN: GT=i 的像素中，根据预测 j 与 i 的不相似度计算惩罚
    # soft_fn[i] = sum_j (1 - H[i,j]) * confusion[i,j]
    soft_fn = np.einsum('ij,ij->i', 1.0 - Hij, confusion)
    
    # Soft IoU[i] = soft_tp[i] / (soft_tp[i] + soft_fp[i] + soft_fn[i])
    denominator = soft_tp + soft_fp + soft_fn
    
    per_class_iou = np.zeros(num_classes, dtype=np.float64)
    valid_classes = denominator > 0
    per_class_iou[valid_classes] = soft_tp[valid_classes] / denominator[valid_classes]
    
    # mean IoU
    if valid_classes.sum() > 0:
        hier_miou = per_class_iou[valid_classes].mean()
    else:
        hier_miou = 0.0
    
    return {
        'per_class_iou': per_class_iou,
        'hier_miou': hier_miou,
        'soft_tp': soft_tp,
        'soft_fp': soft_fp,
        'soft_fn': soft_fn,
        'valid_classes': valid_classes,
        'num_valid_classes': int(valid_classes.sum())
    }


def evaluate_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    Hij: np.ndarray,
    num_classes: int,
    ignore_label: int = 255
) -> Dict:
    """
    统一接口：计算所有指标
    
    Args:
        gt: Ground truth mask (H, W)
        pred: Prediction mask (H, W)  
        Hij: 层级相似度矩阵 (C, C)
        num_classes: 类别数量
        ignore_label: 忽略标签
    
    Returns:
        包含 mIoU 和 Hier-Soft IoU 的字典
    """
    confusion = compute_confusion_matrix(gt, pred, num_classes, ignore_label)
    
    miou_result = compute_miou_from_confusion(confusion)
    hier_result = compute_hier_soft_iou_from_confusion(confusion, Hij)
    
    return {
        'confusion': confusion,
        'miou': miou_result['miou'],
        'hier_miou': hier_result['hier_miou'],
        'per_class_miou': miou_result['per_class_iou'],
        'per_class_hier_miou': hier_result['per_class_iou'],
        'tp': miou_result['tp'],
        'fp': miou_result['fp'],
        'fn': miou_result['fn'],
        'soft_tp': hier_result['soft_tp'],
        'soft_fp': hier_result['soft_fp'],
        'soft_fn': hier_result['soft_fn'],
        'num_valid_classes': miou_result['num_valid_classes']
    }


def accumulate_confusion_matrices(confusion_list: List[np.ndarray]) -> np.ndarray:
    """累加多张图的混淆矩阵"""
    total = confusion_list[0].copy()
    for conf in confusion_list[1:]:
        total += conf
    return total


def evaluate_dataset(
    image_pairs: List[Tuple],
    Hij: np.ndarray,
    num_classes: int,
    ignore_label: int = 255,
    gt_offset: int = 0,
    pred_offset: int = 0,
    use_gt_as_pred: bool = False,
    verbose: bool = True
) -> Dict:
    """
    评估整个数据集
    
    Args:
        image_pairs: [(gt_path, pred_path, name), ...]
        Hij: 层级相似度矩阵
        num_classes: 类别数量
        ignore_label: 忽略标签
        gt_offset: GT 偏移量
        pred_offset: Pred 偏移量
        use_gt_as_pred: 使用 GT 作为 Pred（用于 sanity check）
        verbose: 是否打印进度
    
    Returns:
        包含整体指标的字典
    """
    from tqdm import tqdm
    
    confusion_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_results = []
    
    iterator = tqdm(image_pairs, desc="Evaluating") if verbose else image_pairs
    
    for gt_path, pred_path, name in iterator:
        gt = load_mask(Path(gt_path), offset=gt_offset)
        
        if use_gt_as_pred:
            pred = gt.copy()
        elif pred_path is not None:
            pred_path = Path(pred_path)
            if pred_path.suffix == '.npy':
                pred = np.load(pred_path).astype(np.int32)
            else:
                pred = load_mask(pred_path, offset=pred_offset)
        else:
            # 如果没有 pred，跳过
            continue
        
        confusion = compute_confusion_matrix(gt, pred, num_classes, ignore_label)
        confusion_total += confusion
        
        # 单图指标
        miou_result = compute_miou_from_confusion(confusion)
        hier_result = compute_hier_soft_iou_from_confusion(confusion, Hij)
        
        per_image_results.append({
            'name': name,
            'miou': miou_result['miou'],
            'hier_miou': hier_result['hier_miou'],
            'num_valid_classes': miou_result['num_valid_classes']
        })
    
    # 全局指标
    miou_result = compute_miou_from_confusion(confusion_total)
    hier_result = compute_hier_soft_iou_from_confusion(confusion_total, Hij)
    
    return {
        'confusion': confusion_total,
        'miou': miou_result['miou'],
        'hier_miou': hier_result['hier_miou'],
        'per_class_miou': miou_result['per_class_iou'],
        'per_class_hier_miou': hier_result['per_class_iou'],
        'tp': miou_result['tp'],
        'fp': miou_result['fp'],
        'fn': miou_result['fn'],
        'soft_tp': hier_result['soft_tp'],
        'soft_fp': hier_result['soft_fp'],
        'soft_fn': hier_result['soft_fn'],
        'num_valid_classes': miou_result['num_valid_classes'],
        'per_image_results': per_image_results,
        'num_images': len(per_image_results)
    }