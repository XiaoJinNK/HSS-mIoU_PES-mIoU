"""
粒度上卷模块

将细粒度类别映射到 WordNet 层级中的粗粒度祖先
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


ROOT_LIKE_SYNSETS = {
    "entity.n.01",
    "physical_entity.n.01",
    "abstraction.n.06",
}


def _normalize_synset_path(path_like) -> List[str]:
    """Normalize a synset path to a root->leaf list."""
    if isinstance(path_like, str):
        path = [part.strip() for part in path_like.split("->") if part.strip()]
    else:
        path = [str(part).strip() for part in path_like if str(part).strip()]

    if not path:
        raise ValueError("Empty synset path encountered in class mapping")

    # The ADE150 canonical mapping stores paths as leaf -> ... -> root.
    # Coarsening expects root -> ... -> leaf.
    if path[-1] in ROOT_LIKE_SYNSETS and path[0] not in ROOT_LIKE_SYNSETS:
        path = list(reversed(path))

    return path


def load_class_paths(path: Path) -> Dict[int, List[str]]:
    """
    加载每个类的 WordNet 路径
    
    期望格式（JSON）:
    {
        "0": ["entity.n.01", "physical_entity.n.01", ..., "wall.n.01"],
        "1": ["entity.n.01", ..., "building.n.01"],
        ...
    }
    
    Returns:
        {class_id: [synset_path_from_root]}
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Format A:
    # {
    #   "0": ["entity.n.01", ..., "wall.n.01"],
    #   ...
    # }
    if isinstance(data, dict) and data:
        first_key = next(iter(data))
        first_val = data[first_key]

        if str(first_key).isdigit():
            return {int(k): _normalize_synset_path(v) for k, v in data.items()}

        # Format B (ADE canonical mapping):
        # {
        #   "wall": {
        #      "hypernym_path_canonical": "wall.n.01 -> ... -> entity.n.01",
        #      ...
        #   },
        #   ...
        # }
        if isinstance(first_val, dict):
            class_paths = {}
            for class_id, (_, meta) in enumerate(data.items()):
                path_like = (
                    meta.get("hypernym_path_canonical")
                    or meta.get("hypernym_path_grounded")
                    or meta.get("hypernym_path")
                )
                if path_like is None:
                    raise ValueError(
                        "Class mapping entry is missing hypernym_path_canonical / "
                        "hypernym_path_grounded / hypernym_path"
                    )
                class_paths[class_id] = _normalize_synset_path(path_like)
            return class_paths

    raise ValueError(
        f"Unsupported class path format in {path}. Expected integer-keyed paths "
        "or ADE-style name-keyed metadata entries."
    )


def get_ancestor_at_depth(path: List[str], depth: int) -> str:
    """
    获取路径中指定深度的祖先节点
    
    depth=0 是 root，depth=-1 是叶子节点
    如果 depth 超出路径长度，返回叶子节点
    """
    if depth < 0:
        depth = len(path) + depth
    if depth >= len(path):
        return path[-1]
    return path[depth]


def build_coarse_mapping(
    class_paths: Dict[int, List[str]],
    depth: int
) -> Tuple[Dict[int, int], Dict[str, int], List[str]]:
    """
    构建粗粒度映射
    
    Args:
        class_paths: {class_id: [synset_path]}
        depth: 目标深度（0=root，越大越细）
    
    Returns:
        fine_to_coarse: {fine_class_id: coarse_class_id}
        synset_to_coarse: {synset_name: coarse_class_id}
        coarse_names: [coarse_class_name, ...]
    """
    # 获取每个细粒度类在该深度的祖先
    fine_to_ancestor = {}
    for class_id, path in class_paths.items():
        ancestor = get_ancestor_at_depth(path, depth)
        fine_to_ancestor[class_id] = ancestor
    
    # 给祖先分配粗粒度 ID
    unique_ancestors = sorted(set(fine_to_ancestor.values()))
    synset_to_coarse = {s: i for i, s in enumerate(unique_ancestors)}
    
    # 构建映射
    fine_to_coarse = {
        fine_id: synset_to_coarse[ancestor]
        for fine_id, ancestor in fine_to_ancestor.items()
    }
    
    return fine_to_coarse, synset_to_coarse, unique_ancestors


def coarsen_mask(
    mask: np.ndarray,
    fine_to_coarse: Dict[int, int],
    ignore_label: int = 255
) -> np.ndarray:
    """将细粒度 mask 转换为粗粒度 mask"""
    coarse_mask = mask.copy()
    
    for fine_id, coarse_id in fine_to_coarse.items():
        coarse_mask[mask == fine_id] = coarse_id
    
    return coarse_mask


def coarsen_Hij(
    Hij: np.ndarray,
    fine_to_coarse: Dict[int, int],
    num_coarse_classes: int,
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    将细粒度 Hij 聚合为粗粒度 Hij
    
    Args:
        Hij: 细粒度相似度矩阵
        fine_to_coarse: 映射
        num_coarse_classes: 粗粒度类别数
        aggregation: 'mean' 或 'max'
    
    Returns:
        粗粒度 Hij
    """
    num_fine = Hij.shape[0]
    
    # 构建映射矩阵
    M = np.zeros((num_coarse_classes, num_fine), dtype=np.float64)
    for fine_id, coarse_id in fine_to_coarse.items():
        M[coarse_id, fine_id] = 1.0
    
    # 归一化
    M_norm = M / M.sum(axis=1, keepdims=True).clip(min=1e-10)
    
    if aggregation == 'mean':
        # H_coarse = M_norm @ Hij @ M_norm.T
        Hij_coarse = M_norm @ Hij @ M_norm.T
    else:  # max
        Hij_coarse = np.zeros((num_coarse_classes, num_coarse_classes))
        for i in range(num_coarse_classes):
            for j in range(num_coarse_classes):
                fine_i = np.where(M[i] > 0)[0]
                fine_j = np.where(M[j] > 0)[0]
                if len(fine_i) > 0 and len(fine_j) > 0:
                    Hij_coarse[i, j] = Hij[np.ix_(fine_i, fine_j)].max()
    
    return Hij_coarse


def run_coarsening_experiment(
    image_pairs: List[Tuple],
    class_paths: Dict[int, List[str]],
    Hij: np.ndarray,
    depths: List[int],
    num_classes: int,
    ignore_label: int = 255,
    gt_offset: int = 0,
    pred_offset: int = 0,
    verbose: bool = True
) -> Dict:
    """
    运行多粒度评测实验
    
    Returns:
        {depth: {'miou': ..., 'hier_miou': ..., 'num_coarse_classes': ...}}
    """
    # Support both package-style imports and direct script execution.
    try:
        from .io_utils import load_mask
        from .metrics import (
            compute_confusion_matrix,
            compute_miou_from_confusion,
            compute_hier_soft_iou_from_confusion,
        )
    except ImportError:
        from io_utils import load_mask
        from semantic_metrics import (
            compute_confusion_matrix,
            compute_miou_from_confusion,
            compute_hier_soft_iou_from_confusion,
        )
    from tqdm import tqdm
    
    results = {}
    
    for depth in depths:
        if verbose:
            print(f"\n处理深度 {depth}...")
        
        # 构建映射
        fine_to_coarse, _, coarse_names = build_coarse_mapping(class_paths, depth)
        num_coarse = len(coarse_names)
        
        # 聚合 Hij
        Hij_coarse = coarsen_Hij(Hij, fine_to_coarse, num_coarse, aggregation='mean')
        
        # 累计混淆矩阵
        confusion_total = np.zeros((num_coarse, num_coarse), dtype=np.int64)
        
        iterator = tqdm(image_pairs, desc=f"Depth {depth}") if verbose else image_pairs
        for gt_path, pred_path, name in iterator:
            gt = load_mask(gt_path, offset=gt_offset)
            
            if pred_path is not None:
                if pred_path.suffix == '.npy':
                    pred = np.load(pred_path).astype(np.int32)
                else:
                    pred = load_mask(pred_path, offset=pred_offset)
            else:
                pred = gt.copy()
            
            # 转为粗粒度
            gt_coarse = coarsen_mask(gt, fine_to_coarse, ignore_label)
            pred_coarse = coarsen_mask(pred, fine_to_coarse, ignore_label)
            
            confusion = compute_confusion_matrix(gt_coarse, pred_coarse, num_coarse, ignore_label)
            confusion_total += confusion
        
        # 计算指标
        miou_result = compute_miou_from_confusion(confusion_total)
        hier_result = compute_hier_soft_iou_from_confusion(confusion_total, Hij_coarse)
        
        results[depth] = {
            'miou': miou_result['miou'],
            'hier_miou': hier_result['hier_miou'],
            'num_coarse_classes': num_coarse,
            'coarse_names': coarse_names
        }
    
    return results
