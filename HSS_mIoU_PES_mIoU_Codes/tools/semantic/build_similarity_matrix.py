#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_similarity_matrix.py

从 WordNet induced graph 构建 150×150 类间相似度矩阵 S。

用法:
    python build_similarity_matrix.py \
        --edges [induced_graph_edges.tsv](http://_vscodecontentref_/4) \
        --mapping ade150_to_wn_canonical.json \
        --outdir ./output \
        --mode inv \
        --undirected

输出:
    - S.npy: 150×150 float32 相似度矩阵
    - class_order.txt: 150 行，矩阵索引对应的类名
    - S_meta.json: 元数据（WordNet 版本、参数、生成时间等）
"""

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque
from datetime import datetime

import numpy as np


# ============================================================================
# 图构建函数
# ============================================================================
def load_edges_tsv(edges_path: Path) -> List[Tuple[str, str]]:
    """
    加载边文件 (TSV 格式: child<TAB>parent)
    
    Returns:
        List of (child, parent) tuples
    """
    edges = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                child, parent = parts[0].strip(), parts[1].strip()
                edges.append((child, parent))
    return edges


def load_canonical_mapping(mapping_path: Path) -> Dict[str, Dict]:
    """
    加载 canonical mapping JSON
    
    Returns:
        Dict[label, entry_dict] (过滤掉 _metadata 等非标签条目)
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤掉元数据条目
    mapping = {}
    for key, value in data.items():
        if key.startswith('_'):
            continue
        if isinstance(value, dict):
            mapping[key] = value
    
    return mapping


def build_adjacency_list(
    edges: List[Tuple[str, str]],
    undirected: bool = True
) -> Dict[str, Set[str]]:
    """
    从边列表构建邻接表
    
    Args:
        edges: [(child, parent), ...]
        undirected: 是否将边视为无向边
    
    Returns:
        adjacency: {node: {neighbor1, neighbor2, ...}}
    """
    adjacency: Dict[str, Set[str]] = {}
    
    for child, parent in edges:
        # 确保两个节点都在邻接表中
        if child not in adjacency:
            adjacency[child] = set()
        if parent not in adjacency:
            adjacency[parent] = set()
        
        # 添加边
        adjacency[child].add(parent)
        if undirected:
            adjacency[parent].add(child)
    
    return adjacency


def get_all_nodes(adjacency: Dict[str, Set[str]]) -> Set[str]:
    """获取图中所有节点"""
    nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        nodes.update(neighbors)
    return nodes


def build_children_map_from_edges(
    edges: List[Tuple[str, str]]
) -> Dict[str, Set[str]]:
    """
    从 (child, parent) 边构建 parent->children 邻接表
    """
    children_map: Dict[str, Set[str]] = {}
    for child, parent in edges:
        if parent not in children_map:
            children_map[parent] = set()
        children_map[parent].add(child)
        if child not in children_map:
            children_map[child] = set()
    return children_map


def compute_depth_from_root(
    children_map: Dict[str, Set[str]],
    root: str
) -> Dict[str, int]:
    """
    计算从 root 到各节点的最短距离深度 d(v)
    """
    if root not in children_map:
        return {}
    depth: Dict[str, int] = {root: 0}
    queue: deque = deque([root])
    while queue:
        node = queue.popleft()
        for child in children_map.get(node, set()):
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)
    return depth


def compute_ancestor_distances(
    adjacency: Dict[str, Set[str]],
    node: str
) -> Dict[str, int]:
    """
    从 node 向上（child->parent）BFS，得到所有祖先的最短上行距离 h(node, u)
    """
    dist: Dict[str, int] = {node: 0}
    queue: deque = deque([(node, 0)])
    while queue:
        current, d = queue.popleft()
        for parent in adjacency.get(current, set()):
            if parent not in dist:
                dist[parent] = d + 1
                queue.append((parent, d + 1))
    return dist


# ============================================================================
# 最短路径计算 (BFS)
# ============================================================================
def bfs_shortest_path(
    adjacency: Dict[str, Set[str]],
    source: str,
    targets: Set[str]
) -> Dict[str, int]:
    """
    从 source 到所有 targets 的最短路径长度 (BFS)
    
    Args:
        adjacency: 邻接表
        source: 起始节点
        targets: 目标节点集合
    
    Returns:
        distances: {target: distance} (未到达的 target 不在结果中)
    """
    if source not in adjacency:
        return {}
    
    distances: Dict[str, int] = {}
    visited: Set[str] = {source}
    queue: deque = deque([(source, 0)])
    
    # 如果 source 本身是 target
    if source in targets:
        distances[source] = 0
    
    remaining_targets = targets - {source}
    
    while queue and remaining_targets:
        current, dist = queue.popleft()
        
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                new_dist = dist + 1
                
                if neighbor in remaining_targets:
                    distances[neighbor] = new_dist
                    remaining_targets.remove(neighbor)
                
                queue.append((neighbor, new_dist))
    
    return distances


def compute_pairwise_distances(
    adjacency: Dict[str, Set[str]],
    leaf_synsets: List[str]
) -> Tuple[np.ndarray, List[Tuple[int, int, str, str]]]:
    """
    计算所有 leaf synset 对之间的最短路径距离
    
    Args:
        adjacency: 邻接表 (包含所有节点，包括内部节点)
        leaf_synsets: 叶子 synset 列表 (按顺序)
    
    Returns:
        distance_matrix: n×n 距离矩阵 (int)
        unreachable_pairs: 不可达的 (i, j) 对列表
    """
    n = len(leaf_synsets)
    leaf_set = set(leaf_synsets)
    synset_to_idx = {s: i for i, s in enumerate(leaf_synsets)}
    
    # 初始化距离矩阵 (-1 表示不可达)
    distance_matrix = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(distance_matrix, 0)
    
    unreachable_pairs: List[Tuple[int, int, str, str]] = []
    
    print(f"\n正在计算 {n} 个叶子节点的两两最短路径...")
    
    for i, source in enumerate(leaf_synsets):
        if (i + 1) % 30 == 0 or i == n - 1:
            print(f"  进度: {i + 1}/{n}")
        
        # 从 source 到所有其他 leaf 的最短路径
        distances = bfs_shortest_path(adjacency, source, leaf_set)
        
        for target, dist in distances.items():
            j = synset_to_idx[target]
            distance_matrix[i, j] = dist
    
    # 检查不可达对
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] == -1:
                unreachable_pairs.append((i, j, leaf_synsets[i], leaf_synsets[j]))
    
    return distance_matrix, unreachable_pairs


# ============================================================================
# 相似度计算
# ============================================================================
def compute_wup_similarity(
    node1: str,
    node2: str,
    adjacency: Dict[str, Set[str]],
    depth_cache: Dict[str, int] = None
) -> float:
    """
    计算两个节点的 Wu-Palmer 相似度
    
    WuP = 2 * depth(LCA) / (depth(node1) + depth(node2))
    
    Args:
        node1, node2: 两个节点（synset）
        adjacency: 邻接表（需要是有向图，child -> parent）
        depth_cache: 深度缓存（可选，用于加速）
    
    Returns:
        相似度 [0, 1]
    """
    if depth_cache is None:
        depth_cache = {}
    
    # 找到两个节点到根的路径
    def get_path_to_root(node: str) -> List[str]:
        """从 node 到根的路径"""
        path = [node]
        visited = {node}
        current = node
        
        while True:
            # 找父节点（假设只有一个父节点或取第一个）
            parents = [p for p in adjacency.get(current, []) if p not in visited]
            if not parents:
                break
            parent = parents[0]  # 取第一个父节点
            path.append(parent)
            visited.add(parent)
            current = parent
        
        return path
    
    # 获取路径
    path1 = get_path_to_root(node1)
    path2 = get_path_to_root(node2)
    
    # 找到最近公共祖先（LCA）
    set1 = set(path1)
    lca = None
    for node in path2:
        if node in set1:
            lca = node
            break
    
    if lca is None:
        # 没有公共祖先，相似度为 0
        return 0.0
    
    # 计算深度（从根开始）
    def get_depth(node: str, path: List[str]) -> int:
        """节点在路径中的深度（根为最大深度）"""
        if node in depth_cache:
            return depth_cache[node]
        # 深度 = 路径长度 - 节点在路径中的位置
        depth = len(path) - path.index(node)
        depth_cache[node] = depth
        return depth
    
    depth_lca = get_depth(lca, path1 if lca in path1 else path2)
    depth1 = get_depth(node1, path1)
    depth2 = get_depth(node2, path2)
    
    # Wu-Palmer 公式
    # wup = 2.0 * depth_lca / (depth1 + depth2) if (depth1 + depth2) > 0 else 0.0
    wup_new = depth_lca / max(depth1, depth2) if (depth1 + depth2) > 0 else 0.0
    
    
    return wup_new


def compute_wup_similarity_matrix(
    adjacency: Dict[str, Set[str]],
    leaf_synsets: List[str]
) -> np.ndarray:
    """
    计算基于 Wu-Palmer 相似度的矩阵
    
    Args:
        adjacency: 邻接表（有向图，child -> parent）
        leaf_synsets: 叶子节点列表
    
    Returns:
        similarity_matrix: n×n WuP 相似度矩阵
    """
    n = len(leaf_synsets)
    S = np.zeros((n, n), dtype=np.float32)
    depth_cache = {}
    
    print(f"\n正在计算 Wu-Palmer 相似度矩阵...")
    
    for i in range(n):
        if (i + 1) % 30 == 0 or i == n - 1:
            print(f"  进度: {i + 1}/{n}")
        
        # 对角线
        S[i, i] = 1.0
        
        # 上三角
        for j in range(i + 1, n):
            wup = compute_wup_similarity(
                leaf_synsets[i],
                leaf_synsets[j],
                adjacency,
                depth_cache
            )
            S[i, j] = wup
            S[j, i] = wup  # 对称
    
    return S


def compute_snew_similarity_matrix(
    adjacency: Dict[str, Set[str]],
    edges: List[Tuple[str, str]],
    leaf_synsets: List[str],
    root: str = "entity.n.01",
    agg: str = "max",
    gamma: float = 0.5
) -> np.ndarray:
    """
    计算 S_new 相似度矩阵（WuP 泛化，DAG 版）
    
    公式: s_u(x,y) = 2*d(u) / (2*d(u) + h(x,u) + h(y,u))
    
    其中:
    - d(u): 从 root 到公共祖先 u 的深度
    - h(x,u): 从 x 到 u 的上行距离
    - h(y,u): 从 y 到 u 的上行距离
    
    聚合方式:
    - max: 取所有公共祖先中相似度最大的
    - soft: 用 softmax(gamma * d(u)) 加权平均
    
    Args:
        adjacency: 邻接表（有向图，child -> parent）
        edges: 边列表 [(child, parent), ...]
        leaf_synsets: 叶子节点列表
        root: 根节点 synset
        agg: 聚合方式 "max" 或 "soft"
        gamma: soft 聚合的深度权重系数
    
    Returns:
        similarity_matrix: n×n S_new 相似度矩阵
    """
    n = len(leaf_synsets)
    S = np.zeros((n, n), dtype=np.float32)
    
    # 构建 parent -> children 映射，用于计算深度
    children_map = build_children_map_from_edges(edges)
    
    # 计算从 root 到各节点的深度
    depth_map = compute_depth_from_root(children_map, root)
    
    if not depth_map:
        print(f"警告: root '{root}' 不在图中，尝试自动查找根节点...")
        # 尝试找到一个有效的根节点
        all_parents = set(p for _, p in edges)
        all_children = set(c for c, _ in edges)
        possible_roots = all_parents - all_children
        if possible_roots:
            root = list(possible_roots)[0]
            print(f"  使用根节点: {root}")
            depth_map = compute_depth_from_root(children_map, root)
        
        if not depth_map:
            raise ValueError(f"无法计算深度，请检查 root 参数或图结构")
    
    print(f"  根节点: {root}")
    print(f"  深度图覆盖节点数: {len(depth_map)}")
    
    # 预计算每个叶子节点到所有祖先的上行距离
    print(f"\n正在预计算祖先距离...")
    ancestor_dists: List[Dict[str, int]] = []
    for i, syn in enumerate(leaf_synsets):
        ancestor_dists.append(compute_ancestor_distances(adjacency, syn))
        if (i + 1) % 50 == 0:
            print(f"  预计算进度: {i + 1}/{n}")
    
    print(f"\n正在计算 S_new 相似度矩阵 (agg={agg}, gamma={gamma})...")
    
    for i in range(n):
        if (i + 1) % 30 == 0 or i == n - 1:
            print(f"  进度: {i + 1}/{n}")
        
        # 对角线
        S[i, i] = 1.0
        
        # 上三角
        for j in range(i + 1, n):
            di = ancestor_dists[i]  # 节点 i 到各祖先的距离
            dj = ancestor_dists[j]  # 节点 j 到各祖先的距离
            
            # 找公共祖先
            common_ancestors = set(di.keys()) & set(dj.keys())
            
            if not common_ancestors:
                sim = 0.0
            else:
                # 过滤掉没有深度信息的祖先
                valid_ancestors = [u for u in common_ancestors if u in depth_map]
                
                if not valid_ancestors:
                    sim = 0.0
                elif agg == "max":
                    # 取相似度最大的公共祖先
                    best_sim = 0.0
                    for u in valid_ancestors:
                        du = depth_map[u]  # root 到 u 的深度
                        hu_i = di[u]       # i 到 u 的上行距离
                        hu_j = dj[u]       # j 到 u 的上行距离
                        
                        # S_new 公式: 2*d(u) / (2*d(u) + h(x,u) + h(y,u))
                        denominator = 2.0 * du + hu_i + hu_j
                        if denominator > 0:
                            su = (2.0 * du) / denominator
                        else:
                            su = 0.0
                        
                        if su > best_sim:
                            best_sim = su
                    
                    sim = best_sim
                    
                else:  # soft 聚合
                    # 用 softmax(gamma * d(u)) 加权
                    depths = np.array([depth_map[u] for u in valid_ancestors], dtype=np.float32)
                    
                    # 计算 softmax 权重
                    exp_depths = np.exp(gamma * depths)
                    weights = exp_depths / exp_depths.sum()
                    
                    # 计算每个公共祖先的相似度
                    su_list = []
                    for u in valid_ancestors:
                        du = depth_map[u]
                        hu_i = di[u]
                        hu_j = dj[u]
                        
                        denominator = 2.0 * du + hu_i + hu_j
                        if denominator > 0:
                            su = (2.0 * du) / denominator
                        else:
                            su = 0.0
                        su_list.append(su)
                    
                    # 加权平均
                    sim = float(np.sum(weights * np.array(su_list, dtype=np.float32)))
            
            S[i, j] = sim
            S[j, i] = sim  # 对称
    
    return S


def compute_hybrid_similarity_matrix(
    adjacency_undirected: Dict[str, Set[str]],
    adjacency_directed: Dict[str, Set[str]],
    leaf_synsets: List[str],
    pathsim_mode: str = "inv",
    tau: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算混合相似度矩阵 H = sqrt(PathSim * WuP)
    
    Args:
        adjacency_undirected: 无向邻接表（用于 PathSim）
        adjacency_directed: 有向邻接表（用于 WuP，child -> parent）
        leaf_synsets: 叶子节点列表
        pathsim_mode: PathSim 模式 'inv' 或 'exp'
        tau: exp 模式的温度参数
    
    Returns:
        H: n×n 混合相似度矩阵
        S_path: n×n PathSim 矩阵
        S_wup: n×n WuP 矩阵
    """
    n = len(leaf_synsets)
    leaf_set = set(leaf_synsets)
    synset_to_idx = {s: i for i, s in enumerate(leaf_synsets)}
    
    # ===== 1. 计算 PathSim（基于最短路径距离）=====
    print(f"\n[步骤 1/3] 计算 PathSim 距离矩阵...")
    D = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(D, 0)
    
    for i, source in enumerate(leaf_synsets):
        if (i + 1) % 30 == 0 or i == n - 1:
            print(f"  PathSim 进度: {i + 1}/{n}")
        
        distances = bfs_shortest_path(adjacency_undirected, source, leaf_set)
        for target, dist in distances.items():
            j = synset_to_idx[target]
            D[i, j] = dist
    
    # 转换为相似度
    S_path = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            d = D[i, j]
            if d == -1:
                S_path[i, j] = 0.0
            elif pathsim_mode == 'inv':
                S_path[i, j] = 1.0 / (1.0 + d)
            else:  # exp
                S_path[i, j] = math.exp(-d / tau)
    
    # ===== 2. 计算 WuP =====
    print(f"\n[步骤 2/3] 计算 WuP 相似度矩阵...")
    S_wup = np.zeros((n, n), dtype=np.float32)
    depth_cache = {}
    
    for i in range(n):
        if (i + 1) % 30 == 0 or i == n - 1:
            print(f"  WuP 进度: {i + 1}/{n}")
        
        S_wup[i, i] = 1.0
        
        for j in range(i + 1, n):
            wup = compute_wup_similarity(
                leaf_synsets[i],
                leaf_synsets[j],
                adjacency_directed,
                depth_cache
            )
            S_wup[i, j] = wup
            S_wup[j, i] = wup
    
    # ===== 3. 计算混合相似度 H = sqrt(PathSim * WuP) =====
    print(f"\n[步骤 3/3] 计算混合相似度 H = sqrt(PathSim * WuP)...")
    H = np.sqrt(S_path * S_wup)
    
    # 确保对角线为 1
    np.fill_diagonal(H, 1.0)
    
    return H, S_path, S_wup


def distance_to_similarity(
    distance_matrix: np.ndarray,
    mode: str = 'inv',
    tau: float = 1.0
) -> np.ndarray:
    """
    将距离矩阵转换为相似度矩阵
    
    Args:
        distance_matrix: n×n 距离矩阵 (int, -1 表示不可达)
        mode: 'inv' -> 1/(1+d), 'exp' -> exp(-d/tau)
        tau: exp 模式的温度参数
    
    Returns:
        similarity_matrix: n×n 相似度矩阵 (float32)
    """
    n = distance_matrix.shape[0]
    similarity = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            d = distance_matrix[i, j]
            
            if d == -1:
                similarity[i, j] = 0.0
            elif mode == 'inv':
                similarity[i, j] = 1.0 / (1.0 + d)
            elif mode == 'exp':
                similarity[i, j] = math.exp(-d / tau)
            else:
                raise ValueError(f"Unknown mode: {mode}")
    
    return similarity


# ============================================================================
# Sanity Check
# ============================================================================
def sanity_check(
    S: np.ndarray,
    D: np.ndarray,
    class_order: List[str],
    label_to_synset: Dict[str, str]
) -> bool:
    """
    执行 sanity check 并打印结果
    
    Returns:
        True if all checks pass, False otherwise
    """
    n = S.shape[0]
    all_passed = True
    
    print("\n" + "=" * 70)
    print("SANITY CHECK: 相似度矩阵验证")
    print("=" * 70)
    
    # 1. 对角线检查
    diag_values = np.diag(S)
    diag_all_one = np.allclose(diag_values, 1.0)
    print(f"\n[1] 对角线检查 (S[i,i] == 1.0)")
    print(f"    结果: {'✓ 通过' if diag_all_one else '✗ 失败'}")
    print(f"    对角线范围: [{diag_values.min():.6f}, {diag_values.max():.6f}]")
    if not diag_all_one:
        all_passed = False
    
    # 2. 对称性检查
    is_symmetric = np.allclose(S, S.T)
    print(f"\n[2] 对称性检查 (S == S.T)")
    print(f"    结果: {'✓ 通过' if is_symmetric else '✗ 失败'}")
    if not is_symmetric:
        max_diff = np.abs(S - S.T).max()
        print(f"    最大不对称差异: {max_diff:.6f}")
        all_passed = False
    
    # 3. 值域检查
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_values = S[off_diag_mask]
    min_val, max_val = off_diag_values.min(), off_diag_values.max()
    in_range = (min_val >= 0.0) and (max_val <= 1.0)
    print(f"\n[3] 值域检查 (0 < S[i,j] <= 1 for i != j)")
    print(f"    结果: {'✓ 通过' if in_range else '✗ 失败'}")
    print(f"    非对角线范围: [{min_val:.6f}, {max_val:.6f}]")
    
    # 4. 不可达检查
    zero_count = np.sum(S[off_diag_mask] == 0.0)
    inf_in_D = np.sum(D == -1) - n  # 排除对角线的 0
    print(f"\n[4] 连通性检查")
    print(f"    不可达对数量 (D == -1): {inf_in_D // 2}")  # 对称矩阵除以 2
    print(f"    零相似度对数量: {zero_count // 2}")
    if inf_in_D > 0:
        print(f"    ⚠ 警告: 存在不可达节点对!")
        # 找出不可达对的示例
        for i in range(min(3, n)):
            for j in range(i + 1, n):
                if D[i, j] == -1:
                    print(f"      - {class_order[i]} <-> {class_order[j]}")
                    break
    
    # 5. 距离分布
    valid_D = D[D >= 0]
    print(f"\n[5] 距离分布统计")
    print(f"    最小距离: {valid_D.min()}")
    print(f"    最大距离: {valid_D.max()}")
    print(f"    平均距离: {valid_D.mean():.2f}")
    print(f"    中位距离: {np.median(valid_D):.2f}")
    
    # 6. 近邻对示例 (手动选择一些语义相近的类)
    near_pairs = [
        ("chair", "armchair"),
        ("sofa", "cushion"),
        ("table", "desk"),
        ("floor", "rug"),
        ("wall", "ceiling"),
    ]
    
    print(f"\n[6] 近邻对相似度 (预期较高)")
    synset_to_idx = {label_to_synset.get(c, ""): i for i, c in enumerate(class_order)}
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    
    for c1, c2 in near_pairs:
        if c1 in label_to_idx and c2 in label_to_idx:
            i, j = label_to_idx[c1], label_to_idx[c2]
            sim = S[i, j]
            dist = D[i, j]
            print(f"    {c1:20s} <-> {c2:20s}: S={sim:.4f}, D={dist}")
        else:
            missing = [c for c in [c1, c2] if c not in label_to_idx]
            print(f"    {c1:20s} <-> {c2:20s}: (类 {missing} 不存在)")
    
    # 7. 远邻对示例 (手动选择一些语义较远的类)
    far_pairs = [
        ("person", "sky"),
        ("tree", "building"),
        ("car", "flower"),
        ("water", "ceiling"),
        ("animal", "chair"),
    ]
    
    print(f"\n[7] 远邻对相似度 (预期较低)")
    for c1, c2 in far_pairs:
        if c1 in label_to_idx and c2 in label_to_idx:
            i, j = label_to_idx[c1], label_to_idx[c2]
            sim = S[i, j]
            dist = D[i, j]
            print(f"    {c1:20s} <-> {c2:20s}: S={sim:.4f}, D={dist}")
        else:
            missing = [c for c in [c1, c2] if c not in label_to_idx]
            print(f"    {c1:20s} <-> {c2:20s}: (类 {missing} 不存在)")
    
    # 8. 相似度分布
    print(f"\n[8] 相似度分布统计 (非对角线)")
    print(f"    最小相似度: {min_val:.6f}")
    print(f"    最大相似度: {max_val:.6f}")
    print(f"    平均相似度: {off_diag_values.mean():.6f}")
    print(f"    中位相似度: {np.median(off_diag_values):.6f}")
    
    # 9. 最相似/最不相似对
    print(f"\n[9] 最相似的 5 对 (非对角线)")
    # 获取上三角索引并按相似度排序
    triu_indices = np.triu_indices(n, k=1)
    similarities = S[triu_indices]
    sorted_indices = np.argsort(similarities)[::-1]  # 降序
    
    for rank in range(min(5, len(sorted_indices))):
        idx = sorted_indices[rank]
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        sim = S[i, j]
        dist = D[i, j]
        print(f"    {rank+1}. {class_order[i]:20s} <-> {class_order[j]:20s}: S={sim:.4f}, D={dist}")
    
    print(f"\n[10] 最不相似的 5 对 (非对角线, 排除不可达)")
    # 过滤掉不可达对
    valid_mask = similarities > 0
    valid_similarities = similarities[valid_mask]
    valid_sorted_indices = np.argsort(valid_similarities)  # 升序
    
    count = 0
    for idx in valid_sorted_indices:
        if count >= 5:
            break
        # 找到原始索引
        orig_idx = np.where(valid_mask)[0][idx]
        i, j = triu_indices[0][orig_idx], triu_indices[1][orig_idx]
        sim = S[i, j]
        dist = D[i, j]
        print(f"    {count+1}. {class_order[i]:20s} <-> {class_order[j]:20s}: S={sim:.4f}, D={dist}")
        count += 1
    
    print("\n" + "=" * 70)
    print(f"SANITY CHECK 结果: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print("=" * 70)
    
    return all_passed


# ============================================================================
# 输出函数
# ============================================================================
def save_outputs(
    S: np.ndarray,
    class_order: List[str],
    meta: Dict[str, Any],
    outdir: Path
):
    """保存输出文件"""
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. S.npy
    S_path = outdir / "S.npy"
    np.save(S_path, S)
    print(f"\n已保存相似度矩阵: {S_path}")
    print(f"  形状: {S.shape}, 类型: {S.dtype}")
    
    # 2. class_order.txt
    order_path = outdir / "class_order.txt"
    with open(order_path, 'w', encoding='utf-8') as f:
        for label in class_order:
            f.write(f"{label}\n")
    print(f"已保存类别顺序: {order_path}")
    print(f"  共 {len(class_order)} 个类别")
    
    # 3. S_meta.json
    meta_path = outdir / "S_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已保存元数据: {meta_path}")


# ============================================================================
# Soft Confusion 接口 (签名 + 文档)
# ============================================================================
def compute_soft_confusion(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    S: np.ndarray,
    class_ids: List[int],
    ignore_index: int = 255
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算基于语义相似度的 soft TP/FP/FN
    
    【输入】
    - gt_mask: (H, W) int 数组，GT 分割掩码，值为类别索引 (0~149)，255 为忽略
    - pred_mask: (H, W) int 数组，预测分割掩码，值为类别索引 (0~149)
    - S: (C, C) float32 数组，类间相似度矩阵，C=150
    - class_ids: 评估的类别列表，如 [0, 1, 2, ..., 149]
    - ignore_index: 忽略的像素值，默认 255
    
    【输出】
    - soft_TP: (C,) float 数组，每个类的 soft True Positive
    - soft_FP: (C,) float 数组，每个类的 soft False Positive
    - soft_FN: (C,) float 数组，每个类的 soft False Negative
    
    【计算方法】
    对于类别 c:
    - hard_TP[c] = sum(gt==c & pred==c)
    - hard_FP[c] = sum(gt!=c & pred==c)  
    - hard_FN[c] = sum(gt==c & pred!=c)
    
    Soft 版本考虑预测类和 GT 类的相似度:
    - soft_TP[c] = sum_{pixels where gt==c} S[c, pred[pixel]]
    - soft_FP[c] = sum_{pixels where pred==c & gt!=c} (1 - S[c, gt[pixel]])
    - soft_FN[c] = sum_{pixels where gt==c & pred!=c} (1 - S[gt[pixel], pred[pixel]])
    
    或更简单的加权版本:
    - soft_TP[c] = sum_{pixels where gt==c} S[c, pred[pixel]]
    - soft_FP[c] = sum_{pixels where pred==c} (1 - S[c, gt[pixel]])  # gt!=ignore
    - soft_FN[c] = sum_{pixels where gt==c} (1 - S[c, pred[pixel]])
    
    【用于计算 soft-mIoU】
    soft_IoU[c] = soft_TP[c] / (soft_TP[c] + soft_FP[c] + soft_FN[c])
    soft_mIoU = mean(soft_IoU)
    
    【示例用法】
    ```python
    S = np.load("S.npy")
    gt = load_gt_mask(...)  # (H, W)
    pred = model.predict(...)  # (H, W)
    
    soft_TP, soft_FP, soft_FN = compute_soft_confusion(gt, pred, S, list(range(150)))
    
    soft_IoU = soft_TP / (soft_TP + soft_FP + soft_FN + 1e-10)
    soft_mIoU = soft_IoU.mean()
    ```
    """
    # TODO: 实现完整版本
    # 这里只提供接口签名，完整实现将在接入真实数据后完成
    
    C = len(class_ids)
    soft_TP = np.zeros(C, dtype=np.float64)
    soft_FP = np.zeros(C, dtype=np.float64)
    soft_FN = np.zeros(C, dtype=np.float64)
    
    # 获取有效像素掩码
    valid_mask = (gt_mask != ignore_index)
    
    gt_valid = gt_mask[valid_mask]
    pred_valid = pred_mask[valid_mask]
    
    for idx, c in enumerate(class_ids):
        # GT 为 c 的像素
        gt_is_c = (gt_valid == c)
        # Pred 为 c 的像素
        pred_is_c = (pred_valid == c)
        
        # Soft TP: GT==c 的像素，用 S[c, pred] 加权
        if gt_is_c.sum() > 0:
            pred_at_gt_c = pred_valid[gt_is_c]
            soft_TP[idx] = S[c, pred_at_gt_c].sum()
            
            # Soft FN: GT==c 但 pred!=c 的像素
            fn_mask = gt_is_c & (~pred_is_c)
            if fn_mask.sum() > 0:
                pred_at_fn = pred_valid[fn_mask]
                soft_FN[idx] = (1.0 - S[c, pred_at_fn]).sum()
        
        # Soft FP: pred==c 但 gt!=c 的像素
        fp_mask = pred_is_c & (~gt_is_c)
        if fp_mask.sum() > 0:
            gt_at_fp = gt_valid[fp_mask]
            soft_FP[idx] = (1.0 - S[c, gt_at_fp]).sum()
    
    return soft_TP, soft_FP, soft_FN


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从 WordNet induced graph 构建类间相似度矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 混合模式 H = sqrt(PathSim * WuP)
  python build_similarity_matrix.py \\
    --edges induced_graph_edges.tsv \\
    --mapping ade150_to_wn_canonical.json \\
    --outdir ./output \\
    --mode hybrid --pathsim_mode inv

  # S_new (WuP 泛化)
  python build_similarity_matrix.py \\
    --edges induced_graph_edges.tsv \\
    --mapping ade150_to_wn_canonical.json \\
    --outdir ./output \\
    --mode s_new --root entity.n.01 --snew_agg soft --gamma 0.5
        """
    )

    parser.add_argument(
        "--mode",
        choices=["inv", "exp", "wup", "s_new", "hybrid"],
        default="inv",
        help="相似度计算模式: inv/exp/wup/s_new/hybrid"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="entity.n.01",
        help="S_new 的根节点 synset (默认 entity.n.01)"
    )
    parser.add_argument(
        "--snew_agg",
        choices=["max", "soft"],
        default="max",
        help="S_new 聚合方式: max 或 soft"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="S_new soft 聚合的深度权重系数"
    )
    parser.add_argument(
        "--pathsim_mode",
        choices=["inv", "exp"],
        default="inv",
        help="hybrid 模式中 PathSim 的计算方式 (默认: inv)"
    )

    parser.add_argument(
        "--edges", "-e",
        type=Path,
        required=True,
        help="induced_graph_edges.tsv 文件路径"
    )
    
    parser.add_argument(
        "--mapping", "-m",
        type=Path,
        required=True,
        help="ade150_to_wn_canonical.json 文件路径"
    )
    
    parser.add_argument(
        "--outdir", "-o",
        type=Path,
        default=Path("./similarity_output"),
        help="输出目录 (默认: ./similarity_output)"
    )
    
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="exp 模式的温度参数 (默认: 1.0)"
    )
    
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=True,
        help="将边视为无向边 (默认: True, 仅用于 inv/exp 模式)"
    )
    
    parser.add_argument(
        "--directed",
        action="store_true",
        help="将边视为有向边 (覆盖 --undirected, WuP 模式必须使用有向图)"
    )
    
    parser.add_argument(
        "--save_components",
        action="store_true",
        help="hybrid 模式下同时保存 PathSim 和 WuP 分量矩阵"
    )
    
    args = parser.parse_args()
    
    # WuP / S_new 模式必须使用有向图
    if args.mode in ["wup", "s_new"]:
        undirected = False
        print("\n注意: wup / s_new 模式使用有向图（child -> parent）")
    elif args.mode == "hybrid":
        # hybrid 模式需要同时构建有向和无向图
        print("\n注意: hybrid 模式同时使用有向图和无向图")
        undirected = None  # 特殊标记
    else:
        undirected = not args.directed
    
    # 检查输入文件
    if not args.edges.exists():
        print(f"错误: 边文件不存在: {args.edges}")
        return 1
    
    if not args.mapping.exists():
        print(f"错误: 映射文件不存在: {args.mapping}")
        return 1
    
    print("=" * 70)
    print("WordNet 类间相似度矩阵构建")
    print("=" * 70)
    print(f"边文件: {args.edges}")
    print(f"映射文件: {args.mapping}")
    print(f"输出目录: {args.outdir}")
    if args.mode == "hybrid":
        print(f"模式: hybrid (H = sqrt(PathSim * WuP))")
        print(f"PathSim 模式: {args.pathsim_mode}" + (f" (tau={args.tau})" if args.pathsim_mode == "exp" else ""))
    else:
        print(f"模式: {args.mode}" + (f" (tau={args.tau})" if args.mode == "exp" else ""))
        print(f"图类型: {'无向' if undirected else '有向'}")
    
    # 1-3. 加载数据和构建图
    print(f"\n[1/5] 加载边文件...")
    edges = load_edges_tsv(args.edges)
    print(f"  加载了 {len(edges)} 条边")
    
    print(f"\n[2/5] 加载映射文件...")
    mapping = load_canonical_mapping(args.mapping)
    print(f"  加载了 {len(mapping)} 个类别")
    
    class_order = list(mapping.keys())
    label_to_synset: Dict[str, str] = {}
    leaf_synsets: List[str] = []
    
    for label in class_order:
        entry = mapping[label]
        if entry.get("hypernym_path_grounded"):
            synset = entry["hypernym_path_grounded"].split(" -> ")[0].strip()
        elif entry.get("hypernym_path_canonical"):
            synset = entry["hypernym_path_canonical"].split(" -> ")[0].strip()
        elif entry.get("canonical_synset_name"):
            synset = entry["canonical_synset_name"]
        else:
            synset = entry.get("synset_id", "")
        
        label_to_synset[label] = synset
        leaf_synsets.append(synset)
    
    print(f"  叶子 synset 数量: {len(leaf_synsets)}")
    
    print(f"\n[3/5] 构建邻接表...")
    
    if args.mode == "hybrid":
        # hybrid 模式需要两种邻接表
        adjacency_undirected = build_adjacency_list(edges, undirected=True)
        adjacency_directed = build_adjacency_list(edges, undirected=False)
        print(f"  无向图节点数: {len(get_all_nodes(adjacency_undirected))}")
        print(f"  有向图节点数: {len(get_all_nodes(adjacency_directed))}")
        adjacency = adjacency_undirected  # 用于后续检查
    else:
        adjacency = build_adjacency_list(edges, undirected=undirected)
        print(f"  图中节点数: {len(get_all_nodes(adjacency))}")
    
    all_nodes = get_all_nodes(adjacency)
    
    missing_synsets = [(class_order[i], synset) for i, synset in enumerate(leaf_synsets) 
                      if synset not in all_nodes]
    
    if missing_synsets:
        print(f"\n  ✗ 错误: {len(missing_synsets)} 个叶子 synset 不在图中!")
        for label, synset in missing_synsets[:10]:
            print(f"    - {label}: {synset}")
        return 1
    
    # 4. 计算相似度矩阵（根据模式）
    D = None  # 距离矩阵（仅 inv/exp 模式需要）
    S_path = None  # PathSim 分量（仅 hybrid 模式）
    S_wup = None   # WuP 分量（仅 hybrid 模式）
    
    if args.mode == "hybrid":
        print(f"\n[4/5] 计算混合相似度矩阵 H = sqrt(PathSim * WuP)...")
        S, S_path, S_wup = compute_hybrid_similarity_matrix(
            adjacency_undirected=adjacency_undirected,
            adjacency_directed=adjacency_directed,
            leaf_synsets=leaf_synsets,
            pathsim_mode=args.pathsim_mode,
            tau=args.tau
        )
        
    elif args.mode == "wup":
        print(f"\n[4/5] 计算 Wu-Palmer 相似度矩阵...")
        S = compute_wup_similarity_matrix(adjacency, leaf_synsets)
        
    elif args.mode == "s_new":
        print(f"\n[4/5] 计算 S_new 相似度矩阵...")
        S = compute_snew_similarity_matrix(
            adjacency=adjacency,
            edges=edges,
            leaf_synsets=leaf_synsets,
            root=args.root,
            agg=args.snew_agg,
            gamma=args.gamma
        )

    else:
        # inv 或 exp 模式：先计算距离，再转换
        print(f"\n[4/5] 计算两两最短路径距离...")
        D, unreachable = compute_pairwise_distances(adjacency, leaf_synsets)
        
        if unreachable:
            print(f"\n  ✗ 错误: 发现 {len(unreachable)} 对不可达节点!")
            for i, j, s1, s2 in unreachable[:10]:
                print(f"    - {class_order[i]} ({s1}) <-> {class_order[j]} ({s2})")
            return 1
        
        print(f"\n[5/5] 将距离转换为相似度 (mode={args.mode})...")
        S = distance_to_similarity(D, mode=args.mode, tau=args.tau)
    
    # 5. 构建元数据
    if args.mode == "hybrid":
        formula = f"H = sqrt(PathSim_{args.pathsim_mode} * WuP)"
        distance_def = "Geometric mean of PathSim and Wu-Palmer similarity"
    elif args.mode == "s_new":
        formula = "s_u=2*d(u)/(2*d(u)+h(x,u)+h(y,u)), agg=max/soft"
        distance_def = "WuP-generalization on DAG (common ancestors)"
    elif args.mode == "wup":
        formula = "2 * depth(LCA) / (depth(node1) + depth(node2))"
        distance_def = "Wu-Palmer similarity based on LCA depth"
    elif args.mode == "inv":
        formula = "1 / (1 + d)"
        distance_def = "BFS shortest path in induced graph"
    else:  # exp
        formula = f"exp(-d / {args.tau})"
        distance_def = "BFS shortest path in induced graph"
    
    meta = {
        "description": "WordNet induced graph 类间相似度矩阵",
        "generated_at": datetime.now().isoformat(),
        "edges_file": str(args.edges),
        "mapping_file": str(args.mapping),
        "num_classes": len(class_order),
        "graph_undirected": undirected if undirected is not None else "both",
        "similarity_mode": args.mode,
        "similarity_formula": formula,
        "tau": args.tau if args.mode in ["exp", "hybrid"] else None,
        "distance_definition": distance_def,
        "matrix_shape": list(S.shape),
        "matrix_dtype": str(S.dtype),
        "root": args.root if args.mode == "s_new" else None,
        "snew_agg": args.snew_agg if args.mode == "s_new" else None,
        "gamma": args.gamma if (args.mode == "s_new" and args.snew_agg == "soft") else None,
        "pathsim_mode": args.pathsim_mode if args.mode == "hybrid" else None,
    }
    
    try:
        from nltk.corpus import wordnet as wn
        meta["wordnet_version"] = wn.get_version() or "3.0"
    except Exception:
        meta["wordnet_version"] = "unknown"
    
    # 6. Sanity check
    if D is not None:
        sanity_passed = sanity_check(S, D, class_order, label_to_synset)
    else:
        # WuP / S_new / hybrid 模式的简化检查
        print("\n" + "=" * 70)
        print(f"SANITY CHECK: {args.mode.upper()} 相似度矩阵验证")
        print("=" * 70)
        
        diag_ok = np.allclose(np.diag(S), 1.0)
        sym_ok = np.allclose(S, S.T)
        range_ok = (S.min() >= 0) and (S.max() <= 1.0)
        
        print(f"对角线检查: {'✓' if diag_ok else '✗'}")
        print(f"对称性检查: {'✓' if sym_ok else '✗'}")
        print(f"值域检查 [0,1]: {'✓' if range_ok else '✗'}")
        print(f"相似度范围: [{S.min():.4f}, {S.max():.4f}]")
        print(f"平均相似度: {S[np.triu_indices(S.shape[0], k=1)].mean():.4f}")
        
        # hybrid 模式额外检查分量
        if args.mode == "hybrid" and S_path is not None and S_wup is not None:
            print(f"\n分量统计:")
            print(f"  PathSim 范围: [{S_path.min():.4f}, {S_path.max():.4f}], 均值: {S_path[np.triu_indices(S_path.shape[0], k=1)].mean():.4f}")
            print(f"  WuP 范围: [{S_wup.min():.4f}, {S_wup.max():.4f}], 均值: {S_wup[np.triu_indices(S_wup.shape[0], k=1)].mean():.4f}")
        
        sanity_passed = diag_ok and sym_ok and range_ok
        print("=" * 70)
    
    if not sanity_passed:
        print("\n⚠ Sanity check 发现问题，但仍将保存输出文件")
    
    # 7. 保存输出
    save_outputs(S, class_order, meta, args.outdir)
    
    # hybrid 模式下可选保存分量
    if args.mode == "hybrid" and args.save_components:
        if S_path is not None:
            np.save(args.outdir / "S_pathsim.npy", S_path)
            print(f"已保存 PathSim 分量: {args.outdir / 'S_pathsim.npy'}")
        if S_wup is not None:
            np.save(args.outdir / "S_wup.npy", S_wup)
            print(f"已保存 WuP 分量: {args.outdir / 'S_wup.npy'}")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - {args.outdir}/S.npy")
    print(f"  - {args.outdir}/class_order.txt")
    print(f"  - {args.outdir}/S_meta.json")
    if args.mode == "hybrid" and args.save_components:
        print(f"  - {args.outdir}/S_pathsim.npy")
        print(f"  - {args.outdir}/S_wup.npy")
    
    return 0


if __name__ == "__main__":
    exit(main())
