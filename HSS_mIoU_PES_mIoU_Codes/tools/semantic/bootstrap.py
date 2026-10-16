"""
Bootstrap 统计模块
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def bootstrap_metric_difference(
    values1: np.ndarray,
    values2: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    对两组指标的差异进行 Bootstrap 分析
    
    Args:
        values1: 第一组指标值（每张图一个值）
        values2: 第二组指标值（每张图一个值）
        n_bootstrap: Bootstrap 采样次数
        confidence_level: 置信水平
        seed: 随机种子
    
    Returns:
        dict with 'mean_diff', 'ci_low', 'ci_high', 'p_value'
    """
    rng = np.random.default_rng(seed)
    n_samples = len(values1)
    
    # 原始差异
    diff = values1 - values2
    original_mean = diff.mean()
    
    # Bootstrap 采样
    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_diff = diff[indices]
        bootstrap_means.append(bootstrap_diff.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    # 置信区间（percentile 方法）
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    # p-value（双边检验：差异是否显著不为 0）
    # 近似方法：看 0 在 bootstrap 分布中的位置
    p_value = 2 * min(
        (bootstrap_means <= 0).mean(),
        (bootstrap_means >= 0).mean()
    )
    
    return {
        'mean_diff': float(original_mean),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': float(p_value),
        'n_bootstrap': n_bootstrap,
        'n_samples': n_samples,
        'confidence_level': confidence_level,
        'significant': ci_low > 0 or ci_high < 0  # CI 不跨 0
    }


def bootstrap_comparison(
    per_image_results_fine: List[Dict],
    per_image_results_coarse: List[Dict],
    metric_key: str = 'hier_miou',
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    比较 fine 和 coarse 错误下的指标差异
    
    Args:
        per_image_results_fine: fine corruption 的每图结果
        per_image_results_coarse: coarse corruption 的每图结果
        metric_key: 要比较的指标键
        n_bootstrap: Bootstrap 采样次数
        confidence_level: 置信水平
        seed: 随机种子
    
    Returns:
        Bootstrap 分析结果
    """
    values_fine = np.array([r[metric_key] for r in per_image_results_fine])
    values_coarse = np.array([r[metric_key] for r in per_image_results_coarse])
    
    return bootstrap_metric_difference(
        values_fine, values_coarse, n_bootstrap, confidence_level, seed
    )