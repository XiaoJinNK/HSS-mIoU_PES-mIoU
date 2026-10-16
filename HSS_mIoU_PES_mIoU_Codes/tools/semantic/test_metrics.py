"""
单元测试
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# 添加父目录
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "metrics"))

from semantic_metrics import (
    compute_confusion_matrix,
    compute_miou_from_confusion,
    compute_hier_soft_iou_from_confusion,
    evaluate_metrics
)
from corrupt import (
    compute_corruption_targets,
    corrupt_single_image,
    verify_corruption_ratio
)


class TestMetrics:
    """测试指标计算"""
    
    def test_confusion_matrix_basic(self):
        """测试混淆矩阵基本功能"""
        gt = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int32)
        pred = np.array([[0, 1, 1], [1, 2, 0]], dtype=np.int32)
        
        confusion = compute_confusion_matrix(gt, pred, num_classes=3)
        
        # 验证
        assert confusion.shape == (3, 3)
        assert confusion.sum() == 6  # 总像素数
        assert confusion[0, 0] == 1  # GT=0, Pred=0
        assert confusion[0, 1] == 1  # GT=0, Pred=1
        assert confusion[2, 0] == 1  # GT=2, Pred=0
    
    def test_confusion_matrix_ignore_label(self):
        """测试 ignore_label 处理"""
        gt = np.array([[0, 255, 1], [1, 2, 255]], dtype=np.int32)
        pred = np.array([[0, 0, 1], [0, 2, 1]], dtype=np.int32)
        
        confusion = compute_confusion_matrix(gt, pred, num_classes=3, ignore_label=255)
        
        assert confusion.sum() == 4  # 排除 ignore 区域
    
    def test_miou_perfect_prediction(self):
        """测试完美预测的 mIoU = 1"""
        confusion = np.diag([100, 200, 150])
        result = compute_miou_from_confusion(confusion)
        
        assert np.isclose(result['miou'], 1.0)
        assert np.allclose(result['per_class_iou'], [1.0, 1.0, 1.0])
    
    def test_hier_miou_with_identity_equals_miou(self):
        """核心测试：Hij = I 时，Hier-Soft IoU 应等于 mIoU"""
        np.random.seed(42)
        
        num_classes = 10
        
        # 创建随机混淆矩阵
        confusion = np.random.randint(0, 100, (num_classes, num_classes))
        
        # 单位矩阵
        Hij_identity = np.eye(num_classes)
        
        # 计算两种指标
        miou_result = compute_miou_from_confusion(confusion)
        hier_result = compute_hier_soft_iou_from_confusion(confusion, Hij_identity)
        
        # 验证：应该完全一致
        assert np.isclose(miou_result['miou'], hier_result['hier_miou'], rtol=1e-6), \
            f"mIoU={miou_result['miou']}, Hier-mIoU={hier_result['hier_miou']}"
        
        assert np.allclose(miou_result['per_class_iou'], hier_result['per_class_iou'], rtol=1e-6)
    
    def test_hier_miou_bounds(self):
        """测试 Hier-Soft IoU 值域 [0, 1]"""
        np.random.seed(42)
        
        num_classes = 20
        confusion = np.random.randint(0, 100, (num_classes, num_classes))
        
        # 随机 Hij（对称，对角线为 1）
        Hij = np.random.rand(num_classes, num_classes)
        Hij = (Hij + Hij.T) / 2
        np.fill_diagonal(Hij, 1.0)
        
        result = compute_hier_soft_iou_from_confusion(confusion, Hij)
        
        assert 0 <= result['hier_miou'] <= 1
        assert np.all(result['per_class_iou'] >= 0)
        assert np.all(result['per_class_iou'] <= 1)
    
    def test_soft_tp_fp_fn_non_negative(self):
        """测试 Soft TP/FP/FN 非负"""
        np.random.seed(42)
        
        num_classes = 15
        confusion = np.random.randint(0, 50, (num_classes, num_classes))
        
        Hij = np.random.rand(num_classes, num_classes)
        Hij = (Hij + Hij.T) / 2
        np.fill_diagonal(Hij, 1.0)
        Hij = np.clip(Hij, 0, 1)
        
        result = compute_hier_soft_iou_from_confusion(confusion, Hij)
        
        assert np.all(result['soft_tp'] >= 0)
        assert np.all(result['soft_fp'] >= 0)
        assert np.all(result['soft_fn'] >= 0)


class TestCorruption:
    """测试 Corruption 生成"""
    
    def test_corruption_ratio(self):
        """测试替换比例接近期望值"""
        np.random.seed(42)
        
        num_classes = 10
        gt = np.random.randint(0, num_classes, (100, 100))
        targets = np.arange(1, num_classes + 1) % num_classes  # 简单循环替换
        
        for r in [0.05, 0.10, 0.20, 0.30]:
            pred = corrupt_single_image(gt, r, targets, num_classes, seed=42)
            is_ok, actual_ratio = verify_corruption_ratio(gt, pred, r, num_classes)
            
            assert is_ok, f"Expected ratio {r}, got {actual_ratio}"
    
    def test_corruption_targets(self):
        """测试替换目标计算"""
        num_classes = 5
        Hij = np.array([
            [1.0, 0.9, 0.5, 0.2, 0.1],
            [0.9, 1.0, 0.6, 0.3, 0.2],
            [0.5, 0.6, 1.0, 0.7, 0.4],
            [0.2, 0.3, 0.7, 1.0, 0.8],
            [0.1, 0.2, 0.4, 0.8, 1.0],
        ])
        
        targets = compute_corruption_targets(Hij, k=2, seed=42)
        
        # fine_target[0] 应该是最相似的（1 或 2）
        assert targets['fine'][0] in [1, 2]
        
        # coarse_target[0] 应该是最不相似的（3 或 4）
        assert targets['coarse'][0] in [3, 4]


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()