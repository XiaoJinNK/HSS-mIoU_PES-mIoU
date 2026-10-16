"""IO 工具函数"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
from datetime import datetime


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


def save_mask(mask: np.ndarray, path: Path):
    """保存 mask 为 PNG"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(path)


def load_Hij(path: Path) -> np.ndarray:
    """加载 Hij 矩阵"""
    H = np.load(path)
    return H.astype(np.float64)


def find_image_pairs(
    gt_dir: Path,
    pred_dir: Optional[Path] = None,
    pattern: str = "*.png"
) -> List[Tuple[Path, Optional[Path], str]]:
    """
    查找 GT 和 Pred 的配对图片
    
    Returns:
        List of (gt_path, pred_path, image_name)
    """
    # 确保 gt_dir 是 Path 对象
    gt_dir = Path(gt_dir)
    
    # 尝试多种图片格式
    patterns = [pattern]
    if pattern == "*.png":
        patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    
    gt_files = []
    for p in patterns:
        gt_files.extend(gt_dir.glob(p))
    
    # 去重并排序
    gt_files = sorted(set(gt_files))
    
    pairs = []
    
    def _find_matching_pred(pred_dir: Path, gt_name: str) -> Optional[Path]:
        """尝试多种命名模式找到匹配的预测文件。"""
        patterns = [
            f"{gt_name}_pred_mask.png",  # 优先使用真正的预测 mask
            f"{gt_name}.png",
            f"{gt_name}.PNG",
            f"{gt_name}.npy",
            f"{gt_name}_seg.png",
            f"{gt_name.replace('_val_', '_')}.png",
        ]

        for pattern in patterns:
            pred_path = pred_dir / pattern
            if pred_path.exists():
                return pred_path

        # 模糊匹配时排除可视化文件，避免误配到 overlay/labeled
        if '_' in gt_name:
            core_id = gt_name.split('_')[-1]
            for f in sorted(pred_dir.glob(f"*{core_id}*.png")):
                if '_labeled' not in f.name and '_overlay' not in f.name:
                    return f

        return None

    pred_dir = Path(pred_dir) if pred_dir is not None else None

    for gt_path in gt_files:
        name = gt_path.stem
        if pred_dir is not None:
            pred_path = _find_matching_pred(pred_dir, name)
            pairs.append((gt_path, pred_path, name))
        else:
            pairs.append((gt_path, None, name))
    
    return pairs


def save_json(data: dict, path: Path):
    """保存 JSON 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_serializer)


def _json_serializer(obj):
    """JSON 序列化辅助"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    # 尝试转换其他 numpy 类型
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_json(path: Path) -> dict:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(data: List[dict], path: Path, fieldnames: Optional[List[str]] = None):
    """保存 CSV 文件"""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def save_latex_table(
    data: List[dict],
    path: Path,
    caption: str = "",
    label: str = "",
    precision: int = 4
):
    """
    保存 LaTeX 表格（booktabs 风格）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        return
    
    columns = list(data[0].keys())
    n_cols = len(columns)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        r"\begin{tabular}{" + "l" + "c" * (n_cols - 1) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    
    for row in data:
        values = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
                values.append(f"{v:.{precision}f}")
            else:
                values.append(str(v))
        lines.append(" & ".join(values) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def get_experiment_meta(args) -> dict:
    """生成实验元数据"""
    return {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args) if hasattr(args, '__dict__') else dict(args),
        "numpy_version": np.__version__,
    }


def load_class_names(path: Path) -> List[str]:
    """
    加载类别名称列表
    支持 .json 和 .txt 格式
    """
    path = Path(path)
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # {0: "wall", 1: "building", ...}
                return [data[str(i)] for i in range(len(data))]
    elif path.suffix == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
