# sd_inpainting/utils.py
from PIL import Image, ImageDraw
import hashlib
from pathlib import Path
from typing import List
from PIL import Image
import os
from typing import Dict


def validate_image(image_path):
    """验证图片有效性"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        raise ValueError(f"无效图片文件: {str(e)}")


def preview_mask(image_path, mask_coords, save_path=None):
    """生成遮罩预览图"""
    with Image.open(image_path) as img:
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(mask_coords, fill=255)

        # 生成半透明红色叠加层
        overlay = Image.new("RGBA", img.size, (255, 0, 0, 100))
        preview = img.convert("RGBA").copy()
        preview.alpha_composite(overlay, mask=mask)

        if save_path:
            preview.save(save_path)
        return preview


def calculate_center_mask(image_size, box_size):
    """计算中心区域坐标"""
    w, h = image_size
    return [
        (w - box_size) // 2,
        (h - box_size) // 2,
        (w + box_size) // 2,
        (h + box_size) // 2,
    ]


def generate_filename(
    input_path: str,
    name: str,
    mask_coords: List[int],
    template: str,
    variation_idx: int = 1,
) -> str:
    """生成符合规范的文件名"""
    input_stem = Path(input_path).stem
    mask_hash = hashlib.md5(str(mask_coords).encode()).hexdigest()[:6]
    return template.format(
        input_stem=input_stem, mask_hash=mask_hash, name=name, var_idx=variation_idx
    )


def validate_config(config: Dict):
    """配置验证函数"""
    required_keys = ["model_path", "input_images", "prompt_groups", "output_dir"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"缺失必要配置项: {key}")

    for idx, group in enumerate(config["prompt_groups"]):
        if "name" not in group:
            raise ValueError(f"prompt_groups[{idx}]缺失name字段")
        if "prompt" not in group:
            raise ValueError(f"prompt_groups[{idx}]缺失prompt字段")

    for img in config["input_images"]:
        if not Path(img["path"]).exists():
            raise FileNotFoundError(f"图片不存在: {img['path']}")
