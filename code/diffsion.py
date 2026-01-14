from sd_inpainting.core import InpaintingEngine
from sd_inpainting.utils import (
    validate_image,
    calculate_center_mask,
    generate_filename,
    validate_config,
)
from PIL import Image
import os
import hashlib
import torch
import random
from typing import List, Dict
from pathlib import Path


batch_config = {
    "model_path": "stable-diffusion-3-medium",
    "device": "auto",
    # input_images list
    "input_images": [
        # {
        #     "path": "input/.jpg",
        #     "masks": [
        #         {"type": "center", "size": 100},#中心生成demo
        #         {"coords": [100,100,200,200]},
        #     ],
        # },
    ],
    "prompt_groups": [
        # prompt list
        # {
        #     "name": "",
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "params": {"strength": 0.85,guidance_scale=7.5,num_inference_steps=30,}
        #     "variations": 50,
        # },
        # {
        #     "name": "bread",
        #     "prompt": "A piece of bread.",
        #     "negative_prompt": "(low quality:1.5), (blurry:1.5), (bad anatomy), (text:1.3), (watermark), (ugly), (mutated parts:1.2), (deformed shape),",
        #     "params": {"strength": 0.9},
        #     "variations": 100,
        # },
    ],
    "output_dir": "output",
    "naming_template": "{input_stem}_{mask_hash}_{name}_{var_idx}.jpg",
    "seed_range": [-99999, 99999],
    "clear_cache": True,
}


def process_single_task(
    engine, input_image: Dict, mask_config: Dict, prompt_group: Dict, output_dir: str
):
    """处理单个生成任务 Handle a single generation task"""
    try:
        # 基础信息获取 Acquisition of basic information

        input_stem = Path(input_image["path"]).stem
        group_name = prompt_group["name"]
        variations = prompt_group.get("variations", 1)

        # 创建专属目录// Build an exclusive directory

        group_dir = Path(output_dir) / f"{input_stem}_{group_name}"
        group_dir.mkdir(parents=True, exist_ok=True)

        # 加载图片
        validate_image(input_image["path"])
        img = Image.open(input_image["path"]).convert("RGB")

        # 计算蒙版坐标（单个mask配置）
        if "coords" in mask_config:
            coords = mask_config["coords"]
        elif mask_config.get("type") == "center":
            coords = calculate_center_mask(img.size, mask_config["size"])
        else:
            raise ValueError("无效的蒙版配置")

        # 生成多个变体// # Generate multiple variants

        for var_idx in range(1, variations + 1):
            # 设置随机种子
            seed = random.randint(*batch_config["seed_range"])
            torch.manual_seed(seed)

            # 生成文件名（包含mask特征）
            file_name = generate_filename(
                input_image["path"],
                group_name,
                coords,
                batch_config["naming_template"],
                variation_idx=var_idx,
            )

            # 完整输出路径
            output_path = group_dir / file_name

            # 执行生成
            result = engine.generate(
                image=img,
                mask_coords=coords,
                prompt=prompt_group["prompt"],
                negative_prompt=prompt_group.get("negative_prompt", ""),
                **prompt_group.get("params", {}),
            )

            result.save(output_path)
            print(f"已生成: {output_path} (Seed: {seed})")

    except Exception as e:
        print(f"处理失败: {input_image['path']} - {str(e)}")
    finally:
        if batch_config["clear_cache"] and torch.cuda.is_available():
            torch.cuda.empty_cache()


def batch_main(config: Dict):
    """批量处理主函数//Batch processing main function"""
    # 验证配置有效性
    validate_config(config)

    # 创建输出目录
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # 初始化引擎
    engine = InpaintingEngine(
        model_path=config["model_path"],
        device=config["device"] if config["device"] != "auto" else None,
    )

    # 生成任务队列，每个input_image的每个mask对应所有prompt_groups
    tasks = []
    for input_img in config["input_images"]:
        for mask_config in input_img["masks"]:  # 拆解每个mask配置
            for prompt_group in config["prompt_groups"]:
                tasks.append((input_img, mask_config, prompt_group))

    # 顺序执行任务
    for input_img, mask_config, prompt_group in tasks:  # 接收三个参数
        process_single_task(
            engine, input_img, mask_config, prompt_group, config["output_dir"]
        )
    print("批量处理完成！")


if __name__ == "__main__":
    batch_main(batch_config)
