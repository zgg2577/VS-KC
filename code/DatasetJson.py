# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import argparse  # 引入 argparse 用于命令行参数
from tqdm import tqdm
from typing import Optional

# 支持的图片文件扩展名
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_lora_dataset_per_folder(
    dataset_root: str, output_subdir: Optional[str] = None
):
    """
    为每个子文件夹构建独立的、符合Qwen-VL格式的LoRA微调数据集JSON文件。
    // Build independent LoRA fine-tuning dataset JSON files in the Qwen-VL format for each subfolder.

    :param dataset_root: 数据集根目录路径，包含多个子文件夹，每个子文件夹代表一个实体。
    :param output_subdir: 可选，指定在每个实体文件夹内创建的子目录名称，用于存放生成的JSON文件。
                          如果为 None，则 JSON 文件直接保存在实体文件夹根目录下。
    """
    dataset_path = Path(dataset_root)
    if not dataset_path.is_dir():
        print(f"错误：数据集根目录 '{dataset_root}' 不存在或不是一个目录。")
        return

    print(f"开始处理数据集根目录: {dataset_path}")

    processed_folders = 0
    for folder in dataset_path.iterdir():
        # 确保处理的是文件夹
        if not folder.is_dir():
            continue

        print(f"\n正在处理文件夹: {folder.name}")

        # 创建数据集存储列表
        folder_dataset = []

        # --- 提取实体名称 ---
        # 保持原来的逻辑：假设文件夹名为 xxx_entity 或 entity
        # 如果需要更复杂的逻辑，可以在这里修改
        entity_parts = folder.name.split("_")
        entity = (
            entity_parts[-1] if len(entity_parts) > 1 else folder.name
        )  # 如果没有下划线，则使用整个文件夹名
        print(f"  提取实体名称: {entity}")

        # --- 构建符合 Qwen-VL 格式的对话 ---
        # 使用 'from' 和 'value' 键
        conversation_template = [
            {
                "from": "human",
                # 注意：问题中通常需要包含 <image> 占位符，训练脚本在处理时会查找它
                # 如果你的训练脚本会自动添加，这里可以不加；如果需要这里指定，则取消下面一行的注释
                # "value": "<image>\nThis is a picture of an operating room. Are there any unsafe or unreasonable factors? Answer briefly."
                "value": "This is a picture of an operating room. Are there any unsafe or unreasonable factors? Answer briefly.",  # 使用中文示例
            },
            {
                "from": "gpt",
                # 使用 f-string 动态插入实体名称
                "value": f"yes,The presence of {entity} is inappropriate in an operating room environment.",  # 使用中文示例
            },
        ]
        # ------------------------------------

        image_count = 0
        # 遍历文件夹中的图片文件 (支持多种格式)
        for img_file in folder.iterdir():
            # 检查是否是文件以及扩展名是否支持
            if (
                img_file.is_file()
                and img_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ):
                image_count += 1
                # --- 构建数据条目 ---
                # 'image': 使用相对于JSON文件的图片路径（即仅文件名）
                # 'conversations': 使用上面定义的模板
                entry = {
                    "image": img_file.name,  # 存储相对路径（文件名）
                    "conversations": conversation_template,  # 对于同一文件夹下的所有图片，使用相同的对话模板
                }
                folder_dataset.append(entry)
            # 可以选择性地在这里添加对非图片文件的警告
            # else:
            #     if img_file.is_file(): # 是文件但不是支持的图片格式
            #         print(f"  警告: 跳过不支持的文件类型: {img_file.name}")

        # 仅在找到图片数据时生成文件
        if folder_dataset:
            # --- 构建输出路径 ---
            if output_subdir:
                output_dir = folder / output_subdir
                output_dir.mkdir(
                    parents=True, exist_ok=True
                )  # 创建子目录（如果不存在）
                output_path = output_dir / f"{folder.name}_dataset.json"
            else:
                # 直接保存在实体文件夹根目录下
                output_path = folder / f"{folder.name}_dataset.json"
            # --------------------

            # --- 保存JSON文件 ---
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    # ensure_ascii=False 保证中文字符正确写入
                    json.dump(folder_dataset, f, indent=2, ensure_ascii=False)
                print(
                    f"✅ 已生成 {output_path}，包含 {len(folder_dataset)} 条数据 ({image_count} 张图片)"
                )
                processed_folders += 1
            except IOError as e:
                print(f"❌ 写入 JSON 文件失败: {output_path} - 错误: {e}")
            except Exception as e:
                print(f"❌ 生成 JSON 文件时发生未知错误: {output_path} - 错误: {e}")
        else:
            print(
                f"  警告: 文件夹 {folder.name} 中未找到支持的图片文件，未生成 JSON 文件。"
            )

    print(f"\n处理完成。共处理了 {processed_folders} 个包含有效图片的文件夹。")


def main():
    # --- 使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="为Qwen-VL LoRA微调构建数据集JSON文件")
    parser.add_argument(
        "dataset_root", type=str, help="包含实体子文件夹的数据集根目录路径。"
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,  # 默认不创建子目录
        help="可选：在每个实体文件夹内创建的子目录名称，用于存放生成的JSON文件。",
    )
    args = parser.parse_args()
    # ------------------------------------

    # 调用核心函数
    build_lora_dataset_per_folder(args.dataset_root, args.output_subdir)


if __name__ == "__main__":
    main()

# --- 如何运行 ---
# 在命令行中执行:
# python your_script_name.py /path/to/your/dataset_root
# 或者，如果想将JSON文件保存在子目录中:
# python your_script_name.py /path/to/your/dataset_root --output_subdir json_data
# ----------------
