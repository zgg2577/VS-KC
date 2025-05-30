import torch
import os
import json
import argparse
import shutil
from collections import defaultdict
from llm_Judgment.config import Config
from llm_Judgment.model_loader import load_components
from llm_Judgment.processor import ImageProcessor


def compare_results(input_folder: str, output_folder: str):
    """对比两次处理结果，找出均为yes的图片并复制到输出目录// Compare the processing results of the two times, find the pictures that are both "yes" and copy them to the output directory."""

    result_files = [
        f
        for f in os.listdir(input_folder)
        if f.startswith(Config.RESULT_PREFIX) and f.endswith(".json")
    ]

    if len(result_files) != 2:
        raise ValueError(f"需要2个结果文件，当前找到 {len(result_files)} 个")

    # 读取结果文件
    results = []
    for f in result_files:
        try:
            with open(os.path.join(input_folder, f), "r") as file:
                data = json.load(file)
                results.append(
                    {item["image"]: item["answer"].lower() for item in data["results"]}
                )
        except Exception as e:
            print(f"读取文件 {f} 失败: {str(e)}")
            return

    # 找出两次均为yes的图片
    common_yes = [
        img
        for img in results[0]
        if img in results[1] and results[0][img] == "yes" and results[1][img] == "yes"
    ]

    # 准备目标目录
    os.makedirs(output_folder, exist_ok=True)

    # 复制图片并统计结果
    success, fail = 0, 0
    for img_name in common_yes:
        src = os.path.join(input_folder, img_name)
        dst = os.path.join(output_folder, img_name)

        try:
            # 验证源文件是否存在
            if not os.path.exists(src):
                print(f"警告：源文件 {src} 不存在")
                fail += 1
                continue

            # 执行复制操作
            shutil.copy(src, dst)
            success += 1
        except Exception as e:
            print(f"复制 {img_name} 失败: {str(e)}")
            fail += 1

    print(f"成功复制 {success} 张图片到 {output_folder}")
    if fail > 0:
        print(f"失败 {fail} 次")


def process_folder_twice(img_processor, input_folder: str, output_root: str):
    """处理文件夹两次并生成不同文件名// Process the folder twice and generate different file names."""

    # 为当前输入文件夹创建对应的输出文件夹
    folder_name = os.path.basename(input_folder)
    output_folder = os.path.join(output_root, folder_name)

    for i in range(1, 3):
        try:
            print(f"\n第 {i} 次处理: {folder_name}")

            # 临时修改配置
            original_prefix = Config.RESULT_PREFIX
            Config.RESULT_PREFIX = f"QA_{i}_"

            # 执行处理
            img_processor.process_folder(input_folder)

            # 恢复配置
            Config.RESULT_PREFIX = original_prefix
        except Exception as e:
            print(f"第 {i} 次处理失败: {str(e)}")
            continue

    # 结果对比和复制
    try:
        compare_results(input_folder, output_folder)
    except Exception as e:
        print(f"结果处理失败: {str(e)}")


def main():
    # 初始化模型// # Initialize the model

    model, processor = load_components()
    img_processor = ImageProcessor(model, processor)

    # GPU优化
    if "cuda" in Config.DEVICE:
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True

    # 处理参数
    parser = argparse.ArgumentParser(description="图像处理脚本")
    parser.add_argument("--input", type=str, required=True, help="输入目录路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入目录不存在: {args.input}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 处理每个子文件夹
    for folder in sorted(os.listdir(args.input)):
        folder_path = os.path.join(args.input, folder)
        if os.path.isdir(folder_path):
            try:
                print(f"\n{'='*40}")
                print(f"开始处理文件夹: {folder}")
                process_folder_twice(img_processor, folder_path, args.output)
            except Exception as e:
                print(f"文件夹处理失败: {str(e)}")
                continue


if __name__ == "__main__":
    main()
