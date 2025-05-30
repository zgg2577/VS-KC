import argparse
import os
from pathlib import Path
from acc_llm.processor import SafetyProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗安全检测系统")
    parser.add_argument("--dataset", required=True, help="数据集路径")
    parser.add_argument("--top-n", type=int, help="仅处理前N个文件夹", default=None)
    args = parser.parse_args()

    processor = SafetyProcessor()
    processor.process_dataset(args.dataset, top_n=args.top_n)  # 传递top_n参数
