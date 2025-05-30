# llm_Judgment/stats.py
import csv
import os
from collections import defaultdict
from pathlib import Path
from .config import Config


class StatsCollector:
    def __init__(self):
        # 实体级统计
        self.entity_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        # 文件夹级统计
        self.folder_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    def update(self, folder_name: str, entity: str, is_correct: bool):
        """更新两级统计"""
        # 更新实体统计
        self.entity_stats[entity]["total"] += 1
        if is_correct:
            self.entity_stats[entity]["correct"] += 1

        # 更新文件夹统计
        self.folder_stats[folder_name]["total"] += 1
        if is_correct:
            self.folder_stats[folder_name]["correct"] += 1

    def generate_table(self, output_path: str):
        """生成双维度统计报表"""
        self._generate_entity_report(output_path)
        self._generate_folder_report(output_path)

    def _generate_entity_report(self, output_path: str):
        """生成实体维度报告"""
        headers = ["Entity", "Total Samples", "Correct Answers", "Accuracy"]
        rows = []

        for entity, data in sorted(self.entity_stats.items()):
            rate = (data["correct"] / data["total"]) if data["total"] else 0
            rows.append([entity, data["total"], data["correct"], f"{rate:.2%}"])

        output_file = Path(output_path) / f"{Config.FINAL_PREFIX}entity_accuracy.csv"
        self._write_csv(output_file, headers, rows)

    def _generate_folder_report(self, output_path: str):
        """新增：生成文件夹维度报告"""
        headers = ["Folder Name", "Total Samples", "Correct Answers", "Accuracy"]
        rows = []

        for folder, data in sorted(self.folder_stats.items()):
            rate = (data["correct"] / data["total"]) if data["total"] else 0
            rows.append([folder, data["total"], data["correct"], f"{rate:.2%}"])

        output_file = Path(output_path) / f"{Config.FINAL_PREFIX}folder_accuracy.csv"
        self._write_csv(output_file, headers, rows)

    def _write_csv(self, file_path: Path, headers: list, rows: list):
        """通用CSV写入方法"""
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"报表已生成: {file_path}")
