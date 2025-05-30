import os
import json
import time
import re  # Import re here as it's used in parse_response

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .config import Config
from .api_client import APIClient
from .local_model import LocalModel


class MockModel:  # Base class for shared methods
    def multi_turn_chat(self, messages: List[Dict]) -> List[Dict]:
        """Simulates a multi-turn visual chat."""
        print("--- Mock Model: Multi-turn Chat ---")
        print(f"User Message 1 (Image + Text): {messages[0]['content'][1]['text']}")
        # Simulate model processing
        time.sleep(0.1)
        stage1_resp_text = "Yes, I see potential hazards like sharp objects and slippery surfaces."  # Example Stage 1 response
        print(f"Assistant Response 1: {stage1_resp_text}")

        print(f"User Message 2 (Text Only): {messages[2]['content']}")
        # Simulate model processing with context
        time.sleep(0.1)
        # Example Stage 2 response - needs to relate to the entity
        entity = (
            messages[2]["content"]
            .split("'{entity}'")[0]
            .split("the entity '")[-1]
            .strip("?.")
        )  # Extract entity from prompt
        stage2_resp_text = (
            f"Yes, I recognize '{entity}' in the image."
            if entity in ["scalpel", "lighting"]
            else "No, I don't clearly recognize that entity."
        )  # Example logic
        print(f"Assistant Response 2: {stage2_resp_text}")

        # Return mock conversation structure
        return [
            messages[0],  # User Stage 1
            {"role": "assistant", "content": stage1_resp_text},  # Assistant Stage 1
            messages[2],  # User Stage 2
            {"role": "assistant", "content": stage2_resp_text},  # Assistant Stage 2
        ]

    def get_text_response(self, prompt: str, context: str) -> str:
        """Simulates a single-turn text-only chat based on context."""
        print("--- Mock Model: Single-turn Text Chat ---")
        print(f"Context: {context[:50]}...")  # Print truncated context
        print(f"Prompt: {prompt}")
        # Simulate model processing based on context and prompt
        time.sleep(0.1)

        # Simple mock logic for the semantic check
        prompt_lower = prompt.lower()
        context_lower = context.lower()
        entity_match = re.search(r"related to the entity '(.+?)'", prompt_lower)
        entity = entity_match.group(1) if entity_match else "unknown"

        # Check if hazard and entity seem mentioned in the context
        if "hazard" in context_lower and entity in context_lower:
            response = "Yes"
        elif "hazard" in context_lower or entity in context_lower:
            response = "Not mentioned"  # Hazard or entity mentioned, but not necessarily together/related clearly in this simple mock
        else:
            response = "No"  # Neither mentioned

        print(f"Assistant Text Response: {response}")
        return response


class SafetyProcessor:
    def __init__(self):
        # Use the mock models for demonstration
        self.model = APIClient() if Config.USE_API else LocalModel()
        self._init_stats()

    def _init_stats(self):
        """Initialize statistics data structure"""
        self.stats = {
            "entities": defaultdict(lambda: {"total": 0, "correct": 0}),
            "folders": defaultdict(lambda: {"total": 0, "correct": 0}),
        }

    def extract_entity(self, folder_name: str) -> str:
        """
        从文件夹名称中提取实体，兼容多种命名模式。
        模式1: prefix_entity (例如: 2_1_1_plant -> plant)
        模式2: entity (例如: plant, No Parking sign)
        """
        parts = folder_name.split("_")
        if len(parts) > 1:
            # 如果有下划线，取最后一部分
            potential_entity = parts[-1]
            # 检查最后一部分是否主要由字母组成 (允许空格，例如 "No Parking sign")
            # 如果最后一部分是纯数字，或者过于简短且不是一个词，可能不是我们想要的实体名
            # 这里用一个简单的判断：如果它包含字母，我们就认为它是实体
            if re.search(r"[a-zA-Z]", potential_entity):
                return potential_entity

        # 如果没有下划线，或者按 _ 分割的最后一部分不符合实体特征 (例如纯数字)
        # 则直接使用整个文件夹名称作为实体
        return folder_name

    def parse_response(self, text: str) -> bool:
        """
        Parse Yes/No/True/False like responses (for Stage 1 and Stage 2).
        Supports Chinese and English keywords.
        """
        # Clean and convert to lowercase
        text = text.strip().lower()

        # Strict matching
        if text.endswith(("yes", "是", "true", "对", "y", "t")):
            return True
        elif text.endswith(("no", "否", "false", "错", "n", "f")):
            return False

        # Fuzzy matching for 'yes'/'no'/'是'/'否' using regex
        # Look for the last occurrence of "yes", "no", "是", or "否" as whole words
        # Re-compile regex each time or move outside if performance is critical
        last_answer_match = re.search(
            r"\b(yes|no|是|否)\b(?!.*?\b(yes|no|是|否)\b)", text, re.IGNORECASE
        )

        if last_answer_match:
            return last_answer_match.group(1).lower() in ("yes", "是", "true", "y", "t")

        # If no match, return a default (False seems appropriate for "not clearly yes")
        return False

    def parse_semantic_response(self, text: str) -> str:
        """
        Parse response for Stage 3 specifically ('Yes', 'No', 'Not mentioned').
        """
        text = text.strip().lower()

        if "yes" in text or "是" in text:
            return "Yes"
        elif "no" in text or "否" in text:
            return "No"
        elif (
            "not mentioned" in text or "未提及" in text or "未提及" in text
        ):  # Added 未提及 for clarity
            return "Not mentioned"
        # Default if none of the specific keywords are clearly found
        return "Unknown"  # Or 'Not mentioned', depending on desired strictness

    def _build_multi_turn_messages(self, image_path: str, entity: str) -> List[Dict]:
        """构建符合API要求的 Stage 1 and 2 消息结构"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": Config.FIRST_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": None,
            },  # Placeholder for model's Stage 1 response
            {"role": "user", "content": Config.SECOND_PROMPT.format(entity=entity)},
            # Placeholder for model's Stage 2 response is implicitly handled by the model's multi_turn_chat
        ]

    def process_image(self, image_path: str, entity: str) -> Dict:
        """执行带上下文关联的多轮对话 (Stage 1 & 2) 和基于文本的单轮对话 (Stage 3)"""
        try:
            # --- Stage 1 & 2: Multi-turn visual chat ---
            messages = self._build_multi_turn_messages(image_path, entity)
            conversation = self.model.multi_turn_chat(messages) or []

            # Safely get responses for Stage 1 and Stage 2
            # The conversation structure should be User1, Assistant1, User2, Assistant2
            stage1_response_text = (
                conversation[1]["content"]
                if len(conversation) > 1 and "content" in conversation[1]
                else Config.ERROR_PLACEHOLDER
            )
            stage2_response_text = (
                conversation[3]["content"]
                if len(conversation) > 3 and "content" in conversation[3]
                else Config.ERROR_PLACEHOLDER
            )

            # --- Stage 3: Single-turn text chat based on Stage 1 response ---
            stage3_prompt = Config.THIRD_PROMPT.format(entity=entity)
            stage3_response_text = self.model.get_text_response(
                prompt=stage3_prompt,
                context=stage1_response_text,  # Use Stage 1 response as context
            )

            # Parse responses
            parsed_stage1 = self.parse_response(stage1_response_text)  # Has risk?
            parsed_stage2 = self.parse_response(
                stage2_response_text
            )  # Recognizes entity?
            parsed_stage3 = self.parse_semantic_response(
                stage3_response_text
            )  # Semantically recognizes hazard from Stage 1 text?

            return {
                "dialogue": [
                    {
                        "stage": 1,
                        "question": Config.FIRST_PROMPT,
                        "response": stage1_response_text,
                        "parsed_answer": parsed_stage1,  # Parsed if hazards are present
                        "timestamp": int(time.time()),
                    },
                    {
                        "stage": 2,
                        "question": Config.SECOND_PROMPT.format(entity=entity),
                        "response": stage2_response_text,
                        "parsed_answer": parsed_stage2,  # Parsed if entity is recognized
                        "timestamp": int(time.time()),
                    },
                    {
                        "stage": 3,
                        "question": stage3_prompt,  # The semantic question
                        "response": stage3_response_text,
                        "parsed_answer": parsed_stage3,  # Parsed as 'Yes', 'No', or 'Not mentioned'
                        "timestamp": int(time.time()),
                    },
                ],
                "final_result": {
                    "has_risk": parsed_stage1,  # From Stage 1
                    "recognizes_entity": parsed_stage2,  # From Stage 2
                    "recognizes_hazard_semantic": parsed_stage3,  # From Stage 3
                    # The final "correct" decision will be based on Stage 2 AND Stage 3 later
                },
                "full_conversation_stage1_2": conversation,  # Keep the original multi-turn
                # Stage 3 is a separate interaction
            }
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {"error": str(e), "timestamp": int(time.time())}

    def _process_folder(self, folder: Path, entity: str) -> dict:
        """Process all images in a folder with complete dialogue logging"""
        results = []
        # Use glob and sort to ensure consistent order
        image_files = sorted([f for f in folder.glob("*.jpg") if f.is_file()])

        if not image_files:
            print(f"⚠️ No JPG images found in {folder.name}")
            return {
                "metadata": {
                    "folder": folder.name,
                    "entity": entity,
                    "processing_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "image_count": 0,
                },
                "results": [],
            }

        for img_file in image_files:
            print(f"  Processing {img_file.name}...")
            result = {
                "image": img_file.name,
                **self.process_image(str(img_file), entity),
            }
            results.append(result)
        return {
            "metadata": {
                "folder": folder.name,
                "entity": entity,
                "processing_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_count": len(results),
            },
            "results": results,
        }

    def _save_results(self, folder: Path, data: dict):
        """Save results according to debug settings"""
        if not Config.DEBUG_OUTPUT:
            print("⚠️ 调试模式已关闭，跳过保存JSON文件")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Include entity in filename for better organization if needed, or keep as is
        output_file = Path(Config.OUTPUT_DIR) / f"{folder.name}_QA_{timestamp}.json"

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # Use ensure_ascii=False to correctly save Chinese characters
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 调试文件已保存: {output_file}")
        except Exception as e:
            print(f"❌ Error saving debug file {output_file}: {e}")

    def _update_stats(self, folder_data: dict):
        """Update statistics from processed results"""
        entity = folder_data["metadata"]["entity"]
        folder = folder_data["metadata"]["folder"]

        # Ensure folder and entity stats exist
        if folder not in self.stats["folders"]:
            self.stats["folders"][folder] = {"total": 0, "correct": 0}
        if entity not in self.stats["entities"]:
            self.stats["entities"][entity] = {"total": 0, "correct": 0}

        for item in folder_data["results"]:
            # Only process items that have a final_result (i.e., no major errors during processing)
            if "final_result" in item:
                result = item["final_result"]
                self.stats["entities"][entity]["total"] += 1
                self.stats["folders"][folder]["total"] += 1

                # --- Modified Condition for 'correct' ---
                # Correct only if Stage 2 recognized the entity AND
                # Stage 3 semantically recognized hazards related to the entity from Stage 1 text
                if (
                    result.get("recognizes_entity") is True
                    and result.get("recognizes_hazard_semantic") == "Yes"
                ):
                    self.stats["entities"][entity]["correct"] += 1
                    self.stats["folders"][folder]["correct"] += 1
                # --- End Modified Condition ---

        # Update the folder report immediately after processing the folder
        # Only update if there were images processed in the folder
        if folder_data["metadata"]["image_count"] > 0:
            self._update_folder_report(folder)
        else:
            print(f"Skipping folder report update for {folder} (no images processed)")

    def _update_folder_report(self, folder_name: str):
        """实时更新单个文件夹的统计报告 - Overwrites the line for the specific folder"""
        from csv import reader, writer

        report_path = Path(Config.OUTPUT_DIR) / Config.FOLDER_REPORT
        temp_report_path = report_path.with_suffix(".csv.tmp")

        # Ensure output directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Get current data for the specific folder
        data = self.stats["folders"].get(folder_name, {"total": 0, "correct": 0})
        # Avoid division by zero
        accuracy = data["correct"] / data["total"] if data["total"] else 0
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        new_row = [
            folder_name,
            data["total"],
            data["correct"],
            f"{accuracy:.2%}",
            current_time_str,
        ]

        header = ["Folder", "Total", "Correct", "Accuracy", "UpdateTime"]
        rows = []
        folder_found = False

        # Read existing data, skipping the line for the current folder if it exists
        if report_path.exists():
            try:
                with open(report_path, "r", newline="", encoding="utf-8") as infile:
                    csv_reader = reader(infile)
                    header_read = next(csv_reader)  # Read header

                    # Check if the header matches, if not, treat as no existing file
                    if header_read != header:
                        print(
                            f"⚠️ Warning: Existing report header mismatch in {report_path}. Overwriting."
                        )
                        rows = [header]  # Start with the correct header
                    else:
                        rows.append(header_read)  # Keep the existing header
                        for row in csv_reader:
                            if row and row[0] == folder_name:
                                folder_found = (
                                    True  # Found the old row, we will replace it
                                )
                            elif row:  # Keep other rows
                                rows.append(row)
            except Exception as e:
                print(
                    f"❌ Error reading existing report file {report_path}: {e}. Starting fresh."
                )
                rows = [header]  # Start fresh if read fails

        else:
            rows = [header]  # File does not exist, start with header

        # Add the new/updated row for the current folder
        rows.append(new_row)

        # Sort rows by folder name (optional, but good for consistency)
        # Exclude header from sort, then re-add
        header_row = rows.pop(0)
        rows.sort(key=lambda x: x[0])
        rows.insert(0, header_row)

        # Write all data back to a temporary file
        try:
            with open(temp_report_path, "w", newline="", encoding="utf-8") as outfile:
                csv_writer = writer(outfile)
                csv_writer.writerows(rows)

            # Replace the original file with the temporary file
            os.replace(temp_report_path, report_path)
            print(f"🔄 Folder report updated: {report_path}")
        except Exception as e:
            print(f"❌ Error writing folder report file {report_path}: {e}")
            # Clean up temp file if replace failed
            if temp_report_path.exists():
                temp_report_path.unlink()

    def _generate_reports(self):
        """Generate final entity report"""
        from csv import writer

        # Generate entity report
        entity_report = Path(Config.OUTPUT_DIR) / Config.ENTITY_REPORT
        try:
            with open(entity_report, "w", newline="", encoding="utf-8") as f:
                csv_writer = writer(f)
                csv_writer.writerow(["Entity", "Total", "Correct", "Accuracy"])
                # Sort by entity name for consistent output
                for entity, data in sorted(self.stats["entities"].items()):
                    # Avoid division by zero
                    accuracy = data["correct"] / data["total"] if data["total"] else 0
                    csv_writer.writerow(
                        [entity, data["total"], data["correct"], f"{accuracy:.2%}"]
                    )
            print(f"\n📊 Final entity report generated: {entity_report}")
        except Exception as e:
            print(f"❌ Error generating entity report {entity_report}: {e}")

    def process_dataset(self, dataset_path: str, top_n: Optional[int] = None):
        """Main processing pipeline"""
        # Ensure output directory exists
        output_dir = Path(Config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(dataset_path)
        # Find all directories within the dataset path
        folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

        # Apply TOP_N limit if specified in Config or passed as argument
        process_limit = (
            Config.TOP_N_FOLDERS if Config.TOP_N_FOLDERS is not None else top_n
        )
        if process_limit is not None:
            folders = folders[:process_limit]
            print(f"⏳ Debug mode limit: Processing first {process_limit} folders.")

        print(f"\n🔍 Starting processing of {len(folders)} folders...")

        # Reset folder report or ensure header is written if it's a new run

        for folder in folders:
            entity = self.extract_entity(folder.name)
            print(f"\n📂 Processing {folder.name} (Entity: {entity})")

            folder_results = self._process_folder(folder, entity)
            self._save_results(folder, folder_results)
            # Update stats and folder report immediately after each folder
            if (
                folder_results["metadata"]["image_count"] > 0
            ):  # Only update if images were processed
                self._update_stats(folder_results)
            else:
                print(
                    f"No images processed in {folder.name}, skipping stats update for this folder."
                )

        # After all folders are processed, generate the final entity report
        self._generate_reports()
        print("\n🎉 Processing completed!")
