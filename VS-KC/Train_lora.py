import os
import json
import torch
import glob
import argparse
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
import psutil
import matplotlib.pyplot as plt


class MemoryMonitor:
    """内存监控工具类""" // "Memory Monitoring Tool Class"

    @staticmethod
    def print_usage(prefix=""):
        process = psutil.Process()
        mem = process.memory_info().rss / 1024**2
        print(f"{prefix}Memory used: {mem:.2f} MB")


class ModelLoader:
    """模型加载类""" // "Model Loading Class"

    @staticmethod
    def load_model_and_tokenizer():
        MemoryMonitor.print_usage("Before loading model: ")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.enable_input_require_grads()

        tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-VL-7B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen2.5-VL-7B-Instruct")

        MemoryMonitor.print_usage("After loading model: ")
        return model, tokenizer, processor


class EpochSaver(TrainerCallback):
    """保存每个epoch模型的回调类""" // "The callback class for saving the model of each epoch"

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # 如果state.epoch不为空，则保存模型
        if state.epoch is not None:
            epoch = int(state.epoch)
            # 构建epoch目录
            epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
            # 创建epoch目录
            os.makedirs(epoch_dir, exist_ok=True)
            # 保存模型
            kwargs["model"].save_pretrained(epoch_dir)
            # 打印保存信息
            print(f"Epoch {epoch} model saved to {epoch_dir}")
        # 返回control
        return control


class LossVisualizer:
    """训练损失可视化类""" // "Training Loss Visualization Class"

    MIN_POINTS_PER_EPOCH = 5  # 每个epoch至少需要5个点才绘图

    @classmethod
    def plot_and_save(cls, loss_history, output_dir, actual_epochs):
        """安全绘图方法"""
        try:
            # 验证数据有效性
            if not cls._validate_data(loss_history, actual_epochs):
                print("跳过绘图：数据点不足")
                return

            # 动态计算每个epoch的点数
            points_per_epoch = len(loss_history) / actual_epochs

            plt.figure(figsize=(12, 6))

            # 智能坐标生成
            x = [i / points_per_epoch + 1 for i in range(len(loss_history))]
            plt.plot(x, loss_history, "g-", linewidth=2, markersize=8)

            # 添加辅助线（仅当有完整epoch时）
            if actual_epochs >= 1:
                for epoch in range(1, int(actual_epochs) + 1):
                    plt.axvline(x=epoch, color="r", linestyle="--", alpha=0.3)

            # 配置图表
            plt.title(f"Training Loss (Epochs: {actual_epochs:.1f})")
            plt.xlabel("Epoch Progress")
            plt.ylabel("Loss Value")
            plt.grid(True)

            # 保存图表
            plot_path = os.path.join(output_dir, "safe_loss_curve.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"成功生成损失曲线图：{plot_path}")

        except Exception as e:
            print(f"绘图失败：{str(e)}")
            raise

    @classmethod
    def _validate_data(cls, loss_data, epochs):
        """数据验证逻辑""" // "Data Verification Logic"
        # 空数据检查
        if not loss_data:
            print("验证失败：损失数据为空")
            return False

        # 最小点数检查
        min_points = cls.MIN_POINTS_PER_EPOCH * epochs
        if len(loss_data) < min_points:
            print(f"数据点不足：需要至少{min_points}个点，实际{len(loss_data)}个")
            return False

        # 数值有效性检查
        if any(not isinstance(v, (int, float)) for v in loss_data):
            print("数据包含无效值")
            return False

        return True


class DataProcessor:
    """数据处理工具类""" // "Data Processing Tool Class"

    @staticmethod
    def extract_entity_from_folder(folder_name):
        parts = folder_name.split("_")
        return parts[-1] if len(parts) > 1 else folder_name

    @staticmethod
    def process_func(example, tokenizer, processor):
        MAX_LENGTH = 8192
        conversation = example.get("conversations")
        if not isinstance(conversation, list) or len(conversation) < 2:
            print(f"Warning: Invalid conversation format - {type(conversation)}")
            return None

        input_content = conversation[0].get("value")
        output_content = conversation[1].get("value")

        if input_content is None or output_content is None:
            print("Warning: Missing input/output content")
            return None

        image_path = os.path.join(example["data_dir"], example.get("image", ""))
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 256,
                        "resized_width": 256,
                    },
                    {"type": "text", "text": input_content},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {key: value.tolist() for key, value in inputs.items()}

        response = tokenizer(f"{output_content}", add_special_tokens=False)
        input_ids = (
            inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
        )
        attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
        labels = (
            [-100] * len(inputs["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
        )

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "pixel_values": torch.tensor(inputs["pixel_values"]),
            "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0),
        }

    @classmethod
    def load_datasets_from_folders(cls, base_dir):
        all_data = []
        folders = [
            f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
        ]
        # print(f"Found {len(folders)} folders in {base_dir}")

        for folder in folders:
            folder_path = os.path.join(base_dir, folder)
            json_files = glob.glob(os.path.join(folder_path, "*_dataset.json"))
            # print(f"Found {len(json_files)} JSON files in {folder}")

            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        entity = cls.extract_entity_from_folder(folder)

                        if isinstance(data, list):
                            print(f"Loaded {len(data)} items from {json_file}")
                            for item in data:
                                if isinstance(item, dict):
                                    item["data_dir"] = folder_path
                                    item["entity"] = entity
                                    all_data.append(item)
                        elif isinstance(data, dict):
                            print(f"Loaded single dict from {json_file}")
                            data["data_dir"] = folder_path
                            data["entity"] = entity
                            all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error loading {json_file}: {e}")

        print(f"Total raw samples loaded: {len(all_data)}")
        return all_data

    @classmethod
    def prepare_dataset(cls, all_data, tokenizer, processor):
        dataset = Dataset.from_list(all_data)
        process_fn = lambda example: cls.process_func(example, tokenizer, processor)
        processed_dataset = dataset.map(process_fn, remove_columns=dataset.column_names)
        processed_dataset = processed_dataset.filter(lambda x: x is not None)

        print(f"Processed dataset size: {len(processed_dataset)}")
        if len(processed_dataset) == 0:
            raise ValueError("All samples were filtered out during processing!")
        return processed_dataset


class LoRAConfig:
    """LoRA配置类""" // "LoRA Configuration Class"

    @staticmethod
    def create_config():
        return LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            inference_mode=False,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )


class ModelTrainer:
    @staticmethod
    def train_model(
        model,
        train_dataset,
        tokenizer,
        output_dir,
        learning_rate=1e-4,
        epochs=5,
        batch_size=4,
    ):
        # 计算每个epoch的步数
        steps_per_epoch = len(train_dataset) // (
            batch_size * 4
        )  # gradient_accumulation_steps=4
        logging_steps = max(1, steps_per_epoch // 10)  # 每个epoch约10个记录点

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            logging_strategy="steps",  # 修改为按steps记录
            logging_steps=logging_steps,
            save_strategy="no",
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            gradient_checkpointing=True,
            label_names=["labels"],
            report_to="none",
            fp16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            callbacks=[EpochSaver(output_dir)],
        )

        trainer.train()
        # 收集所有step的loss记录
        loss_history = []
        actual_epochs = 0
        for log in trainer.state.log_history:
            if "loss" in log and "epoch" in log:
                loss_history.append(log["loss"])
                actual_epochs = max(actual_epochs, int(log["epoch"]))

        # 验证epoch数量
        if actual_epochs != epochs:
            print(f"Warning: 要求训练{epochs}个epoch，实际完成{actual_epochs}个")
        # 更精确的epoch计算
        actual_epochs = int(trainer.state.epoch) if trainer.state.epoch else 0
        epoch_fraction = (
            trainer.state.epoch - actual_epochs if trainer.state.epoch else 0
        )

        print(f"\n训练进度验证：")
        print(f"理论epoch数: {epochs}")
        print(f"实际完成epoch数: {actual_epochs} + {epoch_fraction:.2f}")

        return model, loss_history, actual_epochs  # 返回实际epoch数


class AllModeTrainer(ModelTrainer):
    """全量模式训练器""" // "Full Mode Trainer"

    @classmethod
    def train(cls, base_dir, output_dir, learning_rate, epochs, batch_size):
        print("\n=== Starting All Mode Training ===")
        model, tokenizer, processor = ModelLoader.load_model_and_tokenizer()

        # 数据准备
        all_data = DataProcessor.load_datasets_from_folders(base_dir)
        processed_dataset = DataProcessor.prepare_dataset(
            all_data, tokenizer, processor
        )

        # 模型准备
        lora_config = LoRAConfig.create_config()
        peft_model = get_peft_model(model, lora_config)

        # 训练
        peft_model, loss_history, actual_epochs = cls.train_model(
            peft_model,
            processed_dataset,
            tokenizer,
            output_dir=output_dir,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        # 安全绘图（仅当满足条件时）
        if len(loss_history) >= LossVisualizer.MIN_POINTS_PER_EPOCH * actual_epochs:
            LossVisualizer.plot_and_save(loss_history, output_dir, actual_epochs)
        else:
            print("绘图条件不满足，跳过可视化步骤")
        return peft_model  # 返回训练好的模型


class EntityModeTrainer(ModelTrainer):
    """实体模式训练器"""

    # ...


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Qwen2.5-VL")
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "entity"],
        required=True,
        help="Training mode: 'all' for all folders at once, 'entity' for entity-wise training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/Qwen2.5-VL-LoRA",
        help="Output directory for saving models",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to entity configuration JSON file (required for entity mode)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (for 'all' mode)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (for 'all' mode)"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        print("\nDEBUG MODE CONFIGURATION:")
        print(f" - Batch size: {min(args.batch_size, 2)}")
        print(f" - Learning rate: {args.lr}")
        print(f" - Epochs: {args.epochs}\n")

    if args.mode == "all":
        AllModeTrainer.train(
            args.base_dir,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.mode == "entity":
        if not args.config_file:
            raise ValueError("For entity mode, --config_file must be provided")

        with open(args.config_file, "r") as f:
            entity_config = json.load(f)

        EntityModeTrainer.train(
            args.base_dir,
            entity_config,
            output_dir_base=args.output_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
