import torch
import os
import sys
import tempfile
import contextlib
from PIL import Image 
from typing import Dict, List, Optional
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoModel, 
    AutoTokenizer,
    MllamaForConditionalGeneration  # 显式导入官方类
)
from peft import PeftModel
from .config import Config

# 其他模型的导入保持软兼容
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None

# [工具] DeepSeek 输出捕获 (保持原样)
@contextlib.contextmanager
def capture_output_to_file():
    tfile = tempfile.TemporaryFile(mode='w+b')
    try:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        can_redirect = True
    except Exception:
        can_redirect = False

    if not can_redirect:
        yield tfile
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tfile.fileno(), sys.stdout.fileno())
        os.dup2(tfile.fileno(), sys.stderr.fileno())
        yield tfile
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.dup2(old_stderr_fd, sys.stderr.fileno())
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        tfile.seek(0)

class LocalModel:
    def __init__(self):
        self.model_type = self._identify_model_type(Config.MODEL_NAME)
        print(f"Detected Model Type: {self.model_type}")
        
        self.model = None
        self.processor = None

        # 基础参数
        base_args = {
            "torch_dtype": Config.TORCH_DTYPE,
            "trust_remote_code": True,
            "device_map": "auto"
        }
        
        # DeepSeek 必须指定 cuda
        if self.model_type == "deepseek":
            base_args["device_map"] = "cuda"

        print(f"Loading model {Config.MODEL_NAME}...")

        # =========================================================
        # 分支 1: Llama 3.2 Vision (Clean / Official)
        # =========================================================
        if self.model_type == "llama":
            print("Using MllamaForConditionalGeneration (Official Standard)")
            # 没有任何额外的 hack 或 patch
            self.model = MllamaForConditionalGeneration.from_pretrained(
                Config.MODEL_NAME, 
                **base_args
            )
            self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        
        # =========================================================
        # 分支 2: GLM-4V
        # =========================================================
        elif self.model_type == "glm":
            if Glm4vForConditionalGeneration:
                try:
                    self.model = Glm4vForConditionalGeneration.from_pretrained(Config.MODEL_NAME, **base_args)
                except Exception as e:
                    print(f"Glm4v load failed: {e}")
            
            if self.model is None:
                self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, **base_args)
            
            self._apply_glm_patch(self.model)
            self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)

        # =========================================================
        # 分支 3: Qwen2.5-VL
        # =========================================================
        elif self.model_type == "qwen":
            if Qwen2_5_VLForConditionalGeneration:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(Config.MODEL_NAME, **base_args)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, **base_args)
            
            self.processor = AutoProcessor.from_pretrained(
                Config.MODEL_NAME, 
                min_pixels=Config.MIN_PIXELS, 
                max_pixels=Config.MAX_PIXELS,
                trust_remote_code=True
            )

        # =========================================================
        # 分支 4: DeepSeek
        # =========================================================
        elif self.model_type == "deepseek":
            self.model = AutoModel.from_pretrained(
                Config.MODEL_NAME, 
                use_safetensors=True, 
                **base_args
            )
            self.processor = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
            if self.processor.pad_token is None:
                self.processor.pad_token = self.processor.eos_token

        # =========================================================
        # 分支 5: Generic / Auto
        # =========================================================
        else:
            self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, **base_args)
            self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)

        # LoRA
        if Config.LORA_ADAPTER_PATH:
            print(f"Loading LoRA: {Config.LORA_ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(
                self.model, Config.LORA_ADAPTER_PATH, torch_dtype=Config.TORCH_DTYPE
            )
        
        self.model.eval()
        print("✅ Model loaded successfully.")

    def _identify_model_type(self, model_path: str) -> str:
        name = model_path.lower()
        if "llama" in name: return "llama"
        if "glm" in name: return "glm"
        if "deepseek" in name: return "deepseek"
        if "qwen" in name: return "qwen"
        return "unknown"

    def _apply_glm_patch(self, model):
        try:
            patch_config = {"type": "mrope", "mrope_section": [16, 24, 24]}
            if hasattr(model, "config") and (not hasattr(model.config, "rope_scaling") or model.config.rope_scaling is None):
                model.config.rope_scaling = patch_config
            for module in model.modules():
                if hasattr(module, "self_attn"):
                    attn = module.self_attn
                    if hasattr(attn, "rope_scaling") and attn.rope_scaling is None:
                        attn.rope_scaling = patch_config
        except Exception:
            pass

    def generate_response(self, messages: List[Dict], image_path: Optional[str] = None) -> str:
        # 准备输入
        raw_image = None
        if image_path:
            try:
                raw_image = Image.open(image_path).convert("RGB")
            except Exception:
                pass

        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            new_content = []
            
            # 格式化消息内容
            if isinstance(content, str):
                new_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        new_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        # Llama: 仅保留类型标记，图片对象在 processor 调用时传入
                        if self.model_type == "llama":
                             new_content.append({"type": "image"}) 
                        elif self.model_type == "deepseek":
                             pass
                        else:
                            img_val = item.get("image", image_path)
                            if img_val:
                                new_content.append({"type": "image", "image": img_val})
                                
            formatted_messages.append({"role": role, "content": new_content})

        try:
            # =========================================================
            # Llama Logic (Clean / Official Pattern)
            # =========================================================
            if self.model_type == "llama":
                # 官方标准用法:
                # 1. 使用 apply_chat_template 生成带 <|image|> 的文本
                # 2. 将文本和图片传给 processor 生成 tensor
                
                # Step 1: 获取 Prompt 文本
                text = self.processor.apply_chat_template(
                    formatted_messages, 
                    add_generation_prompt=True,
                    tokenize=False 
                )
                
                # Step 2: 处理为 Tensor
                # 注意: Processor 期望 images 是列表格式，如果没有图片则为 None
                image_inputs = [raw_image] if raw_image else None
                
                inputs = self.processor(
                    text=text,
                    images=image_inputs,
                    return_tensors="pt"
                ).to(self.model.device)

                # 生成
                output = self.model.generate(
                    **inputs, 
                    max_new_tokens=Config.GENERATION_CONFIG.get("max_new_tokens", 512),
                    do_sample=Config.GENERATION_CONFIG.get("do_sample", False),
                    temperature=Config.GENERATION_CONFIG.get("temperature", 0.7)
                )
                
                # 解码
                prompt_len = inputs.input_ids.shape[-1]
                return self.processor.decode(output[0][prompt_len:], skip_special_tokens=True)

            # =========================================================
            # DeepSeek Logic
            # =========================================================
            elif self.model_type == "deepseek":
                prompt = ""
                if raw_image and len(formatted_messages) == 1:
                    prompt += "<image>\n"
                for m in formatted_messages:
                    text_part = "".join([c["text"] for c in m["content"] if c["type"] == "text"])
                    prompt += f"{m['role']}: {text_part}\n"
                prompt += "assistant:"
                
                if not os.path.exists(Config.OUTPUT_DIR):
                    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

                final_answer = None
                with capture_output_to_file() as log_file:
                    try:
                        infer_image_arg = image_path if image_path else None
                        res = self.model.infer(
                            self.processor, 
                            prompt=prompt,
                            image_file=infer_image_arg, 
                            output_path=Config.OUTPUT_DIR,
                            base_size=1024,
                            image_size=640,
                            crop_mode=True,
                            save_results=False, 
                            test_compress=False 
                        )
                        final_answer = res
                    except Exception:
                        pass
                
                if not final_answer:
                    try:
                        log_file.seek(0)
                        raw_output = log_file.read().decode('utf-8', errors='ignore')
                        for line in reversed(raw_output.splitlines()):
                            if line.strip():
                                final_answer = line.strip()
                                break
                    except:
                        pass
                return final_answer if final_answer else Config.ERROR_PLACEHOLDER

            # =========================================================
            # GLM Logic
            # =========================================================
            elif self.model_type == "glm":
                inputs = self.processor.apply_chat_template(
                    formatted_messages,
                    add_generation_prompt=True,
                    tokenize=True, 
                    return_dict=True,
                    return_tensors="pt"
                ).to(Config.DEVICE)
                inputs.pop("token_type_ids", None)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **Config.GENERATION_CONFIG)
                return self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

            # =========================================================
            # Qwen / Generic Logic
            # =========================================================
            else:
                text_input = self.processor.apply_chat_template(
                    formatted_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=text_input,
                    images=raw_image if raw_image else None,
                    return_tensors="pt"
                ).to(Config.DEVICE)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **Config.GENERATION_CONFIG)
                
                input_len = inputs.input_ids.shape[1]
                return self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0].strip()

        except Exception as e:
            print(f"[Generate Error] {e}")
            import traceback
            traceback.print_exc()
            return Config.ERROR_PLACEHOLDER

    def multi_turn_chat(self, messages: List[Dict]) -> List[Dict]:
        if not messages: return []
        img_path = None
        if "content" in messages[0]:
            if isinstance(messages[0]["content"], list):
                for item in messages[0]["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_path = item.get("image")
                        break
        
        current_messages = [dict(m) for m in messages]
        
        # Stage 1
        input_s1 = [current_messages[0]]
        response_1 = self.generate_response(input_s1, img_path)
        
        if len(current_messages) > 1 and current_messages[1]["role"] == "assistant":
            current_messages[1]["content"] = response_1
        else:
            current_messages.insert(1, {"role": "assistant", "content": response_1})
            
        # Stage 2
        if len(current_messages) > 2:
            input_s2 = current_messages[:3]
            response_2 = self.generate_response(input_s2, img_path)
            
            if len(current_messages) > 3 and current_messages[3]["role"] == "assistant":
                current_messages[3]["content"] = response_2
            else:
                current_messages.append({"role": "assistant", "content": response_2})
        
        return current_messages

    def get_text_response(self, prompt: str, context: str) -> str:
        # Stage 3: New Conversation (Text Only)
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
        messages = [{"role": "user", "content": full_prompt}]
        return self.generate_response(messages, image_path=None)