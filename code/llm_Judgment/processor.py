import os
import time
import json
import torch
import sys
import contextlib
import logging
import tempfile
import traceback 
from PIL import Image
from typing import List, Dict, Union
from .config import Config

# [屏蔽] transformers 的普通日志
logging.getLogger("transformers").setLevel(logging.ERROR)

# [关键工具] 上下文管理器：将 stdout/stderr 重定向到临时文件
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

# 简单屏蔽器 (用于 Qwen/LLaVA/GLM)
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# 尝试导入 Qwen 工具
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

class ImageProcessor:
    # [新增] 统一支持的图片扩展名，方便复用
    VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.entity_cache = {}

    # [新增] 静态方法：预检查文件夹是否有图片
    # 用途：在加载模型前调用此方法，如果返回 False，则直接跳过该文件夹，避免浪费时间加载模型
    @staticmethod
    def has_valid_images(folder_path: str) -> bool:
        if not os.path.exists(folder_path):
            return False
        # 使用 scandir 比 listdir 更高效，特别是对于大文件夹，找到第一个匹配项即可返回
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(ImageProcessor.VALID_EXTS):
                        return True
        except Exception:
            return False
        return False
        
    def is_qwen(self):
        model_name = getattr(Config, "MODEL_NAME", "").lower()
        if "qwen" in model_name: return True
        if hasattr(self.processor, "__class__"):
            if "qwen" in self.processor.__class__.__name__.lower(): return True
        return False

    def is_deepseek(self):
        model_name = getattr(Config, "MODEL_NAME", "").lower()
        if "deepseek" in model_name: return True
        return False
    
    # GLM 判断
    def is_glm(self):
        model_name = getattr(Config, "MODEL_NAME", "").lower()
        if "glm" in model_name: return True
        return False

    # [新增] Llama 判断
    def is_llama(self):
        model_name = getattr(Config, "MODEL_NAME", "").lower()
        # Llama-3.2-Vision usually contains "llama"
        if "llama" in model_name: return True
        if hasattr(self.processor, "__class__"):
            # MllamaProcessor
            if "mllama" in self.processor.__class__.__name__.lower(): return True
        return False

    def build_qwen_messages(self, image_path: str, entity: str) -> List[Dict]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Does this image contain {entity}? Answer yes/no only."}
            ]
        }]
    
    # GLM 消息构建 (仿照官方格式)
    def build_glm_messages(self, image, entity: str) -> List[Dict]:
        return [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image # 传入 PIL Image 对象
                },
                {
                    "type": "text",
                    "text": f"Does this image contain {entity}? Answer yes/no only."
                }
            ]
        }]

    # [新增] Llama 消息构建 (仿照官方格式)
    def build_llama_messages(self, image, entity: str) -> List[Dict]:
        return [{
            "role": "user",
            "content": [
                {"type": "image"}, # Llama 3.2 Vision uses implicit image type without path/url here usually, or handled by processor
                {"type": "text", "text": f"Does this image contain {entity}? Answer yes/no only."}
            ]
        }]

    def build_llava_messages(self, entity: str) -> List[Dict]:
        return [{
            "role": "user",
            "content": f"<image>\nDoes this image contain {entity}? Answer yes/no only."
        }]
    
    def parse_response(self, response: str) -> str:
        if not response: return "error"
        response = str(response).lower().strip()
        if "yes" in response: return "yes"
        if "no" in response: return "no"
        if response.startswith(("是", "true")): return "yes"
        if response.startswith(("否", "false")): return "no"
        return response[:20]
    
    def process_single_image(self, image_path: str, entity: str) -> str:
        try:
            inputs = None
            
            # --- DeepSeek 逻辑 ---
            if self.is_deepseek():
                prompt = f"<image>\nDoes this image contain {entity}? Answer yes/no only. "
                if not os.path.exists(Config.OUTPUT_DIR):
                    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

                raw_output = ""
                res = None
                
                with capture_output_to_file() as log_file:
                    try:
                        res = self.model.infer(
                            self.processor, 
                            prompt=prompt,
                            image_file=image_path,
                            output_path=Config.OUTPUT_DIR,
                            base_size=1024,
                            image_size=640,
                            crop_mode=True,
                            save_results=False, 
                            test_compress=False 
                        )
                    except Exception:
                        res = None

                try:
                    raw_output = log_file.read().decode('utf-8', errors='ignore')
                    log_file.close()
                except:
                    raw_output = ""

                final_answer = None
                if res and isinstance(res, str) and len(res.strip()) > 0:
                    final_answer = res
                
                if not final_answer and raw_output:
                    lines = raw_output.splitlines()
                    for line in reversed(lines):
                        clean = line.strip().lower()
                        if clean == "yes":
                            final_answer = "yes"; break
                        if clean == "no":
                            final_answer = "no"; break
                
                if not final_answer:
                     raise ValueError("DeepSeek output empty")

                return self.parse_response(final_answer)

            # --- Qwen 逻辑 ---
            elif self.is_qwen():
                if process_vision_info is None:
                    raise ImportError("检测到 Qwen 模型但未安装 qwen_vl_utils")
                    
                messages = self.build_qwen_messages(image_path, entity)
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(Config.DEVICE)

            # --- GLM 逻辑 (带 Hotfix 修复) ---
            elif self.is_glm():
                # [Hotfix] 强力修复: 针对 transformers 版本不匹配导致的 TypeError: 'NoneType' object is not subscriptable
                # 这种错误是因为 GLM-4V 的 modeling 代码期望 config 中有 rope_scaling，但加载的旧权重 config 中没有。
                try:
                    patch_config = {"type": "mrope", "mrope_section": [16, 24, 24]}
                    
                    # 1. 修复 Config 对象
                    if hasattr(self.model, "config"):
                        if not hasattr(self.model.config, "rope_scaling") or self.model.config.rope_scaling is None:
                            self.model.config.rope_scaling = patch_config

                    # 2. [关键] 显式路径修复 (最稳健的方法)
                    # 直接定位到 model.layers，绕过 named_modules 可能的遍历问题
                    layers = None
                    if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                        layers = self.model.model.layers # 常用于 Glm4vForConditionalGeneration
                    elif hasattr(self.model, "layers"):
                        layers = self.model.layers # 常用于 Glm4vModel
                    
                    if layers is not None:
                        for i, layer in enumerate(layers):
                            # 假设 layer 是 DecoderLayer, 试图找到 self_attn
                            if hasattr(layer, "self_attn"):
                                attn = layer.self_attn
                                # 强制注入，防止 None 报错
                                if not hasattr(attn, "rope_scaling") or attn.rope_scaling is None:
                                    attn.rope_scaling = patch_config
                    
                    # 3. 兜底：通用遍历 (防止结构差异)
                    for name, module in self.model.named_modules():
                        if "Attention" in module.__class__.__name__ or hasattr(module, "rope_scaling"):
                             if hasattr(module, "rope_scaling") and (module.rope_scaling is None):
                                 module.rope_scaling = patch_config

                except Exception as e:
                    # print(f"[GLM Debug] Patching failed (Non-fatal): {e}")
                    pass

                try:
                    raw_image = Image.open(image_path).convert("RGB")
                    messages = self.build_glm_messages(raw_image, entity)
                    
                    # 1. Apply Chat Template
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(Config.DEVICE)
                    
                    if inputs is None:
                        raise ValueError(f"inputs is None for {image_path}")
                    
                    # 2. 移除 token_type_ids (GLM 必须步骤)
                    inputs.pop("token_type_ids", None)

                    # 3. 准备生成参数
                    gen_kwargs = {
                        "max_new_tokens": Config.MAX_NEW_TOKENS,
                        "do_sample": False if Config.TEMPERATURE < 1e-5 else True
                    }
                    if Config.TEMPERATURE >= 1e-5:
                         gen_kwargs["temperature"] = Config.TEMPERATURE

                    # 4. Generate
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                    
                    # 5. Decode
                    if generated_ids is None:
                         raise RuntimeError("Model generation returned None")

                    input_len = inputs['input_ids'].shape[1]
                    generated_ids = generated_ids[:, input_len:]
                    
                    response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    return self.parse_response(response)
                
                except Exception as e:
                    print(f"\n[CRITICAL ERROR] GLM Processing Failed for {image_path}")
                    print(f"Error Message: {e}")
                    traceback.print_exc() # 打印完整堆栈
                    return "error"

            # --- Llama (Llama-3.2-Vision) 逻辑 ---
            elif self.is_llama():
                raw_image = Image.open(image_path).convert("RGB")
                messages = self.build_llama_messages(raw_image, entity)
                
                # Apply chat template
                input_text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                
                # Processor call
                # [关键修改] Align with user code: add_special_tokens=False to correctly process image/text
                inputs = self.processor(
                    raw_image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(Config.DEVICE)
                
                # Generate
                gen_kwargs = {
                    "max_new_tokens": Config.MAX_NEW_TOKENS,
                    "do_sample": False if Config.TEMPERATURE < 1e-5 else True
                }
                if Config.TEMPERATURE >= 1e-5:
                     gen_kwargs["temperature"] = Config.TEMPERATURE
                
                with suppress_output():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode
                input_len = inputs.input_ids.shape[1]
                generated_ids = generated_ids[:, input_len:]
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                return self.parse_response(response)

            # --- LLaVA / 通用逻辑 ---
            else:
                raw_image = Image.open(image_path).convert("RGB")
                messages = self.build_llava_messages(entity)
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=text,
                    images=raw_image,
                    return_tensors="pt"
                ).to(Config.DEVICE)

            # --- 通用 Generate (用于 Qwen/LLaVA) ---
            # 注意：GLM 和 Llama 已在上文 return，不会执行到这里
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id

            with suppress_output():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    temperature=Config.TEMPERATURE,
                    pad_token_id=pad_token_id,
                    do_sample=False if Config.TEMPERATURE < 1e-5 else True
                )
            
            # 解码处理
            input_len = inputs.input_ids.shape[1]
            generated_ids = generated_ids[:, input_len:]
            
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return self.parse_response(response)
        
        except Exception as e:
            # 捕获通用逻辑中的错误
            error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
            if hasattr(tqdm, "write"):
                tqdm.write(error_msg)
            else:
                print(error_msg)
            # 在通用错误中也打印一些堆栈，如果是 'NoneType' 相关的
            if "NoneType" in str(e):
                 print("[DEBUG] Triggering traceback for NoneType error in generic block:")
                 traceback.print_exc()
            return "error"
    
    def process_folder(self, folder_path: str, force_run: bool = False):
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split("_")
        entity = parts[-1] if len(parts) > 0 else "object"
        
        # 使用统一的扩展名定义
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(self.VALID_EXTS)]
        
        if not image_files:
            return

        try:
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        except:
            image_files.sort()
        
        output_file = os.path.join(folder_path, f"{Config.RESULT_PREFIX}{folder_name}.json")
        
        results = {
            "metadata": {
                "model": getattr(Config, "MODEL_NAME", "unknown"),
                "device": Config.DEVICE,
                "entity": entity,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": []
        }
        
        processed_images = set()
        if os.path.exists(output_file):
            if not force_run:
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, dict) and "results" in existing_data:
                            results["results"] = existing_data["results"]
                            processed_images = {item["image"] for item in existing_data["results"]}
                            if Config.VERBOSE:
                                print(f"Resuming task. Found {len(processed_images)} records.")
                except Exception:
                    pass
            else:
                if Config.VERBOSE:
                    print(f"Force run enabled. Overwriting existing file.")

        processed_count = 0
        iterator = tqdm(image_files, desc=f"Processing {folder_name}", unit="img", disable=not Config.VERBOSE)
        
        for img_file in iterator:
            if not force_run and img_file in processed_images:
                continue

            img_path = os.path.join(folder_path, img_file)
            answer = self.process_single_image(img_path, entity)
            
            if answer != "error":
                results["results"].append({
                    "image": img_file,
                    "answer": answer,
                    "timestamp": int(time.time())
                })
                processed_count += 1
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            if Config.VERBOSE and processed_count > 0:
                msg = f"Saved results to {output_file}. Newly processed: {processed_count}"
                if hasattr(iterator, "write"):
                    iterator.write(msg)
                else:
                    print(msg)
        except Exception as e:
            print(f"Error saving results: {e}")