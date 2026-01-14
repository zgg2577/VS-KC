import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModel, 
    AutoProcessor,
    AutoTokenizer, 
    AutoConfig
)

# [关键修复] 软导入 Qwen2.5 VL
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

# [关键修复] 软导入 Glm4vForConditionalGeneration (GLM-4V 专用生成类)
try:
    from transformers import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None

# [关键修复] 软导入 MllamaForConditionalGeneration (Llama 3.2 Vision 专用生成类)
try:
    from transformers import MllamaForConditionalGeneration
except ImportError:
    MllamaForConditionalGeneration = None

from .config import Config

def load_components(model_key=None):
    """
    工厂模式加载模型
    args:
        model_key: Config.MODEL_REGISTRY 中的键名
    """
    if model_key is None:
        model_key = Config.DEFAULT_MODEL
      
    if model_key not in Config.MODEL_REGISTRY:
        raise ValueError(f"模型 {model_key} 未在 Config.MODEL_REGISTRY 中定义")

    model_conf = Config.MODEL_REGISTRY[model_key]
    model_path = model_conf["path"]
    model_type = model_conf.get("type", "auto")

    print(f"正在加载模型: {model_key} ({model_type})...")

    # 通用参数
    base_args = {
        "torch_dtype": Config.TORCH_DTYPE,
        "trust_remote_code": True
    }
    
    # [DeepSeek 特殊处理] 
    if model_type == "deepseek":
        base_args["device_map"] = "cuda"
    else:
        base_args["device_map"] = "auto"

    # Flash Attention 适配
    if Config.USE_FLASH_ATTN and "cuda" in Config.DEVICE:
        base_args["attn_implementation"] = "flash_attention_2"

    try:
        # 1. DeepSeek 专用加载逻辑
        if model_type == "deepseek":
            model = AutoModel.from_pretrained(
                model_path,
                use_safetensors=True,
                **base_args
            )
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            if processor.pad_token is None:
                processor.pad_token = processor.eos_token
            
            model.eval()
            return model, processor

        # 2. Qwen 2.5 VL 加载逻辑
        elif model_type == "qwen2.5_vl":
            if Qwen2_5_VLForConditionalGeneration is not None:
                print("Using explicit Qwen2_5_VLForConditionalGeneration class.")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **base_args)
            else:
                print("Warning: Qwen2_5_VLForConditionalGeneration class not found. Falling back to AutoModel.")
                model = AutoModel.from_pretrained(model_path, **base_args)

            processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=Config.MIN_PIXELS,
                max_pixels=Config.MAX_PIXELS,
                trust_remote_code=True
            )
        
        # 3. GLM 加载逻辑 (修复: 优先使用 ForConditionalGeneration)
        elif model_type == "glm":
            model = None
            
            # 尝试 A: 优先使用官方指定的生成类
            if Glm4vForConditionalGeneration is not None:
                try:
                    print("Attempting to load with Glm4vForConditionalGeneration...")
                    model = Glm4vForConditionalGeneration.from_pretrained(model_path, **base_args)
                except Exception as e:
                    print(f"Glm4vForConditionalGeneration failed: {e}")
            
            # 尝试 B: AutoModelForCausalLM (通用生成类)
            if model is None:
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path, **base_args)
                except Exception as e:
                    print(f"AutoModelForCausalLM failed for GLM: {e}")

            # 尝试 C: AutoModel (最后的兜底，但可能没有 generate 方法，通常会报错)
            if model is None:
                print("Warning: Falling back to AutoModel. Note: This might lack .generate() capability!")
                model = AutoModel.from_pretrained(model_path, **base_args)
                
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # 4. [新增] Llama 3.2 Vision 加载逻辑
        # 必须使用 MllamaForConditionalGeneration 才能正确处理视觉参数 (pixel_values)
        elif model_type == "llama_vision":
            # [关键修复] 禁用 Flash Attention 2 
            # Llama 3.2 Vision 在处理复杂 mask/padding 时与 FA2 存在兼容性问题，导致 CUDA Device Assert
            # 我们复制参数并移除 attn_implementation，让其回退到 PyTorch 的 SDPA (稳定)
            llama_args = base_args.copy()
            if "attn_implementation" in llama_args:
                print("Note: Disabling Flash Attention 2 for Llama Vision to prevent CUDA crashes.")
                del llama_args["attn_implementation"]

            if MllamaForConditionalGeneration is not None:
                print("Using explicit MllamaForConditionalGeneration class for Llama Vision.")
                model = MllamaForConditionalGeneration.from_pretrained(model_path, **llama_args)
                
                # [强力修复] 修复 'MllamaVisionAttention' object has no attribute 'is_causal'
                # 使用 named_modules 递归扫描所有层，不依赖特定结构路径
                print("[Llama Patch] Scanning model layers to inject 'is_causal' attribute...")
                patched_count = 0
                try:
                    for name, module in model.named_modules():
                        # 检查类名是否包含 Attention (不论是 VisionAttention 还是其他)
                        if "Attention" in module.__class__.__name__:
                            # 如果确实是 Vision 相关的 Attention，且缺失 is_causal
                            if "Vision" in module.__class__.__name__ or "Cross" in module.__class__.__name__:
                                if not hasattr(module, "is_causal"):
                                    # 视觉/交叉注意力通常非因果
                                    module.is_causal = False 
                                    patched_count += 1
                except Exception as e:
                    print(f"Warning: Failed to apply Llama Vision 'is_causal' patch: {e}")
                
                if patched_count > 0:
                    print(f"[Llama Patch] Successfully patched {patched_count} attention layers.")
                else:
                    print("[Llama Patch] No layers needed patching or patching failed silently.")

            else:
                print("Warning: MllamaForConditionalGeneration class not found. Falling back to AutoModelForCausalLM (Vision inputs might fail!).")
                # 如果没有 Mllama 类，使用 AutoModel 可能会加载成纯文本模型，导致 "model_kwargs not used" 错误
                model = AutoModelForCausalLM.from_pretrained(model_path, **llama_args)
            
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # 5. 通用/LLaVA 加载逻辑 (移除了 llama_vision)
        elif model_type == "auto":
            model = AutoModelForCausalLM.from_pretrained(model_path, **base_args)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **base_args)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        model.eval()
        return model, processor

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"模型 {model_key} 加载失败: {str(e)}")