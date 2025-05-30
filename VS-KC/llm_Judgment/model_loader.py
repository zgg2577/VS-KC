# llm_Judgment/model_loader.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .config import Config

def load_components():
    """加载模型和处理器"""
    model_args = {
        "torch_dtype": Config.TORCH_DTYPE,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    if Config.USE_FLASH_ATTN and "cuda" in Config.DEVICE:
        model_args["attn_implementation"] = "flash_attention_2"
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME,
            **model_args
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            Config.MODEL_NAME,
            min_pixels=Config.MIN_PIXELS,
            max_pixels=Config.MAX_PIXELS
        )
        
        return model, processor
    except Exception as e:
        raise RuntimeError(f"组件加载失败: {str(e)}")