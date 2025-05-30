import torch

class Config:
    # 模型配置
    MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FLASH_ATTN = True
    TORCH_DTYPE = torch.bfloat16
    
    # 图像处理
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28
    
    # 生成参数
    MAX_NEW_TOKENS = 32
    TEMPERATURE = 0.01
    
    # 路径配置
    OUTPUT_DIR = "output"
    RESULT_PREFIX = "QA_"  # 基础前缀
    
    # 调试设置
    VERBOSE = True