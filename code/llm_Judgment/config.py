import torch


class Config:
    # --- 模型注册表 ---
    MODEL_REGISTRY = {
        "Qwen2.5-VL": {"path": " ", "type": "qwen2.5_vl", "prefix": "Qwen_"},
        "deepseek": {"path": " ", "type": "deepseek", "prefix": "deepseek_"},
        "GLM": {"path": " ", "type": "glm", "prefix": "GLM_"},
        "llama": {"path": " ", "type": "llama_vision", "prefix": "llama_"},
    }

    # 默认使用的模型
    DEFAULT_MODEL = "deepseek"
    MODEL_NAME = DEFAULT_MODEL

    # --- 通用系统配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # [关键] DeepSeek/FlashAttn2 必须使用 bfloat16
    USE_FLASH_ATTN = True
    TORCH_DTYPE = torch.bfloat16

    # 图像处理参数
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28

    # 生成参数
    MAX_NEW_TOKENS = 50
    TEMPERATURE = 0.01

    # 路径配置
    OUTPUT_DIR = "output"
    RESULT_PREFIX = "QA_"

    VERBOSE = True
