import torch
from pathlib import Path


class Config:
    # 运行模式配置
    USE_API = True
    API_ENDPOINT = "xxxx"
    API_KEY = "xxx"
    API_MODEL = "xxx"
    API_TIMEOUT = 30
    # 重试策略
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒

    # 错误处理
    ERROR_PLACEHOLDER = "[ERROR]"

    # 本地模型配置
    MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16
    USE_FLASH_ATTN = True

    # 在原有配置中添加
    LORA_ADAPTER_PATH = ""  # 您的LoRA适配器路径
    # LORA_ADAPTER_PATH = "lora_outputs/Qwen2.5-VL-LoRA-pf/epoch_2"  # 您的LoRA适配器路径
    # LORA_ADAPTER_PATH = "lora_outputs/Qwen2.5-VL-LoRA-dataset/epoch_0"  # 您的LoRA适配器路径

    # 本地模型生成参数配置
    GENERATION_CONFIG = {
        "max_new_tokens": 512,  # 最大输出token数
        "temperature": 0.7,  # 生成温度
        "top_p": 0.9,  # 核采样参数
        "do_sample": True,  # 是否采样
        "repetition_penalty": 1.1,  # 重复惩罚
    }

    # API专用参数（如果使用API模式）
    API_GENERATION_CONFIG = {"max_tokens": 512, "temperature": 0.7}

    # 图像处理
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28

    # 提示模板
    FIRST_PROMPT = "This is a picture of an operating room. Are there any unsafe or unreasonable factors? Answer briefly."
    SECOND_PROMPT = "Based **only** on the first response, does it recognize hazards from a {entity}? Answer strictly with 'Yes', 'No', or 'Not mentioned"
    THIRD_PROMPT = "Based **only** on the first response text provided, does it recognize safety hazards related to the entity '{entity}'? Answer strictly with 'Yes', 'No', or 'Not mentioned'."
    ERROR_PLACEHOLDER = "Error or no response"

    # 路径配置
    OUTPUT_DIR = "results"
    ENTITY_REPORT = "entity_stats.csv"
    FOLDER_REPORT = "folder_stats.csv"

    # 调试输出配置
    TOP_N_FOLDERS = None  # 默认处理全部
    MIN_CONVERSATION_LENGTH = 4  # 最小对话长度
    DEBUG_OUTPUT = True  # 是否输出调试文件/True/False

    # 输出目录配置
    OUTPUT_DIR = str(Path("./debug_results" if DEBUG_OUTPUT else "./results").resolve())
