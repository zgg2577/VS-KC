import torch
from pathlib import Path


class Config:
    # 运行模式配置
    USE_API = False
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒

    # 错误处理
    ERROR_PLACEHOLDER = "[ERROR]"

    # 本地模型配置
    MODEL_NAME = " "

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16
    USE_FLASH_ATTN = True

    # LoRA 配置 (如果为空字符串则不加载)
    LORA_ADAPTER_PATH = ""

    # 本地模型生成参数配置
    GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "do_sample":False ,
    }

    # 图像处理
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1280 * 28 * 28

    # 提示模板 (Safety Detection System)
    # Stage 1: 开放式询问
    FIRST_PROMPT = "This is a picture of an operating room. Are there any unsafe or unreasonable factors? Answer briefly."

    # Stage 2: 基于第一轮回复的针对性询问 (多模态上下文)
    SECOND_PROMPT = "Based **only** on the first response, does it recognize hazards from a {entity}? Answer strictly with 'Yes', 'No', or 'Not mentioned'."

    # Stage 3: 纯文本一致性检查
    THIRD_PROMPT = "Based **only** on the first response text provided, does it recognize safety hazards related to the entity '{entity}'? Answer strictly with 'Yes', 'No', or 'Not mentioned'."

    # [新增] 调试输出配置 (修复 AttributeError)
    TOP_N_FOLDERS = None  # 默认处理全部，None 表示不限制
    DEBUG_OUTPUT = False

    # 路径配置
    OUTPUT_DIR = str(Path("./debug_results" if DEBUG_OUTPUT else "./results").resolve())
    ENTITY_REPORT = "entity_stats.csv"
    FOLDER_REPORT = "folder_stats.csv"

    # 支持的图片格式
    VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

