# llm_Judgment/processor.py
import os
import time
import json
import torch
from PIL import Image
from typing import List, Dict
from .config import Config
from qwen_vl_utils import process_vision_info

class ImageProcessor:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.entity_cache = {}
        
    def build_prompt(self, image_path: str, entity: str) -> List[Dict]:
        """构建多模态提示"""
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Does this image contain {entity}? Answer yes/no only."}
            ]
        }]
    
    def parse_response(self, response: str) -> str:
        """解析模型响应"""
        response = response.lower().strip()
        if response.startswith(("yes", "是")):
            return "yes"
        elif response.startswith(("no", "否")):
            return "no"
        return response[:20]
    
    def process_single_image(self, image_path: str, entity: str) -> str:
        """处理单张图片"""
        try:
            # 构建输入
            messages = self.build_prompt(image_path, entity)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 处理视觉输入
            image_inputs, _ = process_vision_info(messages)
            
            # 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(Config.DEVICE)
            
            # 生成响应
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
            # 解码响应
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return self.parse_response(response)
        
        except Exception as e:
            if Config.VERBOSE:
                print(f"Error processing {image_path}: {str(e)}")
            return "error"
    
    def process_folder(self, folder_path: str):
        """处理整个文件夹"""
        start_time = time.time()
        folder_name = os.path.basename(folder_path)
        entity = folder_name.split("_")[-1]
        
        # 获取排序后的图片
        image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        results = {
            "metadata": {
                "model": Config.MODEL_NAME,
                "device": Config.DEVICE,
                "entity": entity,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": []
        }
        
        # 处理每张图片
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            if Config.VERBOSE:
                print(f"Processing: {img_file}")
            
            answer = self.process_single_image(img_path, entity)
            
            results["results"].append({
                "image": img_file,
                "answer": answer,
                "timestamp": int(time.time())
            })
        
        # 保存结果
        output_file = os.path.join(folder_path, f"{Config.RESULT_PREFIX}{folder_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        if Config.VERBOSE:
            print(f"Processed {len(image_files)} images in {time.time()-start_time:.1f}s")