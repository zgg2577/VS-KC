# sd_inpainting/core.py
import torch
import logging
from PIL import Image, ImageDraw
import numpy as np
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
)

# 禁用非必要日志
logging.getLogger("diffusers").setLevel(logging.ERROR)


class InpaintingEngine:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """稳定版模型加载"""
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                variant="fp16" if "cuda" in self.device else None,
                use_safetensors=True,
            )
            self.mode = "inpainting"
        except Exception:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                variant="fp16" if "cuda" in self.device else None,
                use_safetensors=True,
            )
            self.mode = "img2img"

        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(
        self,
        image,
        mask_coords=None,
        prompt="",
        negative_prompt="",
        strength=0.8,
        guidance_scale=7.5,
        num_inference_steps=30,
        mask_image=None,
    ):
        """稳定版生成接口"""
        # 创建蒙版
        if mask_image:
            mask = mask_image.convert("L")
        elif mask_coords:
            mask = self._create_mask(image.size, mask_coords)
        else:
            raise ValueError("必须提供mask_image或mask_coords")

        # 统一图像格式
        image = image.convert("RGB")

        # 执行生成
        if self.mode == "inpainting":
            return self._classic_inpainting(
                image,
                mask,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
            )
        else:
            return self._classic_img2img(
                image,
                mask,
                prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
            )

    def _classic_inpainting(
        self, image, mask, prompt, negative_prompt, strength, guidance_scale, steps
    ):
        """经典修复模式"""
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            output_type="pil",
        ).images[0]

    def _classic_img2img(
        self, image, mask, prompt, negative_prompt, strength, guidance_scale, steps
    ):
        """经典图像混合模式"""
        # 生成噪声混合图像
        img_array = np.array(image)
        mask_array = np.array(mask)

        # 创建3通道蒙版
        mask_3d = np.stack([mask_array] * 3, axis=2) / 255.0

        # 生成符合图像尺寸的噪声
        noise = np.random.randint(0, 255, img_array.shape, dtype=np.uint8)
        modified_img = img_array * (1 - mask_3d) + noise * mask_3d

        # 执行生成
        generated = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=Image.fromarray(modified_img),
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).images[0]

        # 混合结果
        final_array = np.array(image) * (1 - mask_3d) + np.array(generated) * mask_3d
        return Image.fromarray(final_array.astype(np.uint8))

    def _create_mask(self, image_size, coords):
        """创建抗锯齿蒙版"""
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(coords, fill=255, width=0)  # 消除锯齿
        return mask
