from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

import app.utils.gpu as gpu_utils
from app.utils.logger import get_logger

logger = get_logger(__name__)

DIFFUSION_MODEL_MAP: Dict[str, str] = {
    "instruct_pix2pix": "timbrooks/instruct-pix2pix",
    "stable_diffusion": "runwayml/stable-diffusion-v1-5",
    "img2img": "runwayml/stable-diffusion-v1-5",
    "inpaint": "runwayml/stable-diffusion-v1-5",
}


class DiffusionService:
    def __init__(self, device: Any) -> None:
        self.device = device
        self.pipeline: Optional[Any] = None
        self.model_type: str = ""
        self.dtype: Optional[Any] = None

    def load_model(
        self,
        model_path: str = "",
        model_type: str = "instruct_pix2pix",
    ) -> None:
        import torch
        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionPipeline,
        )

        self.model_type = model_type
        self.dtype = torch.float16 if getattr(self.device, "type", "") == "cuda" else torch.float32
        model_id = model_path or DIFFUSION_MODEL_MAP.get(model_type, "")
        if not model_id:
            raise ValueError(f"Unknown diffusion model type: {model_type}")

        logger.info("Loading diffusion model: %s", model_id)
        logger.info("GPU memory before loading diffusion: %s", gpu_utils.gpu_memory_usage())

        safety_checker = None

        if model_type == "instruct_pix2pix":
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=safety_checker,
                requires_safety_checker=False,
            )
        elif model_type == "inpaint":
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=safety_checker,
                requires_safety_checker=False,
            )
        elif model_type in ("stable_diffusion", "img2img"):
            if model_type == "img2img":
                self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    safety_checker=safety_checker,
                    requires_safety_checker=False,
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    safety_checker=safety_checker,
                    requires_safety_checker=False,
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        is_cuda = getattr(self.device, "type", "") == "cuda"

        if is_cuda:
            total_gb = getattr(gpu_utils, "get_total_memory", lambda: 0.0)()
            # On small GPUs (<6 GB) use model CPU offloading — keeps weights on CPU,
            # moves only the active component to GPU. This adds ~20% overhead but
            # allows models that exceed VRAM to run.
            if total_gb < 6:
                logger.info(
                    "GPU has %.1f GB total — enabling model CPU offload to fit diffusion",
                    total_gb,
                )
                self.pipeline.enable_attention_slicing()
                if hasattr(self.pipeline, "enable_vae_slicing"):
                    self.pipeline.enable_vae_slicing()
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    logger.info("Enabled sequential CPU offload for diffusion")
                except Exception:
                    self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
                self.pipeline.enable_attention_slicing()
        else:
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()

        self.pipeline.set_progress_bar_config(disable=True)
        logger.info("Diffusion model loaded on %s", self.device)

    def process(
        self,
        operation: str,
        image: Image.Image,
        params: Dict[str, Any],
    ) -> Image.Image:
        if self.pipeline is None:
            raise RuntimeError("Diffusion model not loaded. Call load_model() first.")

        import torch
        with torch.inference_mode():
            if operation == "remove":
                if "mask" in params:
                    return self._inpaint_with_mask(image, params)
                return self._inpaint_remove(image, params)
            elif operation in ("replace_background",):
                if "mask" in params:
                    return self._inpaint_with_mask(image, params)
                return self._inpaint_replace_background(image, params)
            elif operation in ("change_style", "style_transfer"):
                return self._apply_style(image, params)
            else:
                raise ValueError(f"Unsupported operation for diffusion: {operation}")

    def _inpaint_remove(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        image = image.convert("RGB")
        prompt = params.get("prompt", "empty background, remove subject, clean")
        negative = params.get("negative_prompt", "object, subject, person, thing")
        guidance = params.get("guidance_scale", 7.5)
        strength = params.get("strength", 0.85)

        result = self.pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=params.get("steps", 30),
            guidance_scale=guidance,
            negative_prompt=negative,
            strength=strength,
        ).images[0]

        return result

    def _inpaint_with_mask(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        image = image.convert("RGB")
        prompt = params.get("prompt", "empty background, remove subject, clean")
        negative = params.get("negative_prompt", "object, subject, person, thing")
        guidance = params.get("guidance_scale", 7.5)
        strength = params.get("strength", 0.85)
        mask = params["mask"]

        if isinstance(mask, Image.Image):
            mask = mask.convert("L")

        result = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=params.get("steps", 30),
            guidance_scale=guidance,
            strength=strength,
            negative_prompt=negative,
        ).images[0]

        return result

    def _inpaint_replace_background(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        instruction = params.get("instruction", "change background")
        guidance = params.get("guidance_scale", 7.5)
        strength = params.get("strength", 0.8)

        if self.model_type == "instruct_pix2pix":
            result = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=params.get("steps", 30),
                guidance_scale=guidance,
                image_guidance_scale=params.get("image_guidance_scale", 1.5),
            ).images[0]
        else:
            result = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=params.get("steps", 30),
                guidance_scale=guidance,
                strength=strength,
            ).images[0]

        return result

    def _apply_style(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        image = image.convert("RGB")
        instruction = params.get("instruction", "apply artistic style")
        guidance = params.get("guidance_scale", 7.5)
        strength = params.get("strength", 0.75)
        negative_prompt = params.get("negative_prompt", "")

        if self.model_type == "instruct_pix2pix":
            result = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=params.get("steps", 30),
                guidance_scale=guidance,
                image_guidance_scale=params.get("image_guidance_scale", 1.5),
                negative_prompt=negative_prompt or None,
            ).images[0]
        else:
            result = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=params.get("steps", 30),
                guidance_scale=guidance,
                strength=strength,
                negative_prompt=negative_prompt or None,
            ).images[0]

        return result

    def load_lora(self, lora_path: str, adapter_name: str = "default") -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")
        path = Path(lora_path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        self.pipeline.load_lora_weights(str(path), adapter_name=adapter_name)
        logger.info("LoRA weights loaded from %s (adapter: %s)", lora_path, adapter_name)

    def unload_lora(self) -> None:
        if self.pipeline is not None:
            self.pipeline.unload_lora_weights()
            logger.info("LoRA weights unloaded")

    def unload(self) -> None:
        if self.pipeline is not None:
            logger.info("Unloading diffusion pipeline")
            del self.pipeline
            self.pipeline = None
            gpu_utils.clear_gpu_aggressive()
