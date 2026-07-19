from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.services.execution_log import ExecutionLog, ExecutionLogEntry, LogStatus
from app.services.model_manager import ModelManager
from app.services.style_presets import (
    build_style_instruction,
    build_style_params,
    detect_style_preset,
)
from app.utils.gpu import clear_gpu_aggressive, cuda_available, gpu_memory_usage
from app.utils.image_utils import ensure_rgb, image_to_base64, save_image
from app.utils.logger import get_logger

logger = get_logger(__name__)


MAX_DIFFUSION_DIM = 1024


class OperationHandler:
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    @staticmethod
    def _limit_diffusion_size(
        image: Image.Image, alpha: Image.Image | None = None, max_dim: int = MAX_DIFFUSION_DIM
    ) -> tuple[Image.Image, Image.Image | None, tuple[int, int]]:
        w, h = image.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w = max(64, int(w * ratio) // 8 * 8)
            new_h = max(64, int(h * ratio) // 8 * 8)
            new_size = (new_w, new_h)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            if alpha is not None:
                alpha = alpha.resize(new_size, Image.Resampling.LANCZOS)
        return image, alpha, (w, h)

    @staticmethod
    def _diffusion_max_dim() -> int:
        try:
            import app.utils.gpu as gpu_utils

            total_gb = getattr(gpu_utils, "get_total_memory", lambda: 0.0)()
        except Exception:
            return MAX_DIFFUSION_DIM

        if total_gb <= 0:
            return MAX_DIFFUSION_DIM
        if total_gb < 4.5:
            return 512
        if total_gb < 6:
            return 640
        if total_gb < 8:
            return 768
        return MAX_DIFFUSION_DIM

    @staticmethod
    def _is_human_image(image: Image.Image) -> bool:
        try:
            import cv2
            if not hasattr(cv2, "CascadeClassifier"):
                return False
            gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
            max_dim = 640
            h, w = gray.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            detected = len(faces) > 0
            logger.info("Human detection: %s (%d face(s) found)", detected, len(faces))
            return detected
        except Exception:
            return False

    def handle_segment(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        logger.info("Pipeline step: segment target='%s'", params.get("target", ""))
        service = self.model_manager.load_sam()
        result_image, mask = service.segment_object(image, params)

        if context is not None:
            context["mask"] = mask

        if params.get("save_mask", True):
            mask_path = save_image(
                mask, settings.temp_path, prefix="mask_"
            )
            details = {"mask": str(mask_path)}
        else:
            details = {}

        from app.utils.image_utils import image_to_base64
        mask_b64 = image_to_base64(mask.convert("L")) if mask else ""
        details["mask_image"] = mask_b64

        return result_image, details

    @staticmethod
    def _is_person_removal(instruction: str, image: Image.Image) -> bool:
        person_keywords = [
            "person", "people", "human", "man", "woman", "child", "children",
            "guy", "girl", "boy", "pedestrian", "crowd", "someone", "somebody",
            "portrait", "face", "head", "tourist", "visitor", "passenger",
            "worker", "student", "player", "actor", "model", "subject",
        ]
        instruction_lower = instruction.lower()
        if any(kw in instruction_lower for kw in person_keywords):
            return True
        try:
            return OperationHandler._is_human_image(image)
        except Exception:
            return False

    def handle_remove(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        logger.info("Pipeline step: remove")
        mask = context.get("mask") if context else None
        prompt = params.get("instruction", "remove the main subject, clean background")

        is_person = self._is_person_removal(prompt, image)

        if is_person:
            logger.info("Person removal detected - using specialized person mask")
            try:
                person_service = self.model_manager.load_person_segmentation()
                person_mask = person_service.segment_person(image)
                if context is not None:
                    context["mask"] = person_mask
                mask = person_mask
                logger.info("Person mask generated via u2net_human_seg")
            except Exception as exc:
                logger.warning(
                    "Person segmentation failed, using existing mask: %s", exc
                )

        if mask is not None:
            try:
                logger.info("Using LaMa specialized inpainting for removal")
                service = self.model_manager.load_removal()
                process_params: Dict[str, Any] = {"mask": mask}
                result = service.process("remove", image, process_params)
                if result.size != image.size:
                    result = result.resize(image.size, Image.Resampling.LANCZOS)
                return result, {}
            except Exception as exc:
                logger.warning(
                    "LaMa removal failed, falling back to diffusion inpainting: %s",
                    exc,
                )

        image, _, _ = self._limit_diffusion_size(image, max_dim=self._diffusion_max_dim())

        process_params: Dict[str, Any] = {
            "prompt": f"empty background, {prompt}",
            "negative_prompt": "object, subject, person, thing, detail",
            "guidance_scale": 7.5,
            "strength": 0.85,
            "steps": 30,
        }

        if mask is not None:
            logger.info("Fallback: using SAM mask for guided inpainting removal")
            process_params["mask"] = mask
            service = self.model_manager.load_diffusion(model_type="inpaint")
        else:
            logger.info("Fallback: using diffusion-based removal without mask")
            service = self.model_manager.load_diffusion()

        result = service.process("remove", image, process_params)
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result, {}

    def handle_replace_background(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        logger.info("Pipeline step: replace_background with '%s'", params.get("instruction", ""))
        import io
        import base64 as b64_mod
        mask = context.get("mask") if context else None
        from app.utils.image_utils import image_to_base64

        if mask is None:
            logger.info("No mask in context, segmenting main subject for layer decomposition")
            _, mask_details = self.handle_segment(image, {"target": "main subject", "save_mask": False}, context)
            mask_raw = mask_details.get("mask_image")
            if mask_raw and isinstance(mask_raw, str):
                mask = Image.open(io.BytesIO(b64_mod.b64decode(mask_raw)))
            else:
                mask = mask_raw

        process_params: Dict[str, Any] = {
            "instruction": params.get("instruction", "change background"),
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "strength": 0.8,
            "steps": 30,
        }

        if mask is not None:
            logger.info("Using mask for guided background replacement")
            import numpy as np
            inv_mask = Image.fromarray(
                255 - np.array(mask.convert("L")), mode="L"
            )
            process_params["mask"] = inv_mask
            process_params["prompt"] = process_params.pop("instruction")
            service = self.model_manager.load_diffusion(model_type="inpaint")
        else:
            image, _, _ = self._limit_diffusion_size(image, max_dim=self._diffusion_max_dim())
            service = self.model_manager.load_diffusion()

        result = service.process("replace_background", image, process_params)
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)

        result_b64 = image_to_base64(result)
        layers = []

        if mask is not None:
            mask_l = mask.convert("L")
            if mask_l.size != image.size:
                mask_l = mask_l.resize(image.size, Image.Resampling.LANCZOS)
            orig_rgba = image.convert("RGBA")
            fg_r, fg_g, fg_b, _ = orig_rgba.split()
            fg_layer = Image.merge("RGBA", (fg_r, fg_g, fg_b, mask_l))
            fg_b64 = image_to_base64(fg_layer)

            result_rgba = result.convert("RGBA")
            if result_rgba.size != image.size:
                result_rgba = result_rgba.resize(image.size, Image.Resampling.LANCZOS)
            bg_r_arr = np.array(result_rgba.split()[0], dtype=np.float32)
            bg_g_arr = np.array(result_rgba.split()[1], dtype=np.float32)
            bg_b_arr = np.array(result_rgba.split()[2], dtype=np.float32)
            orig_r_arr = np.array(orig_rgba.split()[0], dtype=np.float32)
            orig_g_arr = np.array(orig_rgba.split()[1], dtype=np.float32)
            orig_b_arr = np.array(orig_rgba.split()[2], dtype=np.float32)
            msk_arr = np.array(mask_l, dtype=np.float32) / 255.0
            eps = 1e-8
            inv_msk_arr = 1.0 - msk_arr
            bg_r = np.clip((bg_r_arr - orig_r_arr * msk_arr) / (inv_msk_arr + eps), 0, 255).astype(np.uint8)
            bg_g = np.clip((bg_g_arr - orig_g_arr * msk_arr) / (inv_msk_arr + eps), 0, 255).astype(np.uint8)
            bg_b = np.clip((bg_b_arr - orig_b_arr * msk_arr) / (inv_msk_arr + eps), 0, 255).astype(np.uint8)
            inv_msk = Image.fromarray(np.clip(inv_msk_arr * 255, 0, 255).astype(np.uint8), mode="L")
            bg_only = Image.merge("RGBA", (
                Image.fromarray(bg_r, mode="L"),
                Image.fromarray(bg_g, mode="L"),
                Image.fromarray(bg_b, mode="L"),
                inv_msk,
            ))
            bg_b64 = image_to_base64(bg_only)

            layers = [
                {"id": "foreground", "name": "Subject", "image": fg_b64, "layer_type": "foreground", "visible": True},
                {"id": "background", "name": "New Background", "image": bg_b64, "layer_type": "background", "visible": True},
            ]
        else:
            layers = [{"id": "result", "name": "Result", "image": result_b64, "layer_type": "composite", "visible": True}]

        return result, {"layers": layers}

    def _fallback_remove_bg(self, image: Image.Image, context: Optional[Dict[str, Any]] = None) -> Image.Image:
        logger.warning("rembg failed. Falling back to SAM-based background removal.")
        if context is None or "mask" not in context:
            _, mask_details = self.handle_segment(image, {"target": "main subject", "save_mask": False}, context)
            if context is None:
                context = {}
            mask = mask_details.get("mask_image")
            if mask is None:
                return image
        else:
            mask = context.get("mask")
        if mask is None:
            return image
        from PIL import ImageFilter
        mask = mask.convert("L").resize(image.size)
        mask = mask.filter(ImageFilter.SMOOTH)
        mask = mask.point(lambda p: 255 if p > 60 else 0)
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask)
        return image_rgba

    def handle_remove_background(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        is_human = self._is_human_image(image)
        if is_human:
            logger.info("Pipeline step: remove_background (human-optimized path)")
        else:
            logger.info("Pipeline step: remove_background (general-object path)")

        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            import rembg
            import numpy as np
            from PIL import ImageFilter

            if is_human:
                session = rembg.new_session("u2net_human_seg", providers=["CPUExecutionProvider"])
                logger.info("Using rembg model: u2net_human_seg (CPU)")
            else:
                session = rembg.new_session("isnet-general-use", providers=["CPUExecutionProvider"])
                logger.info("Using rembg model: isnet-general-use (CPU)")

            try:
                result = rembg.remove(
                    image,
                    session=session,
                    post_process_mask=True,
                )
            except Exception as rembg_err:
                err_str = str(rembg_err)
                if "Half" in err_str or "float" in err_str.lower() or "dtype" in err_str.lower():
                    logger.warning("rembg dtype error, falling back to SAM: %s", err_str)
                    result_rgba = self._fallback_remove_bg(image, context)
                    from app.utils.image_utils import image_to_base64 as _i2b
                    fb_b64 = _i2b(result_rgba)
                    return result_rgba, {"layers": [{"id": "subject", "name": "Subject", "image": fb_b64, "layer_type": "foreground", "visible": True}]}
                raise

            if result.size != image.size:
                result = result.resize(image.size, Image.Resampling.LANCZOS)

            logger.debug("rembg result size after resize: %s", result.size)
            logger.debug("image size: %s", image.size)

            ai_alpha = np.array(result.split()[3])

            dilation = 5 if is_human else 3
            alpha_img = Image.fromarray(ai_alpha, mode="L")
            alpha_img = alpha_img.filter(ImageFilter.MaxFilter(dilation))
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=1.5))

            alpha_np = np.array(alpha_img, dtype=np.float32)
            alpha_np = np.where(alpha_np > 240, 255.0, alpha_np)
            alpha_np = np.where(alpha_np < 15, 0.0, alpha_np)
            alpha_img = Image.fromarray(alpha_np.astype(np.uint8), mode="L")

            orig_rgba = image.convert("RGBA")
            r_orig, g_orig, b_orig, _ = orig_rgba.split()
            result = Image.merge("RGBA", (r_orig, g_orig, b_orig, alpha_img))

            del session
            del ai_alpha, alpha_img, alpha_np
            import gc
            gc.collect()
            gc.collect()
            try:
                import torch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

            from app.utils.image_utils import image_to_base64
            subject_b64 = image_to_base64(result)
            layers = [{
                "id": "subject",
                "name": "Subject",
                "image": subject_b64,
                "layer_type": "foreground",
                "visible": True,
            }]
            return result, {"layers": layers}
        except ImportError:
            logger.warning("rembg is not installed. Falling back to SAM (which has aliased edges).")
            if context is None or "mask" not in context:
                img_temp, mask_details = self.handle_segment(image, {"target": "main subject"}, context)
                if context is None:
                    context = {}
                if "mask" in mask_details:
                    context["mask"] = Image.open(mask_details["mask"])

            mask = context.get("mask")
            if mask is None:
                return image, {}

            from app.utils.image_utils import image_to_base64 as _i2b
            image_rgba = image.convert("RGBA")
            mask_l = mask.convert("L").resize(image_rgba.size)
            image_rgba.putalpha(mask_l)
            rgba_b64 = _i2b(image_rgba)
            return image_rgba, {"layers": [{"id": "subject", "name": "Subject", "image": rgba_b64, "layer_type": "foreground", "visible": True}]}

    def handle_style_transfer(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        alpha = None
        if image.mode == "RGBA":
            alpha = image.split()[3]
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image.convert("RGB"), mask=alpha)
            image = background

        image, alpha, orig_size = self._limit_diffusion_size(image, alpha, max_dim=self._diffusion_max_dim())

        logger.info("Pipeline step: style_transfer with '%s'", params.get("instruction", ""))
        style_preset = detect_style_preset(params.get("instruction", ""), params)
        service = self.model_manager.load_diffusion()
        if style_preset == "ghibli":
            ghibli_lora = getattr(settings, "ghibli_lora_weights", "")
            if ghibli_lora:
                adapter_name = getattr(settings, "ghibli_lora_adapter_name", "ghibli") or "ghibli"
                try:
                    service.load_lora(ghibli_lora, adapter_name=adapter_name)
                    logger.info("Applied style-specific LoRA preset: %s", style_preset)
                except Exception as exc:
                    logger.warning("Failed to load %s LoRA preset: %s", style_preset, exc)

        style_params = build_style_params(style_preset)
        instruction = build_style_instruction(params.get("instruction", "apply artistic style"), style_preset)
        process_params = {
            "instruction": instruction,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "strength": 0.75,
            "steps": 30,
        }
        process_params.update(style_params)
        for key in ("guidance_scale", "image_guidance_scale", "strength", "steps", "negative_prompt"):
            if key in params and params[key] is not None:
                process_params[key] = params[key]
        result = service.process("style_transfer", image, process_params)

        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)

        if alpha is not None:
            result = result.convert("RGBA")
            result.putalpha(alpha)

        return result, {}

    def handle_change_style(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        logger.info("Pipeline step: change_style with '%s'", params.get("instruction", ""))
        return self.handle_style_transfer(image, params, context)

    def handle_upscale(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        logger.info("Pipeline step: upscale")
        service = self.model_manager.load_esrgan()
        result = service.upscale(image)
        return result, {}


class PipelineExecutor:
    def __init__(self) -> None:
        import asyncio
        self.model_manager = ModelManager(
            device=settings.resolved_device,
            cache_timeout_minutes=settings.model_cache_timeout_minutes,
        )
        self.handler = OperationHandler(self.model_manager)
        self._lock = asyncio.Lock()
        self._operation_map = {
            "segment": self.handler.handle_segment,
            "remove": self.handler.handle_remove,
            "replace_background": self.handler.handle_replace_background,
            "remove_background": self.handler.handle_remove_background,
            "style_transfer": self.handler.handle_style_transfer,
            "change_style": self.handler.handle_change_style,
            "upscale": self.handler.handle_upscale,
        }

    async def execute(
        self,
        plan: List[Dict[str, Any]],
        input_image: Image.Image,
        job_id: str,
    ) -> Tuple[List[Dict[str, Any]], ExecutionLog]:
        import asyncio
        async with self._lock:
            return await asyncio.to_thread(self._execute_sync, plan, input_image, job_id)

    def _get_model_name(self, op_name: str, params: Dict[str, Any]) -> str:
        if op_name == "segment":
            return "sam-vit-base"
        elif op_name == "remove":
            return "lama-inpaint"
        elif op_name == "replace_background":
            if params.get("mask") is not None:
                return "sd-inpaint"
            return "instruct-pix2pix"
        elif op_name in ("style_transfer", "change_style"):
            return "instruct-pix2pix"
        elif op_name == "upscale":
            return "esrgan"
        elif op_name == "remove_background":
            return "rembg"
        return ""

    def _build_log_entry(
        self,
        op_name: str,
        params: Dict[str, Any],
        status: LogStatus,
        duration: float,
        reason: str = "",
        model: str = "",
    ) -> ExecutionLogEntry:
        target = ""
        if op_name == "segment":
            target = params.get("target", "")
        elif op_name in ("replace_background", "remove_background"):
            target = "background"

        return ExecutionLogEntry(
            operation=op_name,
            target=target,
            model=model or self._get_model_name(op_name, params),
            parameters=params,
            status=status,
            reason=reason,
            duration=duration,
        )

    def _compute_mask(self, before: Image.Image, after: Image.Image) -> Image.Image:
        if after.size != before.size:
            after = after.resize(before.size, Image.Resampling.LANCZOS)
        before_np = np.array(before.convert("RGB"), dtype=np.float32)
        after_np = np.array(after.convert("RGB"), dtype=np.float32)
        diff = np.abs(before_np - after_np)
        diff_gray = diff.mean(axis=2)
        threshold = 5.0
        mask_np = np.where(diff_gray > threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(mask_np, mode="L")

    def _cleanup_gpu(self) -> None:
        gc.collect()
        if cuda_available():
            clear_gpu_aggressive()

    def _execute_sync(
        self,
        plan: List[Dict[str, Any]],
        input_image: Image.Image,
        job_id: str,
    ) -> Tuple[List[Dict[str, Any]], ExecutionLog]:
        logger.info("Pipeline executing plan with %d steps for job %s", len(plan), job_id)
        current_image = input_image
        if current_image.mode != "RGBA":
            current_image = ensure_rgb(input_image)
        steps: List[Dict[str, Any]] = []
        execution_log = ExecutionLog()
        context: Dict[str, Any] = {}

        for step_index, operation in enumerate(plan):
            op_name = operation["operation"]
            params = {k: v for k, v in operation.items() if k != "operation"}

            logger.info(
                "Executing step %d/%d: %s",
                step_index + 1, len(plan), op_name,
            )
            step_start = time.monotonic()

            try:
                before_image = current_image.copy()

                handler = self._operation_map.get(op_name)
                if handler is None:
                    raise ValueError(f"No handler registered for operation: {op_name}")

                current_image, details = handler(current_image, params, context)
                if current_image.mode != "RGBA":
                    current_image = ensure_rgb(current_image)

                mask = details.pop("mask_image", None) if details else None
                if mask is None:
                    mask = self._compute_mask(before_image, current_image)
                else:
                    if isinstance(mask, Image.Image):
                        mask = mask.convert("L")
                    else:
                        mask = self._compute_mask(before_image, current_image)
                mask_b64 = image_to_base64(mask)

                intermediate = save_image(
                    current_image,
                    settings.output_path / job_id,
                    prefix=f"step_{step_index:02d}_",
                )

                step_duration = (time.monotonic() - step_start) * 1000
                step_entry = {
                    "operation": op_name,
                    "image": image_to_base64(current_image),
                    "mask": mask_b64,
                    "duration_ms": round(step_duration, 2),
                    "details": details if details else None,
                }
                steps.append(step_entry)

                log_entry = self._build_log_entry(
                    op_name=op_name,
                    params=params,
                    status=LogStatus.SUCCESS,
                    duration=step_duration,
                )
                execution_log.append(log_entry)

                logger.info(
                    "Step %d/%d complete: %s (%.0f ms)",
                    step_index + 1, len(plan), op_name, step_duration,
                )

                if context.get("mask") is not None and op_name != "segment":
                    context.pop("mask", None)

                self.model_manager.unload_current()
                self._cleanup_gpu()

            except Exception as exc:
                step_duration = (time.monotonic() - step_start) * 1000
                logger.error(
                    "Pipeline step %d (%s) failed: %s",
                    step_index + 1, op_name, exc,
                )
                error_step = {
                    "operation": op_name,
                    "image": image_to_base64(current_image),
                    "duration_ms": round(step_duration, 2),
                    "error": str(exc),
                }
                steps.append(error_step)

                log_entry = self._build_log_entry(
                    op_name=op_name,
                    params=params,
                    status=LogStatus.FAILED,
                    duration=step_duration,
                    reason=str(exc),
                )
                execution_log.append(log_entry)

                self.model_manager.unload_current()
                self._cleanup_gpu()
                raise RuntimeError(
                    f"Pipeline failed at step {step_index + 1} ({op_name}): {exc}"
                ) from exc

        self.model_manager.unload_current()
        self._cleanup_gpu()

        logger.info(
            "Pipeline complete for job %s: %d steps executed",
            job_id, len(steps),
        )
        return steps, execution_log
