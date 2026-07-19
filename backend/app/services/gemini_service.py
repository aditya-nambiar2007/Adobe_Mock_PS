from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

from google import genai

from app.services.style_presets import build_style_instruction, detect_style_preset
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiError(Exception):
    def __init__(self, message: str, status_code: int = 502, code: str = "GEMINI_ERROR") -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


SYSTEM_PROMPT = """You are an expert image editing planner. Given a user's natural language prompt describing edits to an image, you must output ONLY a valid JSON array of operations.

Each operation object must follow one of these schemas:

{"operation": "segment", "target": "<object to segment>"}
{"operation": "remove"}
{"operation": "replace_background", "new_background": "<description>"}
{"operation": "remove_background"}
{"operation": "change_style", "instruction": "<style description>"}
{"operation": "style_transfer", "instruction": "<style description>"}
{"operation": "upscale"}

Rules:
- Output ONLY the JSON array, no markdown, no explanation.
- Break complex prompts into multiple sequential operations.
- Use "segment" first if the prompt mentions specific objects.
- Use "remove" for object removal (requires prior segment).
- Use "replace_background" to generate a new background using AI.
- Use "remove_background" if the user wants to make the background transparent.
- Use "change_style" or "style_transfer" for artistic transformations.
- End with "upscale" if the final output should be high resolution.
- Maximum 8 operations per plan.
- Every operation after the first uses the previous result as input."""


class GeminiService:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite") -> None:
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate_plan(
        self,
        prompt: str,
        metadata: Dict[str, Any] | None = None,
        options: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        logger.info("Sending prompt to Gemini (%s): %.80s...", self.model, prompt)
        start = time.monotonic()

        style_preset = detect_style_preset(prompt, options)
        style_context = ""
        if style_preset == "ghibli":
            style_context = (
                "\n\nStyle hint:\n"
                "If the user asks for a Ghibli or Ghibli-inspired result, prefer a single "
                "change_style/style_transfer operation with a detailed cinematic animation prompt. "
                "Preserve scene composition and subject identity unless the user explicitly asks for more."
            )
            prompt = build_style_instruction(prompt, style_preset)

        metadata_context = ""
        if metadata:
            metadata_context = f"\n\nImage metadata: {json.dumps(metadata, ensure_ascii=False)}"

        full_prompt = (
            f"{SYSTEM_PROMPT}{style_context}{metadata_context}\n\n"
            f"User prompt: {prompt}\n\nOutput the JSON plan:"
        )

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=full_prompt,
                config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                },
            )
        except Exception as exc:
            error_str = str(exc)
            logger.error("Gemini API call failed: %.200s", error_str)

            if "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                raise GeminiError(
                    message=f"Gemini API quota exceeded for '{self.model}'. Wait for reset or use a different key.",
                    status_code=429,
                    code="GEMINI_QUOTA_EXCEEDED",
                ) from exc
            if "403" in error_str or "PERMISSION_DENIED" in error_str:
                raise GeminiError(
                    message="Gemini API access denied. Check your API key.",
                    status_code=403,
                    code="GEMINI_FORBIDDEN",
                ) from exc
            if "404" in error_str or "NOT_FOUND" in error_str:
                raise GeminiError(
                    message=f"Gemini model '{self.model}' not found. Check the model name.",
                    status_code=404,
                    code="GEMINI_MODEL_NOT_FOUND",
                ) from exc

            raise GeminiError(
                message=f"Gemini API error: {error_str[:300]}",
                status_code=502,
                code="GEMINI_API_ERROR",
            ) from exc

        latency = (time.monotonic() - start) * 1000
        logger.info("Gemini responded in %.0f ms", latency)

        plan = self._parse_response(response)

        if not plan:
            raise GeminiError(
                message="Gemini returned an empty or invalid plan",
                status_code=502,
                code="GEMINI_INVALID_RESPONSE",
            )

        logger.info("Generated plan with %d steps", len(plan))
        return plan

    def _parse_response(self, response: Any) -> List[Dict[str, Any]]:
        try:
            text = response.text.strip()
        except AttributeError:
            logger.warning("Gemini response has no text attribute")
            return []

        if not text:
            return []

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "operations" in parsed:
                return parsed["operations"]
            if isinstance(parsed, dict) and "plan" in parsed:
                return parsed["plan"]
            logger.warning("Unexpected Gemini response structure: %s", type(parsed))
            return []
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON: %s", exc)
            logger.debug("Raw text: %.200s", text)
            return []

    async def close(self) -> None:
        self.client = None
