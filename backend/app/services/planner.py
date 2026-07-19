from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from app.services.style_presets import (
    build_style_instruction,
    detect_style_preset,
)
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.services.gemini_service import GeminiService

logger = get_logger(__name__)

VALID_OPERATIONS = frozenset({
    "segment",
    "remove",
    "replace_background",
    "remove_background",
    "change_style",
    "style_transfer",
    "upscale",
})

OPERATIONS_REQUIRING_TARGET = frozenset({"segment"})
OPERATIONS_REQUIRING_INSTRUCTION = frozenset({
    "change_style",
    "style_transfer",
    "replace_background",
})


class Planner:
    def __init__(self, gemini_service: GeminiService | None) -> None:
        self.gemini_service = gemini_service

    async def create_plan(
        self,
        prompt: str,
        metadata: Dict[str, Any] | None = None,
        options: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        style_preset = detect_style_preset(prompt, options)
        if not self.gemini_service:
            # Offline Fallback Mode
            p = prompt.lower()
            if style_preset == "ghibli":
                return [{
                    "operation": "change_style",
                    "instruction": build_style_instruction(prompt, style_preset),
                }]
            if "remove background" in p or "transparent" in p:
                return [{"operation": "segment", "target": "main subject"}, {"operation": "remove"}]
            if "upscale" in p:
                return [{"operation": "upscale"}]
            from app.services.gemini_service import GeminiError
            raise GeminiError("AI planning disabled (missing API key). Only basic offline commands like 'remove background' are supported.", status_code=503, code="OFFLINE_MODE")

        raw_plan = await self.gemini_service.generate_plan(prompt, metadata=metadata, options=options)
        validated = self._validate(raw_plan)
        logger.info("Plan validated: %d operations", len(validated))
        return validated

    def _validate(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(plan, list):
            raise ValueError("Plan must be a JSON array")

        if len(plan) == 0:
            raise ValueError("Plan must contain at least one operation")

        if len(plan) > 8:
            raise ValueError("Plan exceeds maximum of 8 operations")

        validated: List[Dict[str, Any]] = []
        for i, op in enumerate(plan):
            validated_op = self._validate_operation(op, i)
            validated.append(validated_op)

        return validated

    def _validate_operation(self, op: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(op, dict):
            raise ValueError(f"Operation {index} must be an object")

        operation = op.get("operation", "")
        if not operation:
            raise ValueError(f"Operation {index} is missing 'operation' field")
        if operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operation '{operation}' at index {index}. "
                f"Valid: {', '.join(sorted(VALID_OPERATIONS))}"
            )

        result: Dict[str, Any] = {"operation": operation}

        if operation in OPERATIONS_REQUIRING_TARGET:
            target = op.get("target", "")
            if not target or not isinstance(target, str):
                raise ValueError(
                    f"Operation '{operation}' at index {index} requires a non-empty 'target' field"
                )
            result["target"] = target.strip()

        if operation in OPERATIONS_REQUIRING_INSTRUCTION:
            instruction = (
                op.get("instruction", "") or op.get("new_background", "")
            )
            if not instruction or not isinstance(instruction, str):
                raise ValueError(
                    f"Operation '{operation}' at index {index} requires an 'instruction' or 'new_background' field"
                )
            result["instruction"] = instruction.strip()

        return result
