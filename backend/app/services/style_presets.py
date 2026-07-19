from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StylePreset:
    name: str
    prompt_prefix: str
    negative_prompt: str
    guidance_scale: float
    image_guidance_scale: float
    strength: float
    steps: int
    lora_env_var: str = ""
    adapter_name: str = "default"


GHIBLI_PRESET = StylePreset(
    name="ghibli",
    prompt_prefix=(
        "Ghibli-inspired hand-painted animation still, soft watercolor textures, "
        "lush natural backgrounds, warm sunlight, gentle linework, expressive faces, "
        "cinematic composition, subtle film grain, preserve the original subject and layout"
    ),
    negative_prompt=(
        "photorealistic, harsh contrast, gritty texture, distorted faces, extra limbs, "
        "blurry, low detail, plastic skin, noisy artifacts"
    ),
    guidance_scale=6.8,
    image_guidance_scale=1.7,
    strength=0.62,
    steps=36,
    lora_env_var="GHIBLI_LORA_WEIGHTS",
    adapter_name="ghibli",
)

STYLE_PRESETS: Dict[str, StylePreset] = {
    "ghibli": GHIBLI_PRESET,
}

STYLE_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "ghibli": (
        "ghibli",
        "studio ghibli",
        "ghibli-style",
        "ghibli style",
        "miyazaki",
        "spirited away",
        "totoro",
        "howl's moving castle",
        "howl movie",
    ),
}


def detect_style_preset(prompt: str, options: Optional[Dict[str, Any]] = None) -> Optional[str]:
    options = options or {}
    explicit = (
        options.get("style_preset")
        or options.get("style")
        or options.get("art_style")
    )
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in STYLE_PRESETS:
            return normalized

    prompt_l = prompt.lower()
    for preset_name, keywords in STYLE_KEYWORDS.items():
        if any(keyword in prompt_l for keyword in keywords):
            return preset_name

    return None


def get_style_preset(preset_name: Optional[str]) -> Optional[StylePreset]:
    if not preset_name:
        return None
    return STYLE_PRESETS.get(preset_name)


def build_style_instruction(prompt: str, preset_name: Optional[str]) -> str:
    preset = get_style_preset(preset_name)
    if preset is None:
        return prompt
    return f"{preset.prompt_prefix}. User request: {prompt}"


def build_style_params(preset_name: Optional[str]) -> Dict[str, Any]:
    preset = get_style_preset(preset_name)
    if preset is None:
        return {}

    return {
        "guidance_scale": preset.guidance_scale,
        "image_guidance_scale": preset.image_guidance_scale,
        "strength": preset.strength,
        "steps": preset.steps,
        "negative_prompt": preset.negative_prompt,
    }


def style_preset_from_params(params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    params = params or {}
    preset = params.get("style_preset")
    if isinstance(preset, str) and preset.strip():
        normalized = preset.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in STYLE_PRESETS:
            return normalized
    return None
