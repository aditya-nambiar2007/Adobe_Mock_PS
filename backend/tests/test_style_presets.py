from __future__ import annotations

from app.services.style_presets import (
    build_style_instruction,
    build_style_params,
    detect_style_preset,
)


def test_detect_style_preset_from_prompt() -> None:
    assert detect_style_preset("make this ghibli style") == "ghibli"


def test_detect_style_preset_from_options() -> None:
    assert detect_style_preset("make it soft and warm", {"style_preset": "ghibli"}) == "ghibli"


def test_build_style_instruction_and_params() -> None:
    instruction = build_style_instruction("make the scene warmer", "ghibli")
    params = build_style_params("ghibli")

    assert "Ghibli-inspired" in instruction
    assert "make the scene warmer" in instruction
    assert params["guidance_scale"] > 0
    assert params["steps"] >= 1
