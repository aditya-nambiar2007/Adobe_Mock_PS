from __future__ import annotations

import asyncio

from app.services.planner import Planner


def test_offline_planner_prefers_style_transfer_for_ghibli_prompts() -> None:
    planner = Planner(gemini_service=None)
    plan = asyncio.run(planner.create_plan("make this image ghibli style"))

    assert len(plan) == 1
    assert plan[0]["operation"] == "change_style"
    assert "Ghibli-inspired" in plan[0]["instruction"]
