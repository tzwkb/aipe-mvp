from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.config import Settings
from app.services.vision_service import VisionService


def test_vision_cache_keeps_project_prompts_separate():
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            self.calls.append(prompt)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f"analysis for {prompt}")
                    )
                ]
            )

    completions = FakeCompletions()
    svc = VisionService(Settings())
    svc._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    first = asyncio.run(svc.analyze_image("https://example.test/img.png", system_prompt="project A"))
    second = asyncio.run(svc.analyze_image("https://example.test/img.png", system_prompt="project B"))

    assert first == "analysis for project A"
    assert second == "analysis for project B"
    assert completions.calls == ["project A", "project B"]
