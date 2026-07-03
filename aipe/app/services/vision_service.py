"""图片视觉分析服务：用多模态 LLM 分析 Web 搜索配图，提取对翻译有用的信息。

调用 qwen3.6-plus（与主 LLM 共用同一网关 base_url / api_key）。
失败模式：所有异常静默降级返回 None，上层永远不会收到异常。
内置按图片 URL + system prompt 的进程内内存缓存。
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是《燕云十六声》游戏本地化团队的视觉分析助手，专门从图片中提取对汉译英工作有帮助的信息。"
)

_USER_PROMPT = (
    "这张图片来自围绕游戏内容的网络搜索结果。请仔细分析图片，按以下顺序提取翻译参考信息：\n\n"
    "1. **游戏专有名词**：人物名、地名、门派名、功法名、道具名、称号等（直接列出原文）\n"
    "2. **可见文字**：界面文本、技能名、任务名、物品描述、对话文字等\n"
    "3. **视觉风格与场景**：画面风格（如：水墨、写实、Q版）、场景类型（如：江湖、皇宫、战场）、氛围（如：庄重、诙谐）\n\n"
    "要求：用简洁的中文输出，每类信息一行。专有名词尽量直接引用，不要解释。"
    "若图片与游戏内容明显无关或无法辨认，请一句话说明。"
)


class VisionService:
    """多模态图片分析服务（单例）。"""

    def __init__(self, settings: Settings):
        self.model = settings.vision_model
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url,
            timeout=30.0,
            max_retries=0,
        )
        self._cache: dict[tuple[str, str], str] = {}

    async def analyze_image(self, image_url: str, *, system_prompt: str | None = None) -> str | None:
        """分析图片，返回对翻译有用的文字描述。失败或图片 URL 为空时返回 None。"""
        if not image_url or not image_url.strip():
            return None

        cache_key = (image_url, system_prompt or _SYSTEM_PROMPT)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("vision cache hit url=%r", image_url)
            return cached

        logger.info("vision analyze url=%r model=%s", image_url, self.model)
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=500,
            )
            result = (resp.choices[0].message.content or "").strip() or None
            if result:
                self._cache[cache_key] = result
                logger.info("vision done url=%r chars=%d", image_url, len(result))
            return result
        except Exception as exc:
            logger.warning("vision 分析失败 url=%r: %s", image_url, exc)
            return None

    async def aclose(self) -> None:
        await self._client.close()


__all__ = ["VisionService"]
