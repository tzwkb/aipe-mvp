"""LLM / Embedding 接口封装。

均走 OpenAI 兼容协议（chat.completions / embeddings），通过 ``base_url`` 接入
任意第三方网关或自建服务（如 Qwen3.6-plus、DashScope-兼容端点等）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, APIError, APIStatusError, APITimeoutError, RateLimitError

from app.config import Settings
from app.errors import TranslationError

logger = logging.getLogger(__name__)


Message = dict[str, str]


class LLMService:
    """OpenAI 兼容协议的 LLM + Embedding 客户端。

    - ``translate``：调用 ``chat.completions``，内置指数退避重试与降级标记
    - ``embed`` / ``embed_batch``：调用 ``embeddings``，单条 / 批量
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model
        self.embedding_model = settings.embedding_model

        self._chat_client = AsyncOpenAI(
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=0,  # 退避策略由本服务自管，避免 SDK 静默重试
        )

        # Embedding 服务可能与 LLM 走不同网关；未配置时复用 LLM 的网关。
        self._embed_client = AsyncOpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key or "EMPTY",
            base_url=settings.embedding_base_url or settings.llm_base_url,
            timeout=settings.embedding_timeout,
            max_retries=0,
        )

    # ---------- chat / translate ----------

    async def translate(
        self,
        prompt: str | list[Message],
        *,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> str:
        """调用 chat.completions 生成译文。

        ``prompt`` 支持字符串（自动包装为 user 消息）或完整 messages 列表。
        失败时按 5.5.2 节策略：429 指数退避（1→32s，最多 N 次），5xx 等待 2s 重试，
        最终连续失败抛出 ``TranslationError``，上层负责标记 ``[ERROR: AI_FAIL]``。
        """
        messages = self._coerce_messages(prompt)
        retries = max_retries if max_retries is not None else self.settings.llm_max_retries

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = await self._chat_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=(
                        self.settings.llm_temperature if temperature is None else temperature
                    ),
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except RateLimitError as exc:
                # 部分网关用 429 包装参数错误（如 InvalidParameter），重试无意义
                if "InvalidParameter" in str(exc):
                    raise TranslationError(f"[ERROR: AI_FAIL] LLM 参数错误: {exc}") from exc
                last_exc = exc
                wait = min(2 ** attempt, 32)
                logger.warning("LLM rate limited (attempt %s/%s), sleep %ss", attempt + 1, retries, wait)
                await asyncio.sleep(wait)
            except APITimeoutError as exc:
                last_exc = exc
                logger.warning("LLM timeout (attempt %s/%s)", attempt + 1, retries)
                await asyncio.sleep(2)
            except APIStatusError as exc:
                last_exc = exc
                if 500 <= exc.status_code < 600:
                    logger.warning(
                        "LLM 5xx (%s) (attempt %s/%s)", exc.status_code, attempt + 1, retries
                    )
                    await asyncio.sleep(2)
                else:
                    # 4xx（除 429）通常是请求/鉴权问题，重试无意义。
                    raise TranslationError(f"[ERROR: AI_FAIL] LLM {exc.status_code}: {exc}") from exc
            except APIError as exc:
                last_exc = exc
                logger.warning("LLM APIError (attempt %s/%s): %s", attempt + 1, retries, exc)
                await asyncio.sleep(2)

        raise TranslationError(f"[ERROR: AI_FAIL] LLM 重试 {retries} 次仍失败: {last_exc}")

    @staticmethod
    def _coerce_messages(prompt: str | list[Message]) -> list[Message]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    # ---------- classify ----------

    async def classify(self, messages: list[Message]) -> str:
        """文本类型分类调用：低温度、短输出、单次尝试（分类是尽力而为）。"""
        try:
            resp = await self._chat_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=20,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("文本分类 LLM 调用失败: %s", exc)
            return ""

    # ---------- embedding ----------

    async def embed(self, text: str) -> list[float]:
        """单条文本向量化。"""
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，保持输入顺序与输出对齐。

        内置指数退避重试：timeout / rate-limit / 5xx / APIError 均会重试，
        与 ``translate`` 保持一致。
        """
        if not texts:
            return []

        retries = self.settings.llm_max_retries
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                extra = {}
                if self.settings.embedding_dimensions is not None:
                    extra["dimensions"] = self.settings.embedding_dimensions
                resp = await self._embed_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                    **extra,
                )
                # 网关可能不保证返回顺序，按 ``index`` 字段重新排序。
                ordered = sorted(resp.data, key=lambda d: d.index)
                vecs = [list(d.embedding) for d in ordered]
                # 部分第三方网关忽略 dimensions 参数，客户端侧补充截断
                dim = self.settings.embedding_dimensions
                if dim is not None and vecs and len(vecs[0]) > dim:
                    vecs = [v[:dim] for v in vecs]
                return vecs
            except RateLimitError as exc:
                if "InvalidParameter" in str(exc):
                    raise TranslationError(f"[ERROR: AI_FAIL] Embedding 参数错误: {exc}") from exc
                last_exc = exc
                wait = min(2 ** attempt, 32)
                logger.warning("Embedding rate limited (attempt %s/%s), sleep %ss", attempt + 1, retries, wait)
                await asyncio.sleep(wait)
            except APITimeoutError as exc:
                last_exc = exc
                logger.warning("Embedding timeout (attempt %s/%s)", attempt + 1, retries)
                await asyncio.sleep(2)
            except APIStatusError as exc:
                last_exc = exc
                if 500 <= exc.status_code < 600:
                    logger.warning("Embedding 5xx (%s) (attempt %s/%s)", exc.status_code, attempt + 1, retries)
                    await asyncio.sleep(2)
                else:
                    raise TranslationError(f"[ERROR: AI_FAIL] Embedding {exc.status_code}: {exc}") from exc
            except APIError as exc:
                last_exc = exc
                logger.warning("Embedding APIError (attempt %s/%s): %s", attempt + 1, retries, exc)
                await asyncio.sleep(2)

        raise TranslationError(f"[ERROR: AI_FAIL] Embedding 重试 {retries} 次仍失败: {last_exc}")

    # ---------- lifecycle ----------

    async def aclose(self) -> None:
        await self._chat_client.close()
        if self._embed_client is not self._chat_client:
            await self._embed_client.close()

    def describe(self) -> dict[str, Any]:
        return {
            "llm_model": self.model,
            "llm_base_url": str(self._chat_client.base_url) if self._chat_client.base_url else None,
            "embedding_model": self.embedding_model,
            "embedding_base_url": (
                str(self._embed_client.base_url) if self._embed_client.base_url else None
            ),
        }
