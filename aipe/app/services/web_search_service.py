"""博查（Bocha）Web 搜索服务：RAG 弱召回兜底用的外部上下文补充。

仅在以下场景作为最低优先级参考段注入 prompt：
- 术语库 0 命中
- RAG dense top1 < 阈值
- RAG sparse 命中数 = 0

设计要点：
- 异步 httpx 客户端；429/5xx 线性退避重试；网络错误 / 4xx 直接静默降级返回空
- 本地文件缓存：``data/cache/web_search/{md5(query)}.json``，``tmp.replace`` 原子写
- 进程内 in-flight 去重（同 query 并发只发 1 次 HTTP）
- 4xx（如 401 失效密钥）后进程内禁用，避免雪崩

失败模式：所有失败 (网络 / 4xx / 5xx / 解析) 都返回 ``[]``，由 pipeline 决定是否
继续翻译；本服务永远不抛异常给上层。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.schemas.web_search import WebSearchResult

logger = logging.getLogger(__name__)


_CACHE_VERSION = 1


def _cache_key(query: str) -> str:
    """归一化空白 + 大小写后取 md5；保证 ``  少侠 ``、``少侠`` 命中同一缓存。"""
    norm = " ".join(query.strip().split()).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WebSearchService:
    """博查 Web 搜索服务（单例）。配置变更需重启进程。"""

    def __init__(self, settings: Settings):
        self.endpoint = settings.bocha_endpoint
        self.api_key = settings.bocha_api_key
        self.count = settings.bocha_count
        self.summary = settings.bocha_summary
        self.timeout = settings.bocha_timeout
        self.max_retries = settings.bocha_max_retries
        self.max_snippets = settings.web_search_max_snippets
        self.snippet_max_chars = settings.web_search_snippet_max_chars
        self.cache_enabled = settings.web_search_cache_enabled
        self.cache_dir = Path(settings.web_search_cache_dir)

        # enabled = 全局开关 ∧ key 非空 —— 任一不满足都不会发任何 HTTP
        self.enabled: bool = bool(settings.web_search_enabled and settings.bocha_api_key)

        if self.enabled and self.cache_enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("web search 缓存目录创建失败 %s: %s（关闭缓存继续）", self.cache_dir, exc)
                self.cache_enabled = False

        self._client: httpx.AsyncClient | None = (
            httpx.AsyncClient(timeout=self.timeout) if self.enabled else None
        )
        self._inflight: dict[str, asyncio.Task[list[WebSearchResult]]] = {}

        if not self.enabled:
            if settings.web_search_enabled and not settings.bocha_api_key:
                logger.warning("WEB_SEARCH_ENABLED=true 但 BOCHA_API_KEY 为空，web search 已禁用")
            else:
                logger.info("web search 未启用（WEB_SEARCH_ENABLED=false 或 BOCHA_API_KEY 为空）")

    # ---------- 对外入口 ----------

    async def search(self, query: str, *, prefix: str | None = None) -> list[WebSearchResult]:
        """主入口：缓存命中直接返回；否则发实网；同 query 并发去重。

        永远不抛异常（任何失败都返回空列表）。
        """
        if not self.enabled or not query or not query.strip():
            return []

        query_combined = f"{prefix.strip()} {query}".strip() if prefix and prefix.strip() else query
        key = _cache_key(query_combined)

        cached = self._read_cache(key)
        if cached is not None:
            logger.debug("web_search cache hit key=%s query=%r", key, query[:30])
            return cached

        # in-flight dedupe：同一 query 进程内并发只发 1 次
        existing = self._inflight.get(key)
        if existing is not None:
            return await existing

        task = asyncio.create_task(self._do_search(query, key, query_combined=query_combined))
        self._inflight[key] = task
        try:
            return await task
        finally:
            self._inflight.pop(key, None)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    # ---------- 实际执行 ----------

    async def _do_search(self, query: str, key: str, *, query_combined: str | None = None) -> list[WebSearchResult]:
        assert self._client is not None  # enabled=True 时 client 必有
        query_combined = query_combined or query
        payload = {"query": query_combined, "summary": self.summary, "count": self.count}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempts = max(1, self.max_retries + 1)
        last_status: int | None = None
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = await self._client.post(self.endpoint, json=payload, headers=headers)
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning("bocha 请求异常 query=%r attempt=%d/%d: %s", query[:30], attempt, attempts, exc)
                break  # 网络错误不重试，直接降级

            status = resp.status_code
            last_status = status
            if status == 200:
                try:
                    body = resp.json()
                except ValueError as exc:
                    logger.warning("bocha 响应非 JSON query=%r: %s", query[:30], exc)
                    return []
                results = self._parse_response(body)
                if self.cache_enabled:
                    self._write_cache(key, results, query=query)
                logger.info(
                    "web_search cache miss key=%s query=%r n=%d", key, query[:30], len(results)
                )
                return results

            # 4xx（非 429）→ 进程内禁用，避免无效轮询
            if 400 <= status < 500 and status != 429:
                logger.error(
                    "bocha 4xx query=%r status=%d body=%s；进程内禁用 web search",
                    query[:30],
                    status,
                    resp.text[:200],
                )
                self.enabled = False
                return []

            # 429 / 5xx → 线性退避重试
            if attempt < attempts:
                backoff = 1.0 * attempt
                logger.warning(
                    "bocha %d query=%r attempt=%d/%d, 退避 %.1fs",
                    status,
                    query[:30],
                    attempt,
                    attempts,
                    backoff,
                )
                await asyncio.sleep(backoff)

        logger.warning(
            "bocha 最终失败 query=%r last_status=%s last_exc=%s",
            query[:30],
            last_status,
            last_exc,
        )
        return []

    # ---------- 响应解析 ----------

    def _parse_response(self, body: dict) -> list[WebSearchResult]:
        data = body.get("data") or {}
        pages = ((data.get("webPages") or {}).get("value")) or []
        imgs = ((data.get("images") or {}).get("value")) or []
        top_image: str | None = None
        if imgs:
            first_img = imgs[0] or {}
            top_image = first_img.get("contentUrl") or first_img.get("thumbnailUrl")

        results: list[WebSearchResult] = []
        for i, p in enumerate(pages[: self.max_snippets]):
            raw_snippet = (p.get("summary") or p.get("snippet") or "").strip()
            snippet = raw_snippet[: self.snippet_max_chars]
            results.append(
                WebSearchResult(
                    title=(p.get("name") or "").strip(),
                    snippet=snippet,
                    url=(p.get("url") or "").strip(),
                    site_name=(p.get("siteName") or None),
                    image_url=top_image if i == 0 else None,
                )
            )
        return results

    # ---------- 缓存 ----------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> list[WebSearchResult] | None:
        if not self.cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            payload = raw.get("results") or []
            return [WebSearchResult(**r) for r in payload]
        except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning("web search 缓存损坏 %s: %s（视为 miss）", path, exc)
            return None

    def _write_cache(self, key: str, results: list[WebSearchResult], *, query: str) -> None:
        if not self.cache_enabled:
            return
        path = self._cache_path(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        body = {
            "version": _CACHE_VERSION,
            "cached_at": _now_iso(),
            "query": query,
            "results": [r.model_dump() for r in results],
        }
        try:
            tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("web search 写缓存失败 %s: %s（忽略）", path, exc)


__all__ = ["WebSearchService"]
