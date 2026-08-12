"""博查（Bocha）Web 搜索服务。

仅在以下场景作为最低优先级参考段注入 prompt：
- 术语库 0 命中
- RAG dense top1 < 阈值
- RAG sparse 命中数 = 0

``search`` 保留翻译流水线的静默降级语义；``search_strict`` 为 API 调用方
提供可区分的失败诊断。两个入口共用缓存、in-flight 去重和 Bocha 解析。
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


class WebSearchFailure(RuntimeError):
    code = "web_search_upstream_error"


class WebSearchNotConfigured(WebSearchFailure):
    code = "web_search_not_configured"


class WebSearchTimeout(WebSearchFailure):
    code = "web_search_timeout"


class WebSearchRateLimited(WebSearchFailure):
    code = "web_search_rate_limited"


class WebSearchUpstreamError(WebSearchFailure):
    code = "web_search_upstream_error"


class WebSearchInvalidResponse(WebSearchFailure):
    code = "web_search_invalid_response"


def _cache_key(
    query: str,
    *,
    prefix: str | None = None,
    provider: str = "bocha",
) -> str:
    """缓存键显式隔离 provider / prefix / query，并归一化空白和大小写。"""
    norm_query = " ".join(query.strip().split()).lower()
    norm_prefix = " ".join((prefix or "").strip().split()).lower()
    norm_provider = provider.strip().lower()
    raw = f"{norm_provider}\n{norm_prefix}\n{norm_query}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


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
        self.max_concurrent = max(1, settings.web_search_max_concurrent)
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
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        if not self.enabled:
            if settings.web_search_enabled and not settings.bocha_api_key:
                logger.warning("WEB_SEARCH_ENABLED=true 但 BOCHA_API_KEY 为空，web search 已禁用")
            else:
                logger.info("web search 未启用（WEB_SEARCH_ENABLED=false 或 BOCHA_API_KEY 为空）")

    # ---------- 对外入口 ----------

    async def search(self, query: str, *, prefix: str | None = None) -> list[WebSearchResult]:
        """翻译流水线兼容入口：所有搜索失败都静默降级为空列表。"""
        try:
            return await self.search_strict(query, prefix=prefix)
        except WebSearchNotConfigured:
            return []
        except WebSearchFailure as exc:
            logger.warning("web search 静默降级 code=%s", exc.code)
            return []

    async def search_strict(
        self,
        query: str,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        provider: str = "bocha",
    ) -> list[WebSearchResult]:
        """API 入口：网络和响应失败使用可区分的异常类型。"""
        if not self.enabled:
            raise WebSearchNotConfigured("web search is not configured")
        if not query or not query.strip():
            return []

        query_combined = f"{prefix.strip()} {query}".strip() if prefix and prefix.strip() else query
        key = _cache_key(query, prefix=prefix, provider=provider)
        result_limit = self.max_snippets if limit is None else min(limit, self.max_snippets)

        cached = self._read_cache(key)
        if cached is not None:
            logger.debug("web_search cache hit key=%s", key)
            return cached[:result_limit]

        # in-flight dedupe：同一 query 进程内并发只发 1 次
        existing = self._inflight.get(key)
        if existing is not None:
            return (await existing)[:result_limit]

        task = asyncio.create_task(
            self._do_search_strict(key, query_combined=query_combined, cache_query=query)
        )
        self._inflight[key] = task
        try:
            return (await task)[:result_limit]
        finally:
            self._inflight.pop(key, None)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    # ---------- 实际执行 ----------

    async def _do_search_strict(
        self,
        key: str,
        *,
        query_combined: str,
        cache_query: str,
    ) -> list[WebSearchResult]:
        assert self._client is not None  # enabled=True 时 client 必有
        payload = {"query": query_combined, "summary": self.summary, "count": self.count}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempts = max(1, self.max_retries + 1)
        last_status: int | None = None
        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    resp = await self._client.post(self.endpoint, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                logger.warning("bocha 请求失败 code=web_search_timeout")
                raise WebSearchTimeout("web search timed out") from exc
            except httpx.RequestError as exc:
                logger.warning("bocha 请求失败 code=web_search_upstream_error")
                raise WebSearchUpstreamError("web search transport failed") from exc

            status = resp.status_code
            last_status = status
            if status == 200:
                try:
                    body = resp.json()
                except ValueError as exc:
                    logger.warning("bocha 响应失败 code=web_search_invalid_response")
                    raise WebSearchInvalidResponse("web search returned invalid JSON") from exc
                results = self._parse_response_strict(body)
                if self.cache_enabled:
                    self._write_cache(key, results, query=cache_query)
                logger.info("web_search cache miss key=%s n=%d", key, len(results))
                return results

            # 4xx（非 429）→ 进程内禁用，避免无效轮询
            if 400 <= status < 500 and status != 429:
                logger.error(
                    "bocha 请求失败 status=%d code=web_search_upstream_error；"
                    "进程内禁用 web search",
                    status,
                )
                self.enabled = False
                raise WebSearchUpstreamError("web search upstream rejected the request")

            # 429 / 5xx → 线性退避重试
            if (status == 429 or 500 <= status < 600) and attempt < attempts:
                backoff = 1.0 * attempt
                logger.warning(
                    "bocha 请求失败 status=%d attempt=%d/%d，退避 %.1fs",
                    status,
                    attempt,
                    attempts,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            if status == 429:
                raise WebSearchRateLimited("web search rate limited")
            raise WebSearchUpstreamError("web search upstream failed")

        if last_status == 429:
            raise WebSearchRateLimited("web search rate limited")
        raise WebSearchUpstreamError("web search upstream failed")

    # ---------- 响应解析 ----------

    def _parse_response_strict(self, body: object) -> list[WebSearchResult]:
        if not isinstance(body, dict):
            raise WebSearchInvalidResponse("web search response must be an object")
        code = body.get("code")
        if not isinstance(code, int):
            raise WebSearchInvalidResponse("web search response code is invalid")
        if code != 200:
            raise WebSearchUpstreamError("web search upstream returned an error")

        data = body.get("data")
        if not isinstance(data, dict):
            raise WebSearchInvalidResponse("web search response data is invalid")
        web_pages = data.get("webPages")
        if not isinstance(web_pages, dict) or not isinstance(web_pages.get("value"), list):
            raise WebSearchInvalidResponse("web search response pages are invalid")
        pages = web_pages["value"]

        image_section = data.get("images")
        imgs: list[object] = []
        if image_section is not None:
            if not isinstance(image_section, dict) or not isinstance(image_section.get("value"), list):
                raise WebSearchInvalidResponse("web search response images are invalid")
            imgs = image_section["value"]

        top_image: str | None = None
        if imgs:
            first_img = imgs[0]
            if not isinstance(first_img, dict):
                raise WebSearchInvalidResponse("web search image item is invalid")
            image_value = first_img.get("contentUrl") or first_img.get("thumbnailUrl")
            if image_value is not None and not isinstance(image_value, str):
                raise WebSearchInvalidResponse("web search image URL is invalid")
            top_image = image_value

        results: list[WebSearchResult] = []
        for i, p in enumerate(pages[: self.max_snippets]):
            if not isinstance(p, dict):
                raise WebSearchInvalidResponse("web search result item is invalid")
            for field in ("name", "url", "summary", "snippet", "siteName"):
                value = p.get(field)
                if value is not None and not isinstance(value, str):
                    raise WebSearchInvalidResponse(f"web search result {field} is invalid")
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
        except (OSError, json.JSONDecodeError, ValidationError, TypeError):
            logger.warning("web search 缓存无效 key=%s（视为 miss）", key)
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


__all__ = [
    "WebSearchFailure",
    "WebSearchInvalidResponse",
    "WebSearchNotConfigured",
    "WebSearchRateLimited",
    "WebSearchService",
    "WebSearchTimeout",
    "WebSearchUpstreamError",
]
