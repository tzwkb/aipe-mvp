"""WebSearchService 单元测试。

不依赖真 Bocha API：用 httpx.MockTransport 拦截所有 HTTP 请求并返回预设响应。
"""

from __future__ import annotations

import asyncio
import json

import httpx

from app.config import Settings
from app.services.web_search_service import WebSearchService, _cache_key


# ---------- 帮助函数 ----------


def _bocha_response_body() -> dict:
    """伪造一个最小合法的 Bocha API 响应。"""
    return {
        "code": 200,
        "log_id": "abc",
        "msg": None,
        "data": {
            "_type": "SearchResponse",
            "queryContext": {"originalQuery": "test"},
            "webPages": {
                "webSearchUrl": "",
                "totalEstimatedMatches": 1,
                "value": [
                    {
                        "id": None,
                        "name": "标题 1",
                        "url": "https://a.example.com/1",
                        "displayUrl": "https://a.example.com/1",
                        "snippet": "短摘要 1",
                        "summary": "长摘要 1 " * 30,  # 触发截断
                        "siteName": "示例站 A",
                        "siteIcon": None,
                        "dateLastCrawled": "2026-05-01T00:00:00Z",
                        "cachedPageUrl": None,
                        "language": None,
                        "isFamilyFriendly": None,
                        "isNavigational": None,
                    },
                    {
                        "id": None,
                        "name": "标题 2",
                        "url": "https://b.example.com/2",
                        "displayUrl": "https://b.example.com/2",
                        "snippet": "短摘要 2",
                        "siteName": "示例站 B",
                        "dateLastCrawled": "2026-05-02T00:00:00Z",
                    },
                ],
                "someResultsRemoved": False,
            },
            "images": {
                "value": [
                    {
                        "thumbnailUrl": "https://img.example.com/thumb1.jpg",
                        "contentUrl": "https://img.example.com/1.jpg",
                        "hostPageUrl": "https://a.example.com/1",
                        "width": 600,
                        "height": 400,
                    },
                    {
                        "thumbnailUrl": "https://img.example.com/thumb2.jpg",
                        "contentUrl": "https://img.example.com/2.jpg",
                        "hostPageUrl": "https://b.example.com/2",
                    },
                ],
            },
            "videos": None,
        },
    }


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        web_search_enabled=True,
        bocha_api_key="sk-test",
        bocha_endpoint="https://mock.bocha/v1/web-search",
        bocha_count=8,
        bocha_summary=True,
        bocha_timeout=2.0,
        bocha_max_retries=1,
        web_search_max_snippets=3,
        web_search_snippet_max_chars=50,
        web_search_cache_dir=str(tmp_path / "cache"),
        web_search_cache_enabled=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_service(settings: Settings, transport: httpx.MockTransport) -> WebSearchService:
    svc = WebSearchService(settings)
    # 替换内部 httpx 客户端为 mock 版本，保留同样的 timeout
    svc._client = httpx.AsyncClient(transport=transport, timeout=settings.bocha_timeout)
    return svc


# ---------- 基础场景 ----------


def test_disabled_when_api_key_missing(tmp_path):
    settings = _settings(tmp_path, bocha_api_key="")
    svc = WebSearchService(settings)
    assert svc.enabled is False
    # 直接调用也安全
    res = asyncio.run(svc.search("少侠"))
    assert res == []


def test_disabled_when_global_off(tmp_path):
    settings = _settings(tmp_path, web_search_enabled=False)
    svc = WebSearchService(settings)
    assert svc.enabled is False
    res = asyncio.run(svc.search("少侠"))
    assert res == []


def test_search_parses_bocha_response_and_truncates_snippet(tmp_path):
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        body = _bocha_response_body()
        return httpx.Response(200, json=body)

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    res = asyncio.run(svc.search("凌霄破"))
    assert len(res) == 2  # max_snippets=3 但只有 2 条
    assert res[0].title == "标题 1"
    assert res[0].site_name == "示例站 A"
    # 长摘要被截断到 50 字符
    assert len(res[0].snippet) == settings.web_search_snippet_max_chars
    # 图片只挂在第 0 条
    assert res[0].image_url == "https://img.example.com/1.jpg"
    assert res[1].image_url is None
    assert counter["n"] == 1


def test_cache_hit_skips_http(tmp_path):
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    asyncio.run(svc.search("凌霄破"))
    assert counter["n"] == 1
    # 第二次同 query → 缓存命中
    asyncio.run(svc.search("凌霄破"))
    assert counter["n"] == 1


def test_cache_normalization_hits_same_key(tmp_path):
    """前后空白 / 大小写差异不影响缓存命中。"""
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))
    asyncio.run(svc.search("  Lingxiao  "))
    asyncio.run(svc.search("lingxiao"))
    assert counter["n"] == 1


def test_corrupt_cache_falls_through(tmp_path):
    settings = _settings(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key("凌霄破")
    (cache_dir / f"{key}.json").write_text("{ broken", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_bocha_response_body())

    svc = _make_service(settings, httpx.MockTransport(handler))
    res = asyncio.run(svc.search("凌霄破"))
    assert len(res) == 2  # 应该走 HTTP 拿新结果而非崩溃


def test_4xx_disables_service_permanently(tmp_path):
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(401, json={"msg": "invalid key"})

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    res1 = asyncio.run(svc.search("foo"))
    assert res1 == []
    assert svc.enabled is False
    # 第二次直接短路，不会再发 HTTP
    res2 = asyncio.run(svc.search("bar"))
    assert res2 == []
    assert counter["n"] == 1


def test_429_retries_then_succeeds(tmp_path):
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, json={"msg": "rate limit"})
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path, bocha_max_retries=2)
    svc = _make_service(settings, httpx.MockTransport(handler))

    # 让 sleep 不真的睡，提高测试速度
    import app.services.web_search_service as ws_mod

    real_sleep = ws_mod.asyncio.sleep

    async def fast_sleep(_secs):
        await real_sleep(0)

    ws_mod.asyncio.sleep = fast_sleep  # type: ignore[assignment]
    try:
        res = asyncio.run(svc.search("foo"))
    finally:
        ws_mod.asyncio.sleep = real_sleep  # type: ignore[assignment]

    assert len(res) == 2
    assert state["n"] == 2
    assert svc.enabled is True


def test_network_error_silently_degrades(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conn refused")

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    res = asyncio.run(svc.search("foo"))
    assert res == []
    # 网络错误不应禁用 service（可能是临时网络抖动）
    assert svc.enabled is True


def test_malformed_response_returns_empty(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    res = asyncio.run(svc.search("foo"))
    assert res == []


def test_missing_webpages_field_returns_empty(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 200, "data": {}})

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    res = asyncio.run(svc.search("foo"))
    assert res == []


def test_inflight_dedup(tmp_path):
    """同 query 并发两次只发 1 次 HTTP。"""
    counter = {"n": 0}
    gate = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        # 第一次进来时等待，模拟慢请求；第二个并发请求应直接命中 in-flight
        await gate.wait()
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    async def runner():
        t1 = asyncio.create_task(svc.search("同一 query"))
        t2 = asyncio.create_task(svc.search("同一 query"))
        await asyncio.sleep(0.05)   # 让两个 task 都进到 _do_search 之前
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)
        return r1, r2

    r1, r2 = asyncio.run(runner())
    assert len(r1) == 2 and len(r2) == 2
    assert counter["n"] == 1   # 关键：HTTP 只发了一次


def test_cache_disabled_writes_no_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path, web_search_cache_enabled=False)
    svc = _make_service(settings, httpx.MockTransport(handler))
    asyncio.run(svc.search("foo"))
    cache_dir = tmp_path / "cache"
    # 缓存禁用时根本不应该建目录或写文件
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_cache_payload_format(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))
    asyncio.run(svc.search("凌霄破"))

    key = _cache_key("凌霄破")
    path = tmp_path / "cache" / f"{key}.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["query"] == "凌霄破"
    assert payload["results"] and payload["results"][0]["title"] == "标题 1"
    assert "cached_at" in payload


def test_empty_query_returns_empty(tmp_path):
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(200, json=_bocha_response_body())

    settings = _settings(tmp_path)
    svc = _make_service(settings, httpx.MockTransport(handler))

    res = asyncio.run(svc.search("   "))
    assert res == []
    assert counter["n"] == 0
