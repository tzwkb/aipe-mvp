from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.rag import get_project_resource_manager as rag_projects_dependency
from app.api.rag import get_rag_service
from app.api.web_search import get_project_resource_manager as web_projects_dependency
from app.config import Settings, get_settings
from app.dependencies import get_web_search_service
from app.errors import TranslationError
from app.main import app
from app.schemas.rag import RAGSearchResult
from app.schemas.web_search import WebSearchResult
from app.services.project_service import ProjectProfileError
from app.services.web_search_service import (
    WebSearchInvalidResponse,
    WebSearchNotConfigured,
    WebSearchRateLimited,
    WebSearchService,
    WebSearchTimeout,
    WebSearchUpstreamError,
    _cache_key,
)


class FakeRAGService:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    async def search(self, query, *, threshold, top_k, collection):
        self.calls.append(
            {
                "query": query,
                "threshold": threshold,
                "top_k": top_k,
                "collection": collection,
            }
        )
        if self.error is not None:
            raise self.error
        return [
            RAGSearchResult(
                source="source",
                target="target",
                score=0.5,
                status="Done",
            )
        ]


class FakeProjects:
    def __init__(
        self,
        *,
        collection: str | None = "project_collection",
        prefix: str | None = "project-prefix",
        allow_web_search: bool = True,
        error: Exception | None = None,
    ):
        self.collection = collection
        self.prefix = prefix
        self.allow_web_search = allow_web_search
        self.error = error
        self.calls: list[str | None] = []

    def profile(self, project_id):
        self.calls.append(project_id)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            qdrant_collection=self.collection,
            web_search_prefix=self.prefix,
            allow_web_search=self.allow_web_search,
        )


class FakeWebSearchService:
    def __init__(
        self,
        *,
        results: list[WebSearchResult] | None = None,
        error: Exception | None = None,
    ):
        self.results = [] if results is None else results
        self.error = error
        self.calls: list[dict] = []

    async def search_strict(self, query, *, prefix=None, limit=None, provider="bocha"):
        self.calls.append(
            {"query": query, "prefix": prefix, "limit": limit, "provider": provider}
        )
        if self.error is not None:
            raise self.error
        return self.results[:limit]


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _settings(**overrides) -> Settings:
    values = {
        "rag_global_collection": "global_collection",
        "rag_special_collection": "special_collection",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _override_rag(
    rag: FakeRAGService,
    *,
    settings: Settings | None = None,
    projects: FakeProjects | None = None,
) -> None:
    app.dependency_overrides[get_rag_service] = lambda: rag
    app.dependency_overrides[get_settings] = lambda: settings or _settings()
    app.dependency_overrides[rag_projects_dependency] = lambda: projects or FakeProjects()


def _override_web(web: FakeWebSearchService, projects: FakeProjects | None = None) -> None:
    app.dependency_overrides[get_web_search_service] = lambda: web
    app.dependency_overrides[web_projects_dependency] = lambda: projects or FakeProjects()


def test_rag_legacy_collection_remains_compatible_and_is_not_returned():
    rag = FakeRAGService()
    _override_rag(rag)

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "legacy", "collection": "legacy_collection"},
    )

    assert response.status_code == 200
    assert rag.calls[0]["collection"] == "legacy_collection"
    assert response.json()["scope"] is None
    assert response.json()["project_id"] is None
    assert "collection" not in response.json()


@pytest.mark.parametrize(
    ("scope", "expected_collection"),
    [("global", "global_collection"), ("special", "special_collection")],
)
def test_rag_global_and_special_use_server_settings(scope, expected_collection):
    rag = FakeRAGService()
    _override_rag(rag)

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "scoped", "scope": scope},
    )

    assert response.status_code == 200
    assert rag.calls[0]["collection"] == expected_collection
    assert response.json()["scope"] == scope
    assert "collection" not in response.json()


def test_rag_project_uses_profile_collection():
    rag = FakeRAGService()
    projects = FakeProjects(collection="profile_collection")
    _override_rag(rag, projects=projects)

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "project", "scope": "project", "project_id": "game/zh-en"},
    )

    assert response.status_code == 200
    assert projects.calls == ["game/zh-en"]
    assert rag.calls[0]["collection"] == "profile_collection"
    assert response.json()["project_id"] == "game/zh-en"
    assert "collection" not in response.json()


@pytest.mark.parametrize("scope", ["global", "special"])
def test_rag_scoped_collection_missing_fails_closed(scope):
    rag = FakeRAGService()
    settings = _settings(
        rag_global_collection=None if scope == "global" else "global_collection",
        rag_special_collection=None if scope == "special" else "special_collection",
    )
    _override_rag(rag, settings=settings)

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "scoped", "scope": scope},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_scope_not_configured"
    assert rag.calls == []


def test_rag_project_missing_collection_fails_closed():
    rag = FakeRAGService()
    _override_rag(rag, projects=FakeProjects(collection=None))

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "project", "scope": "project", "project_id": "game/zh-en"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "rag_project_invalid"
    assert rag.calls == []


def test_rag_unknown_project_is_sanitized():
    rag = FakeRAGService()
    projects = FakeProjects(error=ProjectProfileError("project profile not found: /secret/path"))
    _override_rag(rag, projects=projects)

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "project", "scope": "project", "project_id": "missing"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "rag_project_not_found"
    assert "/secret/path" not in response.text
    assert rag.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "bad", "scope": "global", "collection": "escape"},
        {"query": "bad", "scope": "project"},
        {"query": "bad", "scope": "special", "project_id": "escape"},
    ],
)
def test_rag_rejects_ambiguous_scope_inputs(payload):
    rag = FakeRAGService()
    _override_rag(rag)

    response = TestClient(app).post("/api/v1/rag/search", json=payload)

    assert response.status_code == 422
    assert rag.calls == []


def test_rag_search_error_has_stable_safe_detail():
    rag = FakeRAGService(TranslationError("collection secret_internal does not exist"))
    _override_rag(rag)

    response = TestClient(app).post(
        "/api/v1/rag/search",
        json={"query": "scoped", "scope": "global"},
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "rag_upstream_error"
    assert "secret_internal" not in response.text


def test_web_search_uses_project_policy_prefix_and_top_k():
    results = [
        WebSearchResult(title="one", snippet="first", url="https://example.com/1"),
        WebSearchResult(title="two", snippet="second", url="https://example.com/2"),
    ]
    web = FakeWebSearchService(results=results)
    projects = FakeProjects(prefix="approved-prefix")
    _override_web(web, projects)

    response = TestClient(app).post(
        "/api/v1/web-search",
        json={"query": "query", "top_k": 1, "project_id": "game/zh-en", "provider": "auto"},
    )

    assert response.status_code == 200
    assert web.calls == [
        {
            "query": "query",
            "prefix": "approved-prefix",
            "limit": 1,
            "provider": "bocha",
        }
    ]
    assert response.json()["provider"] == "bocha"
    assert response.json()["project_id"] == "game/zh-en"
    assert response.json()["total"] == 1


def test_web_search_successful_empty_result_is_200():
    web = FakeWebSearchService(results=[])
    _override_web(web)

    response = TestClient(app).post(
        "/api/v1/web-search",
        json={"query": "no result", "provider": "bocha"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["results"] == []


def test_web_search_project_policy_can_forbid_search():
    web = FakeWebSearchService()
    _override_web(web, FakeProjects(allow_web_search=False))

    response = TestClient(app).post(
        "/api/v1/web-search",
        json={"query": "blocked", "project_id": "confidential/zh-en"},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "web_search_forbidden"
    assert web.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (WebSearchNotConfigured("secret"), 503, "web_search_not_configured"),
        (WebSearchTimeout("secret"), 504, "web_search_timeout"),
        (WebSearchRateLimited("secret"), 503, "web_search_rate_limited"),
        (WebSearchUpstreamError("secret"), 502, "web_search_upstream_error"),
        (WebSearchInvalidResponse("secret"), 502, "web_search_invalid_response"),
    ],
)
def test_web_search_errors_have_stable_safe_details(error, expected_status, expected_code):
    web = FakeWebSearchService(error=error)
    _override_web(web)

    response = TestClient(app).post(
        "/api/v1/web-search",
        json={"query": "private query"},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert "secret" not in response.text


@pytest.mark.parametrize("provider", ["bing", "google"])
def test_web_search_unsupported_provider_is_501_without_calling_service(provider):
    web = FakeWebSearchService()
    _override_web(web)

    response = TestClient(app).post(
        "/api/v1/web-search",
        json={"query": "query", "provider": provider},
    )

    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "web_search_provider_not_supported"
    assert web.calls == []


def _web_settings(tmp_path, **overrides) -> Settings:
    values = {
        "web_search_enabled": True,
        "bocha_api_key": "SECRET_API_KEY",
        "bocha_endpoint": "https://secret-upstream.example/v1/search",
        "bocha_max_retries": 0,
        "web_search_cache_enabled": False,
        "web_search_cache_dir": str(tmp_path / "cache"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _service_with_transport(settings: Settings, handler) -> WebSearchService:
    service = WebSearchService(settings)
    service._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return service


def _valid_web_body(results: list[dict] | None = None) -> dict:
    return {
        "code": 200,
        "data": {
            "webPages": {"value": [] if results is None else results},
        },
    }


def _run_strict(service: WebSearchService, query: str = "PRIVATE_QUERY"):
    async def run():
        try:
            return await service.search_strict(query)
        finally:
            await service.aclose()

    return asyncio.run(run())


def test_strict_web_search_not_configured(tmp_path):
    service = WebSearchService(_web_settings(tmp_path, web_search_enabled=False))

    with pytest.raises(WebSearchNotConfigured):
        _run_strict(service)


@pytest.mark.parametrize(
    ("handler", "error_type"),
    [
        (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")), WebSearchTimeout),
        (lambda request: httpx.Response(429, text="PRIVATE_UPSTREAM_BODY"), WebSearchRateLimited),
        (lambda request: httpx.Response(500, text="PRIVATE_UPSTREAM_BODY"), WebSearchUpstreamError),
        (lambda request: httpx.Response(200, content=b"not-json"), WebSearchInvalidResponse),
        (
            lambda request: httpx.Response(200, json={"code": 200, "data": {}}),
            WebSearchInvalidResponse,
        ),
    ],
)
def test_strict_web_search_classifies_failures(tmp_path, handler, error_type):
    service = _service_with_transport(_web_settings(tmp_path), handler)

    with pytest.raises(error_type):
        _run_strict(service)


def test_strict_web_search_distinguishes_valid_empty_response(tmp_path):
    service = _service_with_transport(
        _web_settings(tmp_path),
        lambda request: httpx.Response(200, json=_valid_web_body()),
    )

    assert _run_strict(service) == []


def test_web_search_cache_key_isolates_provider_prefix_and_query():
    base = _cache_key("query", prefix="project-a", provider="bocha")

    assert base != _cache_key("query", prefix="project-b", provider="bocha")
    assert base != _cache_key("other", prefix="project-a", provider="bocha")
    assert base != _cache_key("query", prefix="project-a", provider="future-provider")


def test_web_search_global_concurrency_limit(tmp_path):
    active = 0
    peak = 0
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await release.wait()
        active -= 1
        return httpx.Response(200, json=_valid_web_body())

    service = _service_with_transport(
        _web_settings(tmp_path, web_search_max_concurrent=2),
        handler,
    )

    async def run():
        tasks = [asyncio.create_task(service.search_strict(f"query-{i}")) for i in range(5)]
        for _ in range(100):
            if active == 2:
                break
            await asyncio.sleep(0.001)
        assert active == 2
        release.set()
        try:
            await asyncio.gather(*tasks)
        finally:
            await service.aclose()

    asyncio.run(run())
    assert peak == 2


def test_strict_error_logs_do_not_leak_request_or_upstream_details(tmp_path, caplog):
    service = _service_with_transport(
        _web_settings(tmp_path),
        lambda request: httpx.Response(401, text="PRIVATE_UPSTREAM_BODY"),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(WebSearchUpstreamError):
        _run_strict(service)

    error_logs = "\n".join(
        record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
    )
    assert "PRIVATE_QUERY" not in error_logs
    assert "SECRET_API_KEY" not in error_logs
    assert "secret-upstream.example" not in error_logs
    assert "PRIVATE_UPSTREAM_BODY" not in error_logs


def test_day3_routes_declare_success_response_models():
    schema = app.openapi()
    rag_response = schema["paths"]["/api/v1/rag/search"]["post"]["responses"]["200"]
    web_response = schema["paths"]["/api/v1/web-search"]["post"]["responses"]["200"]

    assert rag_response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RAGSearchResponse"
    )
    assert web_response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/WebSearchResponse"
    )
