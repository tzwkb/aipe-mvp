from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_web_search_service
from app.schemas.web_search import WebSearchRequest, WebSearchResponse
from app.services.project_service import (
    ProjectProfileError,
    ProjectResourceManager,
    get_project_resource_manager,
)
from app.services.web_search_service import (
    WebSearchInvalidResponse,
    WebSearchNotConfigured,
    WebSearchRateLimited,
    WebSearchService,
    WebSearchTimeout,
    WebSearchUpstreamError,
)

router = APIRouter(tags=["web-search"])


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


@router.post(
    "/web-search",
    response_model=WebSearchResponse,
    summary="手动外部搜索",
)
async def web_search(
    req: WebSearchRequest,
    svc: WebSearchService = Depends(get_web_search_service),
    projects: ProjectResourceManager = Depends(get_project_resource_manager),
) -> WebSearchResponse:
    provider = req.provider.strip().lower()
    if provider not in {"auto", "bocha"}:
        raise _http_error(
            status.HTTP_501_NOT_IMPLEMENTED,
            "web_search_provider_not_supported",
            "当前搜索 provider 不受支持",
        )

    prefix: str | None = None
    if req.project_id is not None:
        if not req.project_id.strip():
            raise _http_error(
                status.HTTP_400_BAD_REQUEST,
                "web_search_project_invalid",
                "Web Search 项目配置无效",
            )
        try:
            profile = projects.profile(req.project_id)
        except ProjectProfileError as exc:
            is_not_found = "not found" in str(exc).lower()
            raise _http_error(
                status.HTTP_404_NOT_FOUND if is_not_found else status.HTTP_400_BAD_REQUEST,
                "web_search_project_not_found" if is_not_found else "web_search_project_invalid",
                "Web Search 项目不存在" if is_not_found else "Web Search 项目配置无效",
            ) from exc
        if not profile.allow_web_search:
            raise _http_error(
                status.HTTP_403_FORBIDDEN,
                "web_search_forbidden",
                "该项目已禁用 Web Search",
            )
        prefix = profile.web_search_prefix

    try:
        results = await svc.search_strict(
            req.query,
            prefix=prefix,
            limit=req.top_k,
            provider="bocha",
        )
    except WebSearchNotConfigured as exc:
        raise _http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "web_search_not_configured",
            "Web Search 未配置",
        ) from exc
    except WebSearchTimeout as exc:
        raise _http_error(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "web_search_timeout",
            "Web Search 请求超时",
        ) from exc
    except WebSearchRateLimited as exc:
        raise _http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "web_search_rate_limited",
            "Web Search 上游限流",
        ) from exc
    except WebSearchInvalidResponse as exc:
        raise _http_error(
            status.HTTP_502_BAD_GATEWAY,
            "web_search_invalid_response",
            "Web Search 上游响应无效",
        ) from exc
    except WebSearchUpstreamError as exc:
        raise _http_error(
            status.HTTP_502_BAD_GATEWAY,
            "web_search_upstream_error",
            "Web Search 上游请求失败",
        ) from exc

    return WebSearchResponse(
        query=req.query,
        provider="bocha",
        project_id=req.project_id,
        total=len(results),
        results=results,
    )
