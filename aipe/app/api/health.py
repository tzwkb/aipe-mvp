from fastapi import APIRouter

from app.schemas.capabilities import (
    CapabilitiesResponse,
    CapabilitySet,
    HealthResponse,
    ScopedRAGCapability,
    WebSearchCapability,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="查询稳定能力契约",
)
async def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        capabilities=CapabilitySet(
            scoped_rag=ScopedRAGCapability(),
            web_search=WebSearchCapability(),
        )
    )
