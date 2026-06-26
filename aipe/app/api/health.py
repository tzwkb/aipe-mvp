from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="健康检查")
async def health() -> dict:
    return {"status": "ok", "service": "yanyun-ai-translate", "version": "1.0.0"}
