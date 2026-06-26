from fastapi import APIRouter

from app.api import health, rag, style_guide, terminology, translate

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(translate.router)
api_router.include_router(terminology.router)
api_router.include_router(rag.router)
api_router.include_router(style_guide.router)
