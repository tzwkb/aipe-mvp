from fastapi import APIRouter

from app.api import health, rag, style_guide, terminology, translate, web_search

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(translate.router)
api_router.include_router(terminology.router)
api_router.include_router(rag.router)
api_router.include_router(style_guide.router)
api_router.include_router(web_search.router)
