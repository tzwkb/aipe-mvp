from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.config import Settings, get_settings
from app.services.batch_processor import BatchProcessor
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService, get_rag_service
from app.services.style_guide_service import (
    StyleGuideService,
    get_style_guide_service,
)
from app.services.terminology_service import (
    TerminologyService,
    get_terminology_service,
)
from app.services.translation_pipeline import TranslationPipeline
from app.services.vision_service import VisionService
from app.services.web_search_service import WebSearchService


SettingsDep = Depends(get_settings)


@lru_cache
def _build_llm_service() -> LLMService:
    return LLMService(get_settings())


def get_llm_service() -> LLMService:
    """LLMService 全局单例。配置变更需重启服务。"""
    return _build_llm_service()


@lru_cache
def _build_web_search_service() -> WebSearchService:
    return WebSearchService(get_settings())


def get_web_search_service() -> WebSearchService:
    """WebSearchService 全局单例（包含本地缓存与 in-flight 去重）。"""
    return _build_web_search_service()


@lru_cache
def _build_vision_service() -> VisionService:
    return VisionService(get_settings())


def get_vision_service() -> VisionService:
    """VisionService 全局单例（含进程内图片 URL 内存缓存）。"""
    return _build_vision_service()


@lru_cache
def _build_pipeline() -> TranslationPipeline:
    settings = get_settings()
    return TranslationPipeline(
        terminology_svc=get_terminology_service(),
        rag_svc=get_rag_service(),
        style_guide_svc=get_style_guide_service(),
        llm_svc=get_llm_service(),
        web_search_svc=get_web_search_service(),
        web_search_dense_threshold=settings.web_search_dense_threshold,
        vision_svc=get_vision_service(),
    )


def get_translation_pipeline() -> TranslationPipeline:
    """TranslationPipeline 全局单例。"""
    return _build_pipeline()


@lru_cache
def _build_batch_processor() -> BatchProcessor:
    return BatchProcessor(get_translation_pipeline(), get_settings())


def get_batch_processor() -> BatchProcessor:
    """BatchProcessor 全局单例。"""
    return _build_batch_processor()


__all__ = [
    "Settings",
    "get_settings",
    "SettingsDep",
    "get_llm_service",
    "get_terminology_service",
    "get_style_guide_service",
    "get_rag_service",
    "get_web_search_service",
    "get_vision_service",
    "get_translation_pipeline",
    "get_batch_processor",
    "LLMService",
    "TerminologyService",
    "StyleGuideService",
    "RAGService",
    "WebSearchService",
    "VisionService",
    "TranslationPipeline",
    "BatchProcessor",
]
