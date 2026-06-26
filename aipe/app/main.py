import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.config import get_settings
from app.errors import TranslationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("yanyun-ai-translate")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "启动 yanyun-ai-translate | provider=%s model=%s qdrant=%s:%s",
        settings.llm_provider,
        settings.llm_model,
        settings.qdrant_host,
        settings.qdrant_port,
    )
    yield
    logger.info("关闭 yanyun-ai-translate")


app = FastAPI(
    title="燕云十六声 AI 翻译 MVP",
    description="基于 FastAPI 的中→英游戏本地化翻译服务（术语+RAG+风格指南）",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.exception_handler(TranslationError)
async def translation_error_handler(request: Request, exc: TranslationError) -> JSONResponse:
    logger.exception("TranslationError: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "translation_failed", "detail": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_failed", "detail": exc.errors()},
    )


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "yanyun-ai-translate",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
