from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.errors import TranslationError
from app.schemas.rag import (
    CorpusEntry,
    CorpusUploadResponse,
    RAGSearchRequest,
    RAGSearchResponse,
)
from app.services.rag_service import RAGService, get_rag_service
from app.utils.file_parser import CorpusParseError, parse_corpus_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


_ALLOWED_SUFFIX = {".xlsx", ".xls", ".csv", ".json"}


@router.post(
    "/corpus/upload",
    response_model=CorpusUploadResponse,
    summary="上传 RAG 双语语料",
)
async def upload_corpus(
    file: UploadFile = File(..., description="双语语料文件 .xlsx/.xls/.csv/.json"),
    collection: str | None = Form(None, description="目标 Qdrant collection 名，不填则用默认值；不存在时自动创建"),
    svc: RAGService = Depends(get_rag_service),
) -> CorpusUploadResponse:
    filename = file.filename or "uploaded"
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的语料格式: {suffix or '<无扩展名>'}（仅支持 .xlsx/.xls/.csv/.json）",
        )

    try:
        content = await file.read()
    finally:
        await file.close()

    try:
        parsed = parse_corpus_bytes(content, filename)
    except CorpusParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not parsed:
        raise HTTPException(status_code=400, detail="语料为空或全部行无效")

    entries = [CorpusEntry(**row) for row in parsed]
    try:
        indexed = await svc.index(entries, collection=collection)
    except TranslationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "RAG 语料上传完成: filename=%s parsed=%d indexed=%d",
        filename,
        len(parsed),
        indexed,
    )
    return CorpusUploadResponse(
        total=len(parsed),
        indexed=indexed,
        message=f"语料入库成功，filename={filename}",
    )


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    summary="手动测试 RAG 检索",
)
async def search(
    req: RAGSearchRequest,
    svc: RAGService = Depends(get_rag_service),
) -> RAGSearchResponse:
    try:
        results = await svc.search(
            req.query,
            threshold=req.threshold,
            top_k=req.top_k,
            collection=req.collection,
        )
    except TranslationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RAGSearchResponse(query=req.query, total=len(results), results=results)
