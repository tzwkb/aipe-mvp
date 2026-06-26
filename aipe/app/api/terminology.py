from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.errors import TerminologyError
from app.schemas.terminology import (
    TerminologyListResponse,
    TerminologyUploadResponse,
)
from app.services.terminology_service import (
    TerminologyService,
    get_terminology_service,
)
from app.utils.file_parser import parse_terminology_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminology", tags=["terminology"])


_ALLOWED_SUFFIX = {".xlsx", ".xls", ".csv"}


@router.post(
    "/upload",
    response_model=TerminologyUploadResponse,
    summary="上传术语表",
)
async def upload_terminology(
    file: UploadFile = File(..., description="术语表文件，.xlsx / .xls / .csv"),
    svc: TerminologyService = Depends(get_terminology_service),
) -> TerminologyUploadResponse:
    filename = file.filename or "uploaded"
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的术语表格式: {suffix or '<无扩展名>'}（仅支持 .xlsx/.xls/.csv）",
        )

    try:
        content = await file.read()
    finally:
        await file.close()

    try:
        parsed = parse_terminology_bytes(content, filename)
    except TerminologyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not parsed:
        raise HTTPException(status_code=400, detail="术语表为空或全部行无效")

    previous_total = len(svc.entries)
    previous_sources = set(svc.term_dict.keys())

    svc.load(parsed)

    new_sources = set(svc.term_dict.keys())
    added = len(new_sources - previous_sources)
    updated = len(new_sources & previous_sources)
    logger.info(
        "术语表上传完成: filename=%s parsed=%d total=%d added=%d updated=%d duplicates=%d",
        filename,
        len(parsed),
        len(svc.entries),
        added,
        updated,
        svc.duplicate_count,
    )
    return TerminologyUploadResponse(
        total=len(svc.entries),
        added=added,
        updated=updated,
        message=(
            f"术语表加载成功，previous_total={previous_total}, "
            f"duplicates_skipped={svc.duplicate_count}"
        ),
    )


@router.get(
    "",
    response_model=TerminologyListResponse,
    summary="查询当前术语表",
)
async def list_terminology(
    limit: int = 100,
    offset: int = 0,
    svc: TerminologyService = Depends(get_terminology_service),
) -> TerminologyListResponse:
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit 取值范围 [1, 1000]")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset 不能为负")
    items = svc.entries[offset : offset + limit]
    return TerminologyListResponse(total=len(svc.entries), items=items)
