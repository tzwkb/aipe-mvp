from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.errors import StyleGuideError
from app.schemas.style_guide import (
    StyleGuideInfoResponse,
    StyleGuideUploadResponse,
)
from app.services.project_service import (
    ProjectProfileError,
    ProjectResourceManager,
    get_project_resource_manager,
)
from app.services.style_guide_service import (
    StyleGuideService,
    get_style_guide_service,
)
from app.utils.file_parser import parse_style_guide_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/style-guide", tags=["style-guide"])


_PREVIEW_LEN = 500


@router.post(
    "/upload",
    response_model=StyleGuideUploadResponse,
    summary="上传风格指南",
)
async def upload_style_guide(
    file: UploadFile = File(..., description="风格指南文件，.txt / .md / .markdown"),
    project_id: str | None = Query(None, description="项目档案 ID，如 wwm/zh-en；不填使用旧全局风格指南"),
    svc: StyleGuideService = Depends(get_style_guide_service),
    project_resources: ProjectResourceManager = Depends(get_project_resource_manager),
) -> StyleGuideUploadResponse:
    filename = file.filename or "style_guide"
    try:
        raw = await file.read()
    finally:
        await file.close()

    try:
        text = parse_style_guide_bytes(raw, filename)
        if project_id:
            svc = project_resources.replace_style_guide(project_id, text, filename=filename)
        else:
            svc.load(text, filename=filename)
    except ProjectProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StyleGuideError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    info = svc.info()
    return StyleGuideUploadResponse(
        filename=filename,
        char_count=int(info["char_count"]),
        line_count=int(info["line_count"]),
        message=f"风格指南加载成功 (loaded_at={info['loaded_at']})",
    )


@router.get(
    "",
    response_model=StyleGuideInfoResponse,
    summary="查询当前风格指南",
)
async def get_style_guide(
    full: bool = False,
    project_id: str | None = Query(None, description="项目档案 ID，如 wwm/zh-en；不填查询旧全局风格指南"),
    svc: StyleGuideService = Depends(get_style_guide_service),
    project_resources: ProjectResourceManager = Depends(get_project_resource_manager),
) -> StyleGuideInfoResponse:
    try:
        if project_id:
            svc = project_resources.style_guide(project_id)
    except ProjectProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    info = svc.info()
    rules = svc.get_rules()
    return StyleGuideInfoResponse(
        loaded=bool(info["loaded"]),
        filename=info["filename"],  # type: ignore[arg-type]
        char_count=int(info["char_count"]),
        line_count=int(info["line_count"]),
        preview=rules[:_PREVIEW_LEN],
        rules=rules if full else None,
    )
