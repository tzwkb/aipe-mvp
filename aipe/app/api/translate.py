"""翻译接口：批量翻译 + 文件上传 + 任务查询。

设计参考技术设计文档 §4.2 / §5.6。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.dependencies import get_batch_processor
from app.schemas.translate import BatchTranslateResponse, TranslateRequest
from app.services.batch_processor import BatchProcessor
from app.utils.file_parser import parse_text_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translate", tags=["translate"])


_MAX_TEXTS_PER_REQUEST = 5000  # 单次请求上限：超过应该走文件上传

_CSV_HEADER = [
    "source",
    "translation",
    "translation_reason",
    "status",
    "content_type",
    "terminology_used",
    "rag_references",
    "tm_exact_match_used",
    "tm_exact_match_source",
    "tm_exact_match_target",
    "tm_exact_match_status",
    "tm_exact_match_score",
    "web_references",
    "web_search_triggered",
    "image_analysis",
    "error_msg",
]


def _new_task_id() -> str:
    return uuid.uuid4().hex[:16]


@router.post("", response_model=BatchTranslateResponse, summary="单句/批量翻译")
async def translate(
    request: TranslateRequest,
    processor: BatchProcessor = Depends(get_batch_processor),
) -> BatchTranslateResponse:
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts 不能为空")
    if len(request.texts) > _MAX_TEXTS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"单次请求超过上限 {_MAX_TEXTS_PER_REQUEST} 句，请改用 /translate/file",
        )

    task_id = request.task_id or _new_task_id()
    logger.info(
        "translate 接收: task=%s total=%d batch=%d enable_rag=%s enable_web_search=%s enable_vision=%s use_tm_exact_match=%s",
        task_id,
        len(request.texts),
        request.batch_size,
        request.enable_rag,
        request.enable_web_search,
        request.enable_vision,
        request.use_tm_exact_match,
    )
    return await processor.process(
        texts=request.texts,
        task_id=task_id,
        project_id=request.project_id,
        enable_rag=request.enable_rag,
        rag_threshold=request.rag_threshold,
        rag_top_k=request.rag_top_k,
        rag_collection=request.rag_collection,
        batch_size=request.batch_size,
        content_types=request.content_types,
        enable_web_search=request.enable_web_search,
        web_search_dense_threshold=request.web_search_dense_threshold,
        enable_vision=request.enable_vision,
        use_tm_exact_match=request.use_tm_exact_match,
    )


@router.post(
    "/file",
    response_model=BatchTranslateResponse,
    summary="文件上传翻译 (.txt/.csv/.xlsx/.json)",
)
async def translate_file(
    file: UploadFile = File(...),
    batch_size: int = Form(50, ge=1, le=500),
    enable_rag: bool = Form(True),
    rag_threshold: float = Form(0.85, ge=0.0, le=1.0),
    rag_top_k: int = Form(3, ge=1, le=10),
    enable_cluster: bool = Form(True, description="是否启用结构聚类 + 整组翻译路径；关闭后所有句子走单句路径"),
    dialog_mode: bool = Form(
        False,
        description=(
            "对话模式：按 id 聚合并按 time 排序，整段对话一次 LLM 调用。"
            "开启后跳过去重和结构聚类；无 id 的行退化为单句路径。"
            "需要文件中包含 id / 说话人 / time 列。"
        ),
    ),
    project_id: str | None = Form(None, description="项目档案 ID，如 wwm/zh-en；不填使用默认项目或旧全局状态"),
    rag_collection: str | None = Form(None, description="指定 RAG 检索的 Qdrant collection 名，不填则用默认值"),
    enable_web_search: bool = Form(
        False,
        description=(
            "启用外部网络搜索补充（博查）。仅在术语 0 命中且 RAG 弱召回时触发；"
            "需服务端 WEB_SEARCH_ENABLED=true 才生效。"
        ),
    ),
    web_search_dense_threshold: float | None = Form(
        None,
        ge=0.0,
        le=1.0,
        description="触发 Web 搜索的 dense top1 阈值；不填用配置默认 0.6",
    ),
    enable_vision: bool = Form(
        True,
        description="是否启用多模态模型对 Web 搜索配图进行分析；关闭后 image_analysis 始终为 None",
    ),
    use_tm_exact_match: bool = Form(
        False,
        description="是否直接采用 TM 精确源文匹配结果；命中则跳过 LLM 翻译，未命中照常翻译",
    ),
    task_id: str | None = Form(None),
    processor: BatchProcessor = Depends(get_batch_processor),
) -> BatchTranslateResponse:
    filename = file.filename or "uploaded.txt"
    try:
        raw = await file.read()
    finally:
        await file.close()

    try:
        items = parse_text_bytes(raw, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not items:
        raise HTTPException(status_code=400, detail="文件中未解析到任何待译文本")

    texts = [item["source"] for item in items]
    content_types = [item.get("content_type") for item in items]

    dialog_ids = speakers = times = None
    if dialog_mode:
        dialog_ids = [item.get("dialog_id") for item in items]
        speakers = [item.get("speaker") for item in items]
        times = [item.get("time") for item in items]
        if not any(dialog_ids):
            raise HTTPException(
                status_code=400,
                detail="dialog_mode=True 但文件中未找到 id / 对话编号 列，无法按对话聚合",
            )

    tid = task_id or _new_task_id()
    logger.info(
        "translate/file 接收: task=%s filename=%s total=%d enable_cluster=%s dialog_mode=%s enable_web_search=%s enable_vision=%s use_tm_exact_match=%s",
        tid,
        filename,
        len(texts),
        enable_cluster,
        dialog_mode,
        enable_web_search,
        enable_vision,
        use_tm_exact_match,
    )
    return await processor.process(
        texts=texts,
        task_id=tid,
        project_id=project_id,
        enable_rag=enable_rag,
        rag_threshold=rag_threshold,
        rag_top_k=rag_top_k,
        rag_collection=rag_collection,
        batch_size=batch_size,
        content_types=content_types,
        enable_cluster=enable_cluster,
        dialog_ids=dialog_ids,
        speakers=speakers,
        times=times,
        dialog_mode=dialog_mode,
        enable_web_search=enable_web_search,
        web_search_dense_threshold=web_search_dense_threshold,
        enable_vision=enable_vision,
        use_tm_exact_match=use_tm_exact_match,
    )


@router.get(
    "/task/{task_id}",
    response_model=BatchTranslateResponse,
    summary="查询批量任务状态与结果",
)
async def get_task(
    task_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
) -> BatchTranslateResponse:
    resp = processor.get_task_status(task_id)
    if resp is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return resp


@router.get(
    "/task/{task_id}/csv",
    summary="导出任务结果为 CSV（含 TM 精确匹配标记列）",
)
async def export_task_csv(
    task_id: str,
    processor: BatchProcessor = Depends(get_batch_processor),
) -> StreamingResponse:
    resp = processor.get_task_status(task_id)
    if resp is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    buf = io.StringIO()
    # utf-8-sig 让 Excel 直接识别为 UTF-8。
    writer = csv.writer(buf)
    writer.writerow(_CSV_HEADER)
    for r in resp.results:
        term_json = json.dumps(r.terminology_used, ensure_ascii=False) if r.terminology_used else ""
        rag_json = (
            json.dumps(r.rag_references, ensure_ascii=False) if r.rag_references else ""
        )
        web_json = (
            json.dumps(r.web_references, ensure_ascii=False) if r.web_references else ""
        )
        web_triggered = "" if r.web_search_triggered is None else str(r.web_search_triggered).lower()
        writer.writerow(
            [
                r.source,
                r.translation,
                r.translation_reason or "",
                r.status,
                r.content_type or "",
                term_json,
                rag_json,
                str(r.tm_exact_match_used).lower(),
                r.tm_exact_match_source or "",
                r.tm_exact_match_target or "",
                r.tm_exact_match_status or "",
                "" if r.tm_exact_match_score is None else str(r.tm_exact_match_score),
                web_json,
                web_triggered,
                r.image_analysis or "",
                r.error_msg or "",
            ]
        )
    data = ("﻿" + buf.getvalue()).encode("utf-8")
    # 文件名可能含中文等非 latin-1 字符，HTTP 头只能用 latin-1 编码，
    # 故用 RFC 5987 的 filename* 做 UTF-8 百分号编码；ASCII 回退也需可编码。
    filename = f"{task_id}.csv"
    quoted = quote(filename)
    ascii_fallback = filename.encode("ascii", "ignore").decode("ascii") or "task.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quoted}"
            ),
        },
    )
