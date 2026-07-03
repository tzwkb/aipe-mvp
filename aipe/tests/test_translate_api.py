"""translate API 端到端：覆盖 /translate、/translate/file、/translate/task/{id}、/csv。

LLM/RAG 通过依赖注入替换为 fake，不依赖外部服务。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import (
    get_batch_processor,
    get_translation_pipeline,
)
from app.main import app
from app.schemas.translate import TranslationResult
from app.services.batch_processor import BatchProcessor


class FakePipeline:
    async def translate_single(
        self,
        text,
        *,
        enable_rag=True,
        rag_threshold=None,
        rag_top_k=None,
        content_type=None,
        rag_collection=None,
        enable_web_search=False,
        web_search_dense_threshold=None,
        enable_vision=True,
        project_id=None,
    ):
        if not text or not text.strip():
            return TranslationResult(source=text, translation="", status="error", error_msg="empty")
        return TranslationResult(
            source=text,
            translation=text.upper(),
            status="success",
        )


def _override(processor: BatchProcessor):
    app.dependency_overrides[get_batch_processor] = lambda: processor
    app.dependency_overrides[get_translation_pipeline] = lambda: processor.pipeline


def _clear_overrides():
    app.dependency_overrides.clear()


def test_translate_inline_batch(tmp_path):
    settings = Settings(progress_dir=str(tmp_path), batch_size=2)
    proc = BatchProcessor(FakePipeline(), settings)  # type: ignore[arg-type]
    _override(proc)
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/translate",
                json={"texts": ["你好", "世界", "燕云"], "batch_size": 2, "enable_rag": False},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["completed"] == 3
        assert data["status"] == "completed"
        assert [r["translation"] for r in data["results"]] == ["你好".upper(), "世界".upper(), "燕云".upper()]
    finally:
        _clear_overrides()


def test_translate_empty_texts_400(tmp_path):
    settings = Settings(progress_dir=str(tmp_path))
    proc = BatchProcessor(FakePipeline(), settings)  # type: ignore[arg-type]
    _override(proc)
    try:
        with TestClient(app) as client:
            resp = client.post("/api/v1/translate", json={"texts": []})
        assert resp.status_code == 400
    finally:
        _clear_overrides()


def test_translate_file_txt_and_csv_export(tmp_path):
    settings = Settings(progress_dir=str(tmp_path), batch_size=2)
    proc = BatchProcessor(FakePipeline(), settings)  # type: ignore[arg-type]
    _override(proc)
    try:
        with TestClient(app) as client:
            txt = "你好\n世界\n\n燕云".encode("utf-8")
            resp = client.post(
                "/api/v1/translate/file",
                files={"file": ("input.txt", txt, "text/plain")},
                data={
                    "batch_size": "2",
                    "enable_rag": "false",
                    "task_id": "task_csv",
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["completed"] == 3

            # 查询任务
            r2 = client.get("/api/v1/translate/task/task_csv")
            assert r2.status_code == 200
            assert r2.json()["completed"] == 3

            # 导出 CSV
            r3 = client.get("/api/v1/translate/task/task_csv/csv")
            assert r3.status_code == 200
            assert "text/csv" in r3.headers["content-type"]
            body = r3.content.decode("utf-8-sig")
            lines = [l for l in body.splitlines() if l]
            assert lines[0] == "source,translation,translation_reason,status,content_type,terminology_used,rag_references,web_references,web_search_triggered,image_analysis,error_msg"
            assert any("世界" in l and "世界".upper() in l for l in lines[1:])
    finally:
        _clear_overrides()


def test_get_unknown_task_404(tmp_path):
    settings = Settings(progress_dir=str(tmp_path))
    proc = BatchProcessor(FakePipeline(), settings)  # type: ignore[arg-type]
    _override(proc)
    try:
        with TestClient(app) as client:
            assert client.get("/api/v1/translate/task/no_such").status_code == 404
            assert client.get("/api/v1/translate/task/no_such/csv").status_code == 404
    finally:
        _clear_overrides()
