"""translate API 端到端：覆盖 /translate、/translate/file、/translate/task/{id}、/csv。

LLM/RAG 通过依赖注入替换为 fake，不依赖外部服务。
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import (
    get_batch_processor,
    get_translation_pipeline,
)
from app.main import app
from app.schemas.translate import TranslationResult
from app.services.batch_processor import BatchProcessor
from app.services.project_service import get_project_resource_manager


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
        use_tm_exact_match=False,
        **_context,
    ):
        if not text or not text.strip():
            return TranslationResult(source=text, translation="", status="error", error_msg="empty")
        return TranslationResult(
            source=text,
            translation=text.upper(),
            status="success",
        )


def _override(processor: BatchProcessor, project_resources=None):
    app.dependency_overrides[get_batch_processor] = lambda: processor
    app.dependency_overrides[get_translation_pipeline] = lambda: processor.pipeline
    if project_resources is not None:
        app.dependency_overrides[get_project_resource_manager] = lambda: project_resources


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
            assert lines[0] == (
                "source,translation,translation_reason,status,content_type,terminology_used,"
                "rag_references,tm_exact_match_used,tm_exact_match_source,tm_exact_match_target,"
                "tm_exact_match_status,tm_exact_match_score,web_references,web_search_triggered,"
                "image_analysis,error_msg"
            )
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


def test_translate_json_forwards_parallel_dialog_context(tmp_path):
    class FakeDialogPipeline(FakePipeline):
        def __init__(self):
            self.dialog_calls = []

        async def translate_dialog(self, sources, speakers, **kwargs):
            self.dialog_calls.append((list(sources), list(speakers), kwargs))
            return [
                TranslationResult(source=source, translation=source.upper(), status="success")
                for source in sources
            ]

    pipeline = FakeDialogPipeline()
    processor = BatchProcessor(
        pipeline,  # type: ignore[arg-type]
        Settings(progress_dir=str(tmp_path), batch_size=50),
    )
    _override(processor)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/translate",
                json={
                    "project_id": None,
                    "texts": ["第一句", "第二句"],
                    "content_types": ["剧情", "剧情"],
                    "dialog_ids": ["scene-1", "scene-1"],
                    "speakers": ["甲", "乙"],
                    "times": [2.0, 1.0],
                    "addressees": ["乙", "甲"],
                    "scene_ids": ["scene.demo", "scene.demo"],
                    "relationship_stages": ["early", "early"],
                    "scene_tones": ["guarded", "guarded"],
                    "context_notes": ["later", "earlier"],
                    "enable_rag": False,
                    "task_id": "json_dialog_context",
                },
            )
        assert response.status_code == 200, response.text
        assert len(pipeline.dialog_calls) == 1
        sources, speakers, kwargs = pipeline.dialog_calls[0]
        assert sources == ["第二句", "第一句"]
        assert speakers == ["乙", "甲"]
        assert kwargs["addressees"] == ["甲", "乙"]
        assert kwargs["scene_ids"] == ["scene.demo", "scene.demo"]
        assert kwargs["context_notes"] == ["earlier", "later"]
    finally:
        _clear_overrides()


def test_translate_file_uses_project_workbook_layout_and_auto_dialog_mode(tmp_path):
    class FakeDialogPipeline(FakePipeline):
        def __init__(self):
            self.dialog_calls = []

        async def translate_dialog(self, sources, speakers, **kwargs):
            self.dialog_calls.append((list(sources), list(speakers), kwargs))
            return [
                TranslationResult(source=source, translation=source.upper(), status="success")
                for source in sources
            ]

    content_scope = {
        "workbook_layout": {
            "sheet": "试译",
            "sections": [
                {
                    "id": "dialogue",
                    "row_start": 2,
                    "row_end": 3,
                    "source_column": "B",
                    "speaker_column": "A",
                    "content_type": "剧情",
                    "dialog_id": "scene.demo",
                    "scene_id": "scene.demo",
                    "use_row_as_time": True,
                }
            ],
        }
    }

    class FakeProjectResources:
        def profile(self, project_id):
            assert project_id == "demo/zh-en"
            return SimpleNamespace(content_scope=content_scope)

    cover = pd.DataFrame([["说明页"]])
    trial = pd.DataFrame(
        [
            ["角色", "原文"],
            ["奥黛丽", "下午好。"],
            ["阿尔杰", "这是配方。"],
        ]
    )
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        cover.to_excel(writer, sheet_name="说明", header=False, index=False)
        trial.to_excel(writer, sheet_name="试译", header=False, index=False)

    pipeline = FakeDialogPipeline()
    processor = BatchProcessor(
        pipeline,  # type: ignore[arg-type]
        Settings(progress_dir=str(tmp_path), batch_size=50),
    )
    _override(processor, FakeProjectResources())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/translate/file",
                files={
                    "file": (
                        "trial.xlsx",
                        workbook.getvalue(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
                data={
                    "project_id": "demo/zh-en",
                    "enable_rag": "false",
                    "task_id": "profile_layout_dialog",
                },
            )
        assert response.status_code == 200, response.text
        assert response.json()["completed"] == 2
        assert len(pipeline.dialog_calls) == 1
        sources, speakers, kwargs = pipeline.dialog_calls[0]
        assert sources == ["下午好。", "这是配方。"]
        assert speakers == ["奥黛丽", "阿尔杰"]
        assert kwargs["dialog_id"] == "scene.demo"
        assert kwargs["scene_ids"] == ["scene.demo", "scene.demo"]
    finally:
        _clear_overrides()
