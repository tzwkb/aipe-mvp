"""ProgressTracker 单元测试。"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.utils.progress_tracker import ProgressTracker


@pytest.fixture
def tmp_progress(tmp_path, monkeypatch):
    # 每用例独立 task_id，避免 _locks 类变量串扰
    return tmp_path


def test_load_returns_empty_when_missing(tmp_progress):
    tr = ProgressTracker("never_existed_xyz", tmp_progress)
    state = tr.load()
    assert state["total"] == 0
    assert state["completed_indices"] == []
    assert state["results_by_batch"] == {}


def test_init_creates_file(tmp_progress):
    tr = ProgressTracker("task_init", tmp_progress)
    state = asyncio.run(tr.init(total=10, batch_size=5, total_batches=2))
    assert tr.file_path.exists()
    assert state["total"] == 10
    assert state["batch_size"] == 5
    assert state["total_batches"] == 2


def test_save_batch_and_collect(tmp_progress):
    tr = ProgressTracker("task_save", tmp_progress)
    asyncio.run(tr.init(total=4, batch_size=2, total_batches=2))
    asyncio.run(
        tr.save_batch(0, [{"source": "a", "translation": "A", "status": "success"}])
    )
    asyncio.run(
        tr.save_batch(
            1,
            [
                {"source": "b", "translation": "B", "status": "success"},
                {"source": "c", "translation": "C", "status": "success"},
            ],
        )
    )
    results = tr.collect_results()
    assert [r["source"] for r in results] == ["a", "b", "c"]


def test_save_batch_out_of_order_collects_in_order(tmp_progress):
    tr = ProgressTracker("task_oo", tmp_progress)
    asyncio.run(tr.init(total=4, batch_size=2, total_batches=2))
    # 故意先保存 batch 1 再 batch 0，确认 collect 按 index 排序
    asyncio.run(tr.save_batch(1, [{"source": "b", "translation": "B", "status": "success"}]))
    asyncio.run(tr.save_batch(0, [{"source": "a", "translation": "A", "status": "success"}]))
    results = tr.collect_results()
    assert [r["source"] for r in results] == ["a", "b"]


def test_init_preserves_existing_state(tmp_progress):
    tr = ProgressTracker("task_resume", tmp_progress)
    asyncio.run(tr.init(total=4, batch_size=2, total_batches=2))
    asyncio.run(
        tr.save_batch(0, [{"source": "a", "translation": "A", "status": "success"}])
    )
    # 二次 init 不应清空
    state = asyncio.run(tr.init(total=4, batch_size=2, total_batches=2))
    assert state["completed_indices"] == [0]


def test_init_treats_missing_tm_exact_flag_as_legacy_false(tmp_progress):
    tr = ProgressTracker("task_legacy_tm_context", tmp_progress)
    legacy_context = {"project_id": "wwm/zh-en", "enable_rag": True}
    asyncio.run(
        tr.init(
            total=1,
            batch_size=1,
            total_batches=1,
            sources=["source"],
            content_types=[None],
            units=[[0]],
            orig_to_unique=[0],
            context=legacy_context,
        )
    )

    state = asyncio.run(
        tr.init(
            total=1,
            batch_size=1,
            total_batches=1,
            sources=["source"],
            content_types=[None],
            units=[[0]],
            orig_to_unique=[0],
            context={**legacy_context, "use_tm_exact_match": False},
        )
    )

    assert state["context"] == legacy_context


def test_init_rejects_enabling_tm_exact_for_legacy_progress(tmp_progress):
    tr = ProgressTracker("task_legacy_tm_context_true", tmp_progress)
    legacy_context = {"project_id": "wwm/zh-en", "enable_rag": True}
    asyncio.run(
        tr.init(
            total=1,
            batch_size=1,
            total_batches=1,
            sources=["source"],
            content_types=[None],
            units=[[0]],
            orig_to_unique=[0],
            context=legacy_context,
        )
    )

    with pytest.raises(ValueError, match="context.use_tm_exact_match"):
        asyncio.run(
            tr.init(
                total=1,
                batch_size=1,
                total_batches=1,
                sources=["source"],
                content_types=[None],
                units=[[0]],
                orig_to_unique=[0],
                context={**legacy_context, "use_tm_exact_match": True},
            )
        )


@pytest.mark.parametrize("legacy_context_state", ["empty", "missing"])
def test_init_rejects_enabling_tm_exact_when_legacy_context_is_empty(
    tmp_progress, legacy_context_state
):
    tr = ProgressTracker(f"task_legacy_empty_context_{legacy_context_state}", tmp_progress)
    init_args = {
        "total": 1,
        "batch_size": 1,
        "total_batches": 1,
        "sources": ["source"],
        "content_types": [None],
        "units": [[0]],
        "orig_to_unique": [0],
    }
    asyncio.run(tr.init(**init_args, context={}))
    if legacy_context_state == "missing":
        legacy_state = json.loads(tr.file_path.read_text(encoding="utf-8"))
        legacy_state.pop("context")
        tr.file_path.write_text(json.dumps(legacy_state), encoding="utf-8")

    with pytest.raises(ValueError, match="context.use_tm_exact_match"):
        asyncio.run(
            tr.init(
                **init_args,
                context={"project_id": "wwm/zh-en", "use_tm_exact_match": True},
            )
        )


def test_atomic_write_no_tmp_leftover(tmp_progress):
    tr = ProgressTracker("task_atomic", tmp_progress)
    asyncio.run(tr.init(total=2, batch_size=1, total_batches=2))
    asyncio.run(tr.save_batch(0, [{"source": "a", "translation": "A", "status": "success"}]))
    leftover = list(tmp_progress.glob("*.tmp"))
    assert leftover == []
    # 文件可被重新解析
    raw = json.loads(tr.file_path.read_text(encoding="utf-8"))
    assert raw["task_id"] == "task_atomic"
