"""BatchProcessor 集成测试：切分 / 并发 / 断点续传。"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.schemas.translate import TranslationResult
from app.services.batch_processor import BatchProcessor


class FakePipeline:
    """伪 Pipeline：根据规则生成确定的译文，可注入失败规则。

    - ``translate_single``：每句独立返回 ``text.upper()``
    - ``translate_group``：整组返回带 ``[G]`` 前缀的 upper（区别于单句路径）
    """

    def __init__(self, *, fail_for: set[str] | None = None, sleep_each: float = 0.0):
        self.calls: list[str] = []
        self.group_calls: list[list[str]] = []
        self.project_ids: list[str | None] = []
        self.fail_for = fail_for or set()
        self.sleep_each = sleep_each

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
        self.calls.append(text)
        self.project_ids.append(project_id)
        if self.sleep_each:
            await asyncio.sleep(self.sleep_each)
        if text in self.fail_for:
            return TranslationResult(
                source=text,
                translation=f"[ERROR: AI_FAIL] {text}",
                status="error",
                error_msg="injected",
            )
        return TranslationResult(
            source=text,
            translation=text.upper(),
            status="success",
        )

    async def translate_group(
        self,
        sources,
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
        self.group_calls.append(list(sources))
        self.project_ids.append(project_id)
        if self.sleep_each:
            await asyncio.sleep(self.sleep_each)
        return [
            TranslationResult(source=s, translation=f"[G]{s.upper()}", status="success")
            for s in sources
        ]


def _settings(progress_dir, *, batch_size=2, max_concurrent=3, batch_sleep=0.0):
    return Settings(
        progress_dir=str(progress_dir),
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        batch_sleep=batch_sleep,
    )


def test_process_completes_all(tmp_path):
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    texts = ["a", "b", "c", "d", "e"]
    resp = asyncio.run(proc.process(texts, task_id="t1"))

    assert resp.status == "completed"
    assert resp.total == 5
    assert resp.completed == 5
    assert [r.source for r in resp.results] == texts
    assert all(r.translation == s.upper() for r, s in zip(resp.results, texts))


def test_process_resume_from_progress_file(tmp_path):
    """先跑一次失败一半，再用同 task_id 再跑一次：已完成 batch 不重复调用 pipeline。"""
    # 第一次：第二条会触发"提前停"——通过 monkey-patch 让 pipeline 抛异常
    pipe1 = FakePipeline()
    proc = BatchProcessor(pipe1, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    asyncio.run(proc.process(["a", "b", "c", "d"], task_id="t_resume"))
    assert len(pipe1.calls) == 4  # 全部跑完

    # 第二次：换一个 pipeline，应该一次都不调用（因为进度文件显示全部已完成）
    pipe2 = FakePipeline()
    proc2 = BatchProcessor(pipe2, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    resp = asyncio.run(proc2.process(["a", "b", "c", "d"], task_id="t_resume"))
    assert pipe2.calls == []
    assert resp.completed == 4
    assert [r.source for r in resp.results] == ["a", "b", "c", "d"]


def test_process_partial_resume(tmp_path):
    """手动注入 batch 0 已完成 → 第二次只跑剩下的 batch。"""
    from app.utils.progress_tracker import ProgressTracker

    tracker = ProgressTracker("t_partial", tmp_path)
    asyncio.run(tracker.init(total=4, batch_size=2, total_batches=2))
    asyncio.run(
        tracker.save_batch(
            0,
            [
                {"source": "a", "translation": "A_PRE", "status": "success"},
                {"source": "b", "translation": "B_PRE", "status": "success"},
            ],
        )
    )

    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    resp = asyncio.run(proc.process(["a", "b", "c", "d"], task_id="t_partial"))

    # 只应该跑 batch 1（"c", "d"），batch 0 复用
    assert pipe.calls == ["c", "d"]
    assert resp.completed == 4
    assert resp.results[0].translation == "A_PRE"  # 复用历史
    assert resp.results[2].translation == "C"


def test_process_failure_does_not_block(tmp_path):
    pipe = FakePipeline(fail_for={"b", "d"})
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    resp = asyncio.run(proc.process(["a", "b", "c", "d"], task_id="t_fail"))

    assert resp.completed == 4
    statuses = [r.status for r in resp.results]
    assert statuses == ["success", "error", "success", "error"]
    assert "[ERROR: AI_FAIL]" in resp.results[1].translation


def test_get_task_status_returns_none_for_unknown(tmp_path):
    proc = BatchProcessor(FakePipeline(), _settings(tmp_path))  # type: ignore[arg-type]
    assert proc.get_task_status("never_existed") is None


def test_get_task_status_after_run(tmp_path):
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    asyncio.run(proc.process(["a", "b", "c"], task_id="t_status"))
    snap = proc.get_task_status("t_status")
    assert snap is not None
    assert snap.completed == 3
    assert snap.status == "completed"


# ---------- 聚类调度 ----------


def test_process_clusters_templates_into_group_unit(tmp_path):
    """同模板句子应进入一次 translate_group 调用，非模板句仍走 translate_single。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = [
        "活动【柿业有成】获取",
        "无关短句一",
        "活动【燕衔嘉礼】获取",
        "活动【聆音响岁】获取",
        "另外一种完全不同的内容",
        "活动【踏岳寻珍】获取",
    ]
    resp = asyncio.run(proc.process(texts, task_id="t_cluster"))

    # 4 句模板进入同一个 group 调用
    assert len(pipe.group_calls) == 1
    assert pipe.group_calls[0] == [
        "活动【柿业有成】获取",
        "活动【燕衔嘉礼】获取",
        "活动【聆音响岁】获取",
        "活动【踏岳寻珍】获取",
    ]
    # 剩余 2 句走单句
    assert sorted(pipe.calls) == ["另外一种完全不同的内容", "无关短句一"]

    # 返回结果按原始输入顺序
    assert resp.completed == 6
    assert [r.source for r in resp.results] == texts
    # 模板组译文带 [G] 前缀（来自 FakePipeline.translate_group）
    assert resp.results[0].translation == "[G]活动【柿业有成】获取".upper()
    assert resp.results[1].translation == "无关短句一".upper()


def test_process_forwards_project_id_to_pipeline(tmp_path):
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]

    asyncio.run(proc.process(["一句散句"], task_id="t_project", project_id="nrc/zh-en"))

    assert pipe.project_ids == ["nrc/zh-en"]


def test_process_rejects_resume_when_project_context_changes(tmp_path):
    pipe1 = FakePipeline()
    proc1 = BatchProcessor(pipe1, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    asyncio.run(proc1.process(["一句散句"], task_id="t_project_context", project_id="isekai/en-de"))

    pipe2 = FakePipeline()
    proc2 = BatchProcessor(pipe2, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="project_id"):
        asyncio.run(proc2.process(["一句散句"], task_id="t_project_context", project_id="isekai/en-fr"))


def test_process_cluster_resume_preserves_order(tmp_path):
    """断点续传后，结果仍按原始顺序还原，已完成 unit 不再调用 pipeline。"""
    texts = [
        "活动【A】获取",
        "今天天气真好",
        "活动【B】获取",
        "活动【C】获取",
        "明天会下雨吗",
    ]

    pipe1 = FakePipeline()
    proc1 = BatchProcessor(pipe1, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    asyncio.run(proc1.process(texts, task_id="t_cluster_resume"))
    assert len(pipe1.group_calls) == 1
    assert sorted(pipe1.calls) == ["今天天气真好", "明天会下雨吗"]

    # 第二次：所有 unit 已完成，pipeline 不应被调用
    pipe2 = FakePipeline()
    proc2 = BatchProcessor(pipe2, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    resp = asyncio.run(proc2.process(texts, task_id="t_cluster_resume"))
    assert pipe2.group_calls == []
    assert pipe2.calls == []
    assert resp.completed == 5
    assert [r.source for r in resp.results] == texts


def test_process_cluster_disabled_falls_back_to_legacy(tmp_path):
    """关闭聚类时退回旧行为：纯单句 + 按 batch_size 切片。"""
    from app.config import Settings

    settings = Settings(
        progress_dir=str(tmp_path),
        batch_size=2,
        max_concurrent=3,
        batch_sleep=0.0,
        cluster_enabled=False,
    )
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, settings)  # type: ignore[arg-type]
    texts = ["活动【A】获取", "活动【B】获取", "活动【C】获取"]
    resp = asyncio.run(proc.process(texts, task_id="t_no_cluster"))

    # 即使是同模板句子，关闭聚类后也走单句
    assert pipe.group_calls == []
    assert pipe.calls == texts
    assert [r.source for r in resp.results] == texts


def test_get_task_status_returns_original_order_with_clusters(tmp_path):
    """通过 HTTP 查询任务状态时，应按原始输入顺序返回（不是 unit 顺序）。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["无关一", "活动【A】获取", "活动【B】获取", "无关二"]
    asyncio.run(proc.process(texts, task_id="t_status_order"))

    snap = proc.get_task_status("t_status_order")
    assert snap is not None
    assert [r.source for r in snap.results] == texts


# ---------- 去重 ----------


def test_process_dedup_same_text(tmp_path):
    """相同原文只翻译一次，结果 fan-out 到所有原始位置，保证一致性。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    texts = ["a", "b", "a", "c", "b"]  # "a" 和 "b" 各重复一次
    resp = asyncio.run(proc.process(texts, task_id="t_dedup"))

    # 只翻译了 3 句 unique：a, b, c
    assert pipe.calls == ["a", "b", "c"]
    assert pipe.group_calls == []

    # 返回 5 条结果，按原始顺序
    assert resp.total == 5
    assert resp.completed == 5
    assert [r.source for r in resp.results] == texts
    # 重复文本的译文完全一致
    assert resp.results[0].translation == resp.results[2].translation == "A"
    assert resp.results[1].translation == resp.results[4].translation == "B"
    assert resp.results[3].translation == "C"


def test_process_dedup_with_clusters(tmp_path):
    """去重与聚类同时工作：unique 中的同模板句仍走 group 路径。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = [
        "活动【A】获取",
        "活动【B】获取",
        "活动【A】获取",  # 重复
        "无关句",
        "活动【B】获取",  # 重复
    ]
    resp = asyncio.run(proc.process(texts, task_id="t_dedup_cluster"))

    # unique 有 3 句：活动A、活动B、无关句
    # 活动A + 活动B 结构相似，应进同一个 group
    assert len(pipe.group_calls) == 1
    assert pipe.group_calls[0] == ["活动【A】获取", "活动【B】获取"]
    assert pipe.calls == ["无关句"]

    # fan-out 后 5 条结果，重复位置译文一致
    assert resp.total == 5
    assert [r.source for r in resp.results] == texts
    assert resp.results[0].translation == resp.results[2].translation
    assert resp.results[1].translation == resp.results[4].translation


def test_process_dedup_resume(tmp_path):
    """去重后断点续传：重发同一输入，已完成的 unique unit 不再调用 pipeline。"""
    texts = ["a", "b", "a", "c", "b"]

    pipe1 = FakePipeline()
    proc1 = BatchProcessor(pipe1, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    asyncio.run(proc1.process(texts, task_id="t_dedup_resume"))
    assert pipe1.calls == ["a", "b", "c"]

    pipe2 = FakePipeline()
    proc2 = BatchProcessor(pipe2, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    resp = asyncio.run(proc2.process(texts, task_id="t_dedup_resume"))
    # 全部已完成，pipeline 不应被调用
    assert pipe2.calls == []
    assert pipe2.group_calls == []
    assert resp.completed == 5
    assert [r.source for r in resp.results] == texts
    assert resp.results[0].translation == "A"


def test_get_task_status_with_dedup(tmp_path):
    """HTTP 查询去重任务时，应正确 fan-out 并保留原始顺序。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    texts = ["x", "y", "x", "z", "y"]
    asyncio.run(proc.process(texts, task_id="t_dedup_status"))

    snap = proc.get_task_status("t_dedup_status")
    assert snap is not None
    assert snap.total == 5
    assert [r.source for r in snap.results] == texts
    assert snap.results[0].translation == snap.results[2].translation == "X"
    assert snap.results[1].translation == snap.results[4].translation == "Y"
    assert snap.results[3].translation == "Z"


# ---------- content_type 去重 ----------


def test_process_dedup_same_text_different_content_type(tmp_path):
    """相同原文但不同 content_type 应分别翻译，不合并。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    texts = ["a", "a", "a"]
    content_types = ["UI文本", None, "UI文本"]
    resp = asyncio.run(
        proc.process(texts, task_id="t_ct_dedup", content_types=content_types)
    )

    # "a"+"UI文本" 和 "a"+None 是两个 unique，分别翻译
    # 两个 "a" 会被结构聚类为同一组，走 translate_group
    assert len(pipe.group_calls) == 1
    assert sorted(pipe.group_calls[0]) == ["a", "a"]
    assert resp.total == 3
    assert resp.completed == 3
    # 原始顺序还原正确
    assert [r.source for r in resp.results] == texts


def test_process_dedup_same_text_same_content_type_merged(tmp_path):
    """相同原文且相同 content_type 应合并为一次翻译。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    texts = ["a", "a", "a"]
    content_types = ["UI文本", "UI文本", "UI文本"]
    resp = asyncio.run(
        proc.process(texts, task_id="t_ct_merge", content_types=content_types)
    )

    # 全部相同 (source, content_type)，只翻译一次
    assert pipe.calls == ["a"]
    assert resp.total == 3
    assert resp.completed == 3
    assert [r.source for r in resp.results] == texts


def test_process_content_type_passed_to_pipeline(tmp_path):
    """content_type 应正确传递到 pipeline.translate_single，跳过预分类。"""
    pipe = FakePipeline()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=2))  # type: ignore[arg-type]
    texts = ["a", "b"]
    content_types = ["UI文本", "剧情"]
    resp = asyncio.run(
        proc.process(texts, task_id="t_ct_pass", content_types=content_types)
    )
    assert resp.completed == 2
    assert [r.source for r in resp.results] == texts


# ---------- 对话模式 ----------


class FakePipelineWithDialog(FakePipeline):
    """扩展 FakePipeline，记录 translate_dialog 调用。"""

    def __init__(self, *, fail_for: set[str] | None = None):
        super().__init__(fail_for=fail_for)
        self.dialog_calls: list[dict] = []

    async def translate_dialog(
        self,
        sources,
        speakers,
        *,
        dialog_id=None,
        times=None,
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
        self.dialog_calls.append(
            {
                "sources": list(sources),
                "speakers": list(speakers),
                "dialog_id": dialog_id,
                "times": list(times) if times else None,
            }
        )
        return [
            TranslationResult(source=s, translation=f"[D:{dialog_id}]{s.upper()}", status="success")
            for s in sources
        ]


def test_dialog_mode_groups_by_id_and_sorts_by_time(tmp_path):
    pipe = FakePipelineWithDialog()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["A1", "A2", "B1", "A0", "B2"]
    dialog_ids = ["a", "a", "b", "a", "b"]
    speakers = ["若萍", "玩家", "甲", "若萍", "甲"]
    times = [5.0, 10.0, 1.0, None, 3.0]

    resp = asyncio.run(
        proc.process(
            texts,
            task_id="t_dlg_sort",
            dialog_ids=dialog_ids,
            speakers=speakers,
            times=times,
            dialog_mode=True,
        )
    )

    # 两个对话各调用一次 translate_dialog
    assert len(pipe.dialog_calls) == 2
    # 单句路径 / group 路径都没走
    assert pipe.calls == []
    assert pipe.group_calls == []

    # 对话 a：time=None 排第一（A0），然后 5.0（A1），10.0（A2）
    dlg_a = next(c for c in pipe.dialog_calls if c["dialog_id"] == "a")
    assert dlg_a["sources"] == ["A0", "A1", "A2"]
    assert dlg_a["speakers"] == ["若萍", "若萍", "玩家"]
    assert dlg_a["times"] == [None, 5.0, 10.0]

    # 对话 b：1.0 → 3.0
    dlg_b = next(c for c in pipe.dialog_calls if c["dialog_id"] == "b")
    assert dlg_b["sources"] == ["B1", "B2"]

    # 返回结果按原始输入顺序还原
    assert resp.completed == 5
    assert [r.source for r in resp.results] == texts
    assert resp.results[0].translation == "[D:a]A1"
    assert resp.results[2].translation == "[D:b]B1"


def test_dialog_mode_single_id_falls_back_to_single_unit(tmp_path):
    """只有一行的 dialog_id 不值得开 dialog prompt，退化为 single 路径。"""
    pipe = FakePipelineWithDialog()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["solo"]
    dialog_ids = ["only_one"]
    speakers = ["甲"]
    times = [1.0]

    resp = asyncio.run(
        proc.process(
            texts,
            task_id="t_dlg_solo",
            dialog_ids=dialog_ids,
            speakers=speakers,
            times=times,
            dialog_mode=True,
        )
    )

    # 没走 dialog 路径，走 single
    assert pipe.dialog_calls == []
    assert pipe.calls == ["solo"]
    assert resp.results[0].translation == "SOLO"


def test_dialog_mode_orphans_without_id_go_to_single(tmp_path):
    """无 dialog_id 的行应该走 single 路径，不影响 dialog units。"""
    pipe = FakePipelineWithDialog()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["orphan1", "dlg1", "dlg2", "orphan2"]
    dialog_ids = [None, "x", "x", None]
    speakers = [None, "甲", "乙", None]
    times = [None, 1.0, 2.0, None]

    resp = asyncio.run(
        proc.process(
            texts,
            task_id="t_dlg_orphan",
            dialog_ids=dialog_ids,
            speakers=speakers,
            times=times,
            dialog_mode=True,
        )
    )

    assert len(pipe.dialog_calls) == 1
    assert pipe.dialog_calls[0]["sources"] == ["dlg1", "dlg2"]
    assert sorted(pipe.calls) == ["orphan1", "orphan2"]
    assert [r.source for r in resp.results] == texts


def test_dialog_mode_skips_dedup(tmp_path):
    """对话模式下相同 source 不去重：上下文影响译法。"""
    pipe = FakePipelineWithDialog()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["你好", "你好"]
    dialog_ids = ["a", "b"]
    speakers = ["甲", "乙"]
    times = [None, None]

    resp = asyncio.run(
        proc.process(
            texts,
            task_id="t_dlg_no_dedup",
            dialog_ids=dialog_ids,
            speakers=speakers,
            times=times,
            dialog_mode=True,
        )
    )

    # 两个 id 各只有一行 → 都退化为 single；但 dedup 已跳过，所以两次都进 pipe.calls
    # 即使 source 相同，也独立翻译两次
    assert pipe.calls == ["你好", "你好"]
    assert resp.total == 2
    assert resp.completed == 2


def test_dialog_mode_disabled_keeps_legacy_path(tmp_path):
    """显式 dialog_mode=False（默认）时，dialog_ids 等参数被忽略，走默认去重+聚类路径。"""
    pipe = FakePipelineWithDialog()
    proc = BatchProcessor(pipe, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["a", "a", "b"]
    resp = asyncio.run(
        proc.process(
            texts,
            task_id="t_dlg_off",
            dialog_ids=["x", "x", "y"],  # 即使传了，dialog_mode=False 也不会用
            speakers=["甲", "乙", "丙"],
            times=[1.0, 2.0, 3.0],
        )
    )
    assert pipe.dialog_calls == []
    # "a" 被去重 → 只翻译 1 次 a + 1 次 b
    assert pipe.calls == ["a", "b"]
    assert [r.source for r in resp.results] == texts


def test_dialog_mode_resume_skips_completed_units(tmp_path):
    pipe1 = FakePipelineWithDialog()
    proc1 = BatchProcessor(pipe1, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    texts = ["A1", "A2", "B1", "B2"]
    dialog_ids = ["a", "a", "b", "b"]
    speakers = ["甲", "乙", "丙", "丁"]
    times = [1.0, 2.0, 1.0, 2.0]

    asyncio.run(
        proc1.process(
            texts,
            task_id="t_dlg_resume",
            dialog_ids=dialog_ids,
            speakers=speakers,
            times=times,
            dialog_mode=True,
        )
    )
    assert len(pipe1.dialog_calls) == 2

    pipe2 = FakePipelineWithDialog()
    proc2 = BatchProcessor(pipe2, _settings(tmp_path, batch_size=50))  # type: ignore[arg-type]
    resp = asyncio.run(
        proc2.process(
            texts,
            task_id="t_dlg_resume",
            dialog_ids=dialog_ids,
            speakers=speakers,
            times=times,
            dialog_mode=True,
        )
    )
    # 全部已完成，pipeline 不应被调用
    assert pipe2.dialog_calls == []
    assert pipe2.calls == []
    assert resp.completed == 4
    assert [r.source for r in resp.results] == texts
