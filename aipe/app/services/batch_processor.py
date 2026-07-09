"""批量翻译处理器：聚类 + 工作单元调度 + 断点续传。

设计要点：
- **结构聚类**：先把"句式高度相似"的句子聚成模板组，整组一次 LLM 调用走 ``translate_group``，
  强制使用同一英文句式（一致性优于逐句翻译）。
- **工作单元**：每个模板组 = 一个 unit（一次 LLM 调用）；其余单句按 ``batch_size`` 切片，
  每片 = 一个 unit（串行调用 ``translate_single``）。
- **并发**：不同 unit 通过 ``asyncio.Semaphore(max_concurrent)`` 并发，每完成一个 unit 落盘。
- **断点续传**：启动时读 ``.progress.json``，跳过已完成 unit。聚类对相同输入是确定性的，
  因此 unit 索引在 resume 间稳定。
- **单句容错**：``translate_single`` / ``translate_group`` 内部已捕获异常，整组解析失败会
  自动回退到逐句翻译，不会拖垮整个批量。
- **结果顺序**：unit 内顺序对应原始输入顺序的子序列；最终返回前按原始索引重排。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from app.config import Settings
from app.schemas.translate import BatchTranslateResponse, TranslationResult
from app.services.translation_pipeline import TranslationPipeline
from app.utils.cluster import cluster_by_structure
from app.utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


# 单句翻译可选 hook 类型（测试可以注入伪 pipeline）。
TranslateFn = Callable[[str], Awaitable[TranslationResult]]


UnitKind = Literal["group", "single", "dialog"]


@dataclass(frozen=True)
class WorkUnit:
    """一次 LLM 调度单元。

    - ``kind == "group"``：``indices`` 指向同模板组的原始索引列表，整组走 ``translate_group``
    - ``kind == "single"``：``indices`` 是若干离散单句的原始索引，逐句走 ``translate_single``
    - ``kind == "dialog"``：``indices`` 已按 ``time`` 排序，整段走 ``translate_dialog``；
      ``speakers`` / ``times`` 与 ``indices`` 平行，``dialog_id`` 用于日志与 prompt
    """

    kind: UnitKind
    indices: list[int]
    speakers: list[str | None] | None = None
    times: list[float | None] | None = None
    dialog_id: str | None = None


class BatchProcessor:
    """批量任务调度器。无状态：每次 ``process`` 自行加载 / 写入对应 task 的进度文件。"""

    def __init__(self, pipeline: TranslationPipeline, settings: Settings):
        self.pipeline = pipeline
        self.settings = settings
        self.batch_size = settings.batch_size
        self.max_concurrent = settings.max_concurrent
        self.batch_sleep = settings.batch_sleep
        self.progress_dir = settings.progress_dir

    async def process(
        self,
        texts: list[str],
        task_id: str,
        *,
        project_id: str | None = None,
        enable_rag: bool = True,
        rag_threshold: float | None = None,
        rag_top_k: int | None = None,
        rag_collection: str | None = None,
        batch_size: int | None = None,
        content_types: list[str | None] | None = None,
        enable_cluster: bool | None = None,
        dialog_ids: list[str | None] | None = None,
        speakers: list[str | None] | None = None,
        times: list[float | None] | None = None,
        dialog_mode: bool = False,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        use_tm_exact_match: bool = False,
    ) -> BatchTranslateResponse:
        """执行批量翻译。返回包含全部已完成结果的响应（按原始顺序）。

        默认路径：先按 ``(source, content_type)`` 去重，再结构聚类，按工作单元并发翻译。

        ``dialog_mode=True`` 启用对话路径：按 ``dialog_id`` 聚合并按 ``time`` 排序，
        整段对话一次 LLM 调用；**跳过去重和结构聚类**——同一句话在不同对话语境里译法
        可能不同，且对话中相邻句子的结构相似性是巧合，不应强制套同句式。无 ``dialog_id``
        或单条对话退化为 single unit 走 ``translate_single``。

        ``content_types`` / ``dialog_ids`` / ``speakers`` / ``times`` 均与 ``texts``
        一一对应，传入非空值时直接复用。
        """
        bs = batch_size or self.batch_size
        effective_project_id = project_id or self.settings.default_project
        effective_cluster: bool | None = None
        total_orig = len(texts)
        cts = content_types or [None] * total_orig
        if len(cts) != total_orig:
            raise ValueError(
                f"content_types 长度({len(cts)}) 与 texts 长度({total_orig}) 不一致"
            )

        if dialog_mode:
            d_ids = dialog_ids or [None] * total_orig
            d_speakers = speakers or [None] * total_orig
            d_times = times or [None] * total_orig
            if len(d_ids) != total_orig or len(d_speakers) != total_orig or len(d_times) != total_orig:
                raise ValueError(
                    "dialog_mode 下 dialog_ids / speakers / times 长度必须与 texts 一致"
                )

            # 对话模式跳过去重：dialog 上下文影响译法，强制 1:1 映射
            unique_texts = list(texts)
            unique_types = list(cts)
            orig_to_unique = list(range(total_orig))

            units = self._build_dialog_units(
                unique_texts, d_ids, d_speakers, d_times, batch_size=bs
            )
            total_unique = len(unique_texts)
            n_dialog = sum(1 for u in units if u.kind == "dialog")
            covered = sum(len(u.indices) for u in units if u.kind == "dialog")
            logger.info(
                "task=%s 对话模式：总句数=%d 对话组=%d (覆盖 %d 句) 散句 unit=%d",
                task_id,
                total_orig,
                n_dialog,
                covered,
                len(units) - n_dialog,
            )
        else:
            # 1. 去重：按 (source, content_type) 去重，保持首次出现顺序
            seen: dict[tuple[str, str | None], int] = {}
            unique_texts = []
            unique_types = []
            orig_to_unique = []
            for text, ct in zip(texts, cts):
                key = (text, ct)
                idx = seen.get(key)
                if idx is None:
                    idx = len(unique_texts)
                    seen[key] = idx
                    unique_texts.append(text)
                    unique_types.append(ct)
                orig_to_unique.append(idx)

            total_unique = len(unique_texts)
            if total_unique < total_orig:
                saved = total_orig - total_unique
                logger.info(
                    "task=%s 去重: %d -> %d (节省 %d 句, %.1f%%)",
                    task_id,
                    total_orig,
                    total_unique,
                    saved,
                    saved / total_orig * 100,
                )

            # 2. 对 unique_texts 做结构聚类、构建调度单元
            effective_cluster = enable_cluster if enable_cluster is not None else self.settings.cluster_enabled
            units = self._build_units(unique_texts, bs, enable_cluster=effective_cluster)

        total_units = len(units)

        tracker = ProgressTracker(task_id, self.progress_dir)
        state = await tracker.init(
            total=total_orig,
            batch_size=bs,
            total_batches=total_units,
            sources=unique_texts,
            content_types=unique_types,
            units=[list(u.indices) for u in units],
            orig_to_unique=orig_to_unique,
            context={
                "project_id": effective_project_id,
                "rag_collection": rag_collection,
                "enable_rag": bool(enable_rag),
                "rag_threshold": rag_threshold,
                "rag_top_k": rag_top_k,
                "enable_web_search": bool(enable_web_search),
                "web_search_dense_threshold": web_search_dense_threshold,
                "enable_vision": bool(enable_vision),
                "use_tm_exact_match": bool(use_tm_exact_match),
                "dialog_mode": bool(dialog_mode),
                "enable_cluster": effective_cluster,
            },
        )
        completed = set(state.get("completed_indices", []))
        if completed:
            logger.info(
                "task=%s 断点续传：已完成 %d / %d units", task_id, len(completed), total_units
            )

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _run_one_unit(idx: int, unit: WorkUnit) -> None:
            async with sem:
                t0 = time.monotonic()
                sources = [unique_texts[i] for i in unit.indices]
                # 整组取第一个非空 content_type；单句则逐句传入
                unit_cts = [unique_types[i] for i in unit.indices]
                if unit.kind == "group":
                    group_ct = next((ct for ct in unit_cts if ct), None)
                    results = await self._run_group(
                        sources,
                        enable_rag=enable_rag,
                        rag_threshold=rag_threshold,
                        rag_top_k=rag_top_k,
                        rag_collection=rag_collection,
                        content_type_hint=group_ct,
                        enable_web_search=enable_web_search,
                        web_search_dense_threshold=web_search_dense_threshold,
                        enable_vision=enable_vision,
                        project_id=effective_project_id,
                        use_tm_exact_match=use_tm_exact_match,
                    )
                elif unit.kind == "dialog":
                    dialog_ct = next((ct for ct in unit_cts if ct), None)
                    results = await self._run_dialog(
                        sources,
                        speakers=unit.speakers or [None] * len(sources),
                        times=unit.times,
                        dialog_id=unit.dialog_id,
                        enable_rag=enable_rag,
                        rag_threshold=rag_threshold,
                        rag_top_k=rag_top_k,
                        rag_collection=rag_collection,
                        content_type_hint=dialog_ct,
                        enable_web_search=enable_web_search,
                        web_search_dense_threshold=web_search_dense_threshold,
                        enable_vision=enable_vision,
                        project_id=effective_project_id,
                        use_tm_exact_match=use_tm_exact_match,
                    )
                else:
                    results = await self._run_singles(
                        sources,
                        enable_rag=enable_rag,
                        rag_threshold=rag_threshold,
                        rag_top_k=rag_top_k,
                        rag_collection=rag_collection,
                        content_types=unit_cts,
                        enable_web_search=enable_web_search,
                        web_search_dense_threshold=web_search_dense_threshold,
                        enable_vision=enable_vision,
                        project_id=effective_project_id,
                        use_tm_exact_match=use_tm_exact_match,
                    )
                await tracker.save_batch(
                    idx, [r.model_dump(mode="json") for r in results]
                )
                logger.info(
                    "task=%s unit %d/%d (%s, n=%d) 完成 %.2fs",
                    task_id,
                    idx + 1,
                    total_units,
                    unit.kind,
                    len(sources),
                    time.monotonic() - t0,
                )
                if self.batch_sleep > 0:
                    await asyncio.sleep(self.batch_sleep)

        coros = [
            _run_one_unit(i, u) for i, u in enumerate(units) if i not in completed
        ]
        if coros:
            await asyncio.gather(*coros)

        # 3. 收集 unique 结果并按 unique 索引重排
        final_state = tracker.load()
        results_by_batch = final_state.get("results_by_batch") or {}
        ordered_unique_dicts = self._restore_order(units, results_by_batch, total_unique)

        # 4. fan-out：复制到原始索引位置
        ordered_dicts = [ordered_unique_dicts[orig_to_unique[i]] for i in range(total_orig)]

        results = [TranslationResult(**d) for d in ordered_dicts]
        completed_count = len(results)
        progress_pct = (completed_count / total_orig * 100.0) if total_orig else 100.0

        was_resumed = bool(completed) and len(completed) < total_units
        if completed_count >= total_orig:
            status = "resumed-completed" if was_resumed else "completed"
        else:
            status = "running"

        return BatchTranslateResponse(
            task_id=task_id,
            total=total_orig,
            completed=completed_count,
            results=results,
            progress_pct=round(progress_pct, 2),
            status=status,
        )

    # ---------- 单元构建 ----------

    def _build_units(self, texts: list[str], batch_size: int, *, enable_cluster: bool = True) -> list[WorkUnit]:
        """对输入做结构聚类，构造调度单元列表。

        - 关闭聚类时退化为旧行为：按 ``batch_size`` 顺序切片，每片一个 single unit
        - 启用聚类时：每个模板组一个 group unit；剩余单句按 ``batch_size`` 切片
        - 输出顺序：先所有 group units，后所有 single units（便于早出一致性结果，
          且任意一种次序都不影响最终按原始索引重排）
        """
        if not texts:
            return []

        if not enable_cluster:
            return [
                WorkUnit(kind="single", indices=list(range(i, min(i + batch_size, len(texts)))))
                for i in range(0, len(texts), batch_size)
            ]

        res = cluster_by_structure(
            texts,
            pair_threshold=self.settings.cluster_pair_threshold,
            min_coverage=self.settings.cluster_min_coverage,
            max_group_size=self.settings.cluster_max_group_size,
        )

        units: list[WorkUnit] = [
            WorkUnit(kind="group", indices=list(grp)) for grp in res.groups
        ]
        for start in range(0, len(res.singletons), batch_size):
            chunk = res.singletons[start : start + batch_size]
            units.append(WorkUnit(kind="single", indices=chunk))

        if res.groups:
            logger.info(
                "结构聚类：总句数=%d，模板组=%d (覆盖 %d 句)，单句 unit=%d",
                len(texts),
                len(res.groups),
                sum(len(g) for g in res.groups),
                len(units) - len(res.groups),
            )
        return units

    def _build_dialog_units(
        self,
        texts: list[str],
        dialog_ids: list[str | None],
        speakers: list[str | None],
        times: list[float | None],
        *,
        batch_size: int,
    ) -> list[WorkUnit]:
        """按 ``dialog_id`` 聚合并按 ``time`` 排序，构造 dialog units。

        - 无 ``dialog_id`` 或单条对话 → 走 single unit（按 ``batch_size`` 切片）
        - 多条对话 → 每个 dialog_id 一个 dialog unit，按 ``time`` 升序排列（None 视为
          最早，稳定排序保持原始相对顺序）
        - 输出顺序：dialog units 在前，散句 single units 在后
        """
        if not texts:
            return []

        grouped: dict[str, list[int]] = {}
        first_seen_order: list[str] = []
        orphans: list[int] = []
        for i, did in enumerate(dialog_ids):
            key = str(did).strip() if did is not None else ""
            if not key:
                orphans.append(i)
                continue
            if key not in grouped:
                grouped[key] = []
                first_seen_order.append(key)
            grouped[key].append(i)

        units: list[WorkUnit] = []
        for did in first_seen_order:
            idxs = grouped[did]
            if len(idxs) == 1:
                # 单条对话退化为散句，避免单条 dialog unit 浪费上下文 prompt
                orphans.extend(idxs)
                continue
            idxs_sorted = sorted(
                idxs,
                key=lambda i: (
                    0 if times[i] is None else 1,
                    times[i] if times[i] is not None else 0.0,
                    i,
                ),
            )
            unit_speakers = [speakers[i] for i in idxs_sorted]
            unit_times = [times[i] for i in idxs_sorted]
            units.append(
                WorkUnit(
                    kind="dialog",
                    indices=idxs_sorted,
                    speakers=unit_speakers,
                    times=unit_times,
                    dialog_id=did,
                )
            )

        # 散句按原始顺序切片成 single units
        orphans.sort()
        for start in range(0, len(orphans), batch_size):
            chunk = orphans[start : start + batch_size]
            units.append(WorkUnit(kind="single", indices=chunk))
        return units

    # ---------- 单元执行 ----------

    async def _run_group(
        self,
        sources: list[str],
        *,
        enable_rag: bool,
        rag_threshold: float | None,
        rag_top_k: int | None,
        rag_collection: str | None = None,
        content_type_hint: str | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> list[TranslationResult]:
        from app.services.style_guide_service import ContentType

        ct: ContentType | None = None
        if content_type_hint:
            for c in ContentType:
                if c.value == content_type_hint:
                    ct = c
                    break
            if ct is None:
                # 宽松匹配
                for c in ContentType:
                    if c != ContentType.UNKNOWN and c.value in content_type_hint:
                        ct = c
                        break
        try:
            return await self.pipeline.translate_group(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                rag_collection=rag_collection,
                content_type=ct,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
                use_tm_exact_match=use_tm_exact_match,
            )
        except Exception as exc:  # 兜底：理论上 translate_group 不抛
            logger.exception("translate_group 兜底捕获: %s", exc)
            return [
                TranslationResult(
                    source=s,
                    translation=f"[ERROR: AI_FAIL] {s}",
                    status="error",
                    error_msg=str(exc),
                )
                for s in sources
            ]

    async def _run_dialog(
        self,
        sources: list[str],
        *,
        speakers: list[str | None],
        times: list[float | None] | None,
        dialog_id: str | None,
        enable_rag: bool,
        rag_threshold: float | None,
        rag_top_k: int | None,
        rag_collection: str | None = None,
        content_type_hint: str | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> list[TranslationResult]:
        from app.services.style_guide_service import ContentType

        ct: ContentType | None = None
        if content_type_hint:
            for c in ContentType:
                if c.value == content_type_hint:
                    ct = c
                    break
            if ct is None:
                for c in ContentType:
                    if c != ContentType.UNKNOWN and c.value in content_type_hint:
                        ct = c
                        break
        try:
            return await self.pipeline.translate_dialog(
                sources,
                speakers,
                dialog_id=dialog_id,
                times=times,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                rag_collection=rag_collection,
                content_type=ct,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
                use_tm_exact_match=use_tm_exact_match,
            )
        except Exception as exc:  # 兜底：理论上 translate_dialog 不抛
            logger.exception(
                "translate_dialog 兜底捕获 dialog_id=%s: %s", dialog_id, exc
            )
            return [
                TranslationResult(
                    source=s,
                    translation=f"[ERROR: DIALOG_FAIL] {s}",
                    status="error",
                    error_msg=str(exc),
                )
                for s in sources
            ]

    async def _run_singles(
        self,
        sources: list[str],
        *,
        enable_rag: bool,
        rag_threshold: float | None,
        rag_top_k: int | None,
        rag_collection: str | None = None,
        content_types: list[str | None] | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> list[TranslationResult]:
        from app.services.style_guide_service import ContentType

        results: list[TranslationResult] = []
        cts = content_types or [None] * len(sources)
        for src, ct_hint in zip(sources, cts):
            ct: ContentType | None = None
            if ct_hint:
                for c in ContentType:
                    if c.value == ct_hint:
                        ct = c
                        break
                if ct is None:
                    for c in ContentType:
                        if c != ContentType.UNKNOWN and c.value in ct_hint:
                            ct = c
                            break
            try:
                r = await self.pipeline.translate_single(
                    src,
                    enable_rag=enable_rag,
                    rag_threshold=rag_threshold,
                    rag_top_k=rag_top_k,
                    rag_collection=rag_collection,
                    content_type=ct,
                    enable_web_search=enable_web_search,
                    web_search_dense_threshold=web_search_dense_threshold,
                    enable_vision=enable_vision,
                    project_id=project_id,
                    use_tm_exact_match=use_tm_exact_match,
                )
            except Exception as exc:  # 兜底：理论上 translate_single 不抛
                logger.exception("translate_single 兜底捕获: %s", exc)
                r = TranslationResult(
                    source=src,
                    translation=f"[ERROR: AI_FAIL] {src}",
                    status="error",
                    error_msg=str(exc),
                )
            results.append(r)
        return results

    # ---------- 结果重排 ----------

    @staticmethod
    def _restore_order(
        units: list[WorkUnit],
        results_by_batch: dict,
        total: int,
    ) -> list[dict]:
        """按原始输入索引重排，剔除未完成位置后输出紧凑列表。"""
        slot: list[dict | None] = [None] * total
        for unit_idx, unit in enumerate(units):
            unit_results = results_by_batch.get(str(unit_idx))
            if not unit_results:
                continue
            if len(unit_results) != len(unit.indices):
                logger.warning(
                    "unit %d 结果数量与索引数不一致（got=%d expected=%d），已跳过",
                    unit_idx,
                    len(unit_results),
                    len(unit.indices),
                )
                continue
            for orig_idx, r in zip(unit.indices, unit_results):
                slot[orig_idx] = r
        return [r for r in slot if r is not None]

    # ---------- 查询 ----------

    def get_task_status(self, task_id: str) -> BatchTranslateResponse | None:
        """读取已落盘的任务状态（HTTP GET 用）。任务不存在返回 None。

        若进度文件包含 ``units`` 映射（聚类后任务），按原始输入顺序还原结果；
        老格式（无 units，逐批等于顺序切片）退化为按 batch_index 顺序返回。

        若进度文件包含 ``orig_to_unique``（去重后任务），先还原 unique 结果再做 fan-out。
        """
        tracker = ProgressTracker(task_id, self.progress_dir)
        if not tracker.file_path.exists():
            return None
        state = tracker.load()
        total = int(state.get("total") or 0)

        units_meta: list[list[int]] = state.get("units") or []
        results_by_batch = state.get("results_by_batch") or {}

        if units_meta:
            unit_objs = [WorkUnit(kind="single", indices=list(idxs)) for idxs in units_meta]
            # unique 结果数 = units 覆盖的最大索引 + 1，或从 sources 长度推导
            unique_sources = state.get("sources") or []
            total_unique = len(unique_sources) if unique_sources else total
            ordered_unique_dicts = self._restore_order(
                unit_objs, results_by_batch, total_unique
            )
        else:
            ordered_unique_dicts = tracker.collect_results(state)
            total_unique = len(ordered_unique_dicts)

        # 去重任务：fan-out 到原始索引
        orig_to_unique = state.get("orig_to_unique")
        if orig_to_unique and total_unique < total:
            ordered_dicts = [
                ordered_unique_dicts[orig_to_unique[i]]
                for i in range(total)
                if orig_to_unique[i] < len(ordered_unique_dicts)
            ]
        else:
            ordered_dicts = ordered_unique_dicts

        results = [TranslationResult(**d) for d in ordered_dicts]
        completed_count = len(results)
        progress_pct = (completed_count / total * 100.0) if total else 0.0
        status = "completed" if total and completed_count >= total else "running"
        return BatchTranslateResponse(
            task_id=task_id,
            total=total,
            completed=completed_count,
            results=results,
            progress_pct=round(progress_pct, 2),
            status=status,
        )


__all__ = ["BatchProcessor", "WorkUnit"]
