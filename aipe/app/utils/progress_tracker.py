"""断点续传：批量翻译的进度文件读写。

文件格式（``data/progress/{task_id}.progress.json``）:
```
{
  "task_id": "...",
  "total": 1234,
  "batch_size": 50,
  "total_batches": 25,
  "completed_indices": [0, 1, 2, ...],
  "results_by_batch": {"0": [TranslationResult, ...], "1": [...]},
  "last_updated": "2026-05-09T12:00:00Z"
}
```

设计要点：
- 写入用 ``tmp + replace`` 原子化，避免进程崩溃残留半截文件
- ``results_by_batch`` 用字符串 key（JSON 兼容），加载时不重组顺序，由调用方按 0..N 拼回
- 同一个 ``task_id`` 在进程内通过 ``asyncio.Lock`` 串行化写入，避免并发写交错
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """单个 task 的进度文件读写。线程内单实例，进程内同名 task 共享 ``asyncio.Lock``。"""

    _locks: dict[str, asyncio.Lock] = {}

    def __init__(self, task_id: str, progress_dir: str | Path):
        self.task_id = task_id
        self.progress_dir = Path(progress_dir)
        self.file_path = self.progress_dir / f"{task_id}.progress.json"
        self._lock = self._locks.setdefault(task_id, asyncio.Lock())

    # ---------- IO ----------

    def load(self) -> dict[str, Any]:
        """读取进度文件；不存在 / 解析失败时返回空骨架。"""
        if not self.file_path.exists():
            return self._empty_state()
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("进度文件解析失败 %s: %s（按空状态继续）", self.file_path, exc)
            return self._empty_state()

        # 字段补齐，避免老格式缺字段时下游 KeyError。
        state = self._empty_state()
        state.update(data)
        # 反序列化的 JSON object key 一定是 string，这里统一为 string，写回也用 string。
        state["completed_indices"] = list(state.get("completed_indices") or [])
        state["results_by_batch"] = dict(state.get("results_by_batch") or {})
        return state

    async def init(
        self,
        *,
        total: int,
        batch_size: int,
        total_batches: int,
        sources: list[str] | None = None,
        content_types: list[str | None] | None = None,
        units: list[list[int]] | None = None,
        orig_to_unique: list[int] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """初始化进度文件。已存在则按现状返回，便于断点续传。

        ``units`` 是每个 batch_index 对应的"原始输入索引列表"，用于查询时把结果
        按原始顺序还原（聚类后 batch 顺序 ≠ 原始顺序）。

        ``orig_to_unique`` 是"原始索引 → unique 索引"的映射，用于去重后的 fan-out。
        """
        requested_context = _normalize_context(context)
        async with self._lock:
            if self.file_path.exists():
                state = self.load()
                # 如果总句数变了（输入不一致）→ 警告但不覆盖，让上层决定。
                if state.get("total") != total:
                    logger.warning(
                        "断点续传：任务 %s 现有 total=%s，新输入 total=%s（保留已有进度）",
                        self.task_id,
                        state.get("total"),
                        total,
                    )
                mismatches = _state_mismatches(
                    state,
                    {
                        "total": total,
                        "batch_size": batch_size,
                        "total_batches": total_batches,
                        "sources": sources or [],
                        "content_types": content_types or [],
                        "units": units or [],
                        "orig_to_unique": orig_to_unique or [],
                        "context": requested_context,
                    },
                )
                if mismatches:
                    joined = ", ".join(mismatches)
                    raise ValueError(
                        f"断点续传任务 {self.task_id} 的输入/上下文不一致: {joined}; "
                        "请使用新的 task_id 或清理旧 progress 文件"
                    )
                return state

            state = self._empty_state()
            state.update(
                {
                    "task_id": self.task_id,
                    "total": total,
                    "batch_size": batch_size,
                    "total_batches": total_batches,
                    "sources": sources or [],
                    "content_types": content_types or [],
                    "units": units or [],
                    "orig_to_unique": orig_to_unique or [],
                    "context": requested_context,
                    "created_at": _now_iso(),
                    "last_updated": _now_iso(),
                }
            )
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write(state)
            return state

    async def save_batch(self, batch_index: int, results: list[dict]) -> dict[str, Any]:
        """把单个 batch 的结果落盘。``results`` 已是 dict 形式（TranslationResult.model_dump()）。"""
        async with self._lock:
            state = self.load()
            results_by_batch = state.setdefault("results_by_batch", {})
            completed = set(state.setdefault("completed_indices", []))

            results_by_batch[str(batch_index)] = results
            completed.add(batch_index)
            state["completed_indices"] = sorted(completed)
            state["last_updated"] = _now_iso()
            self._atomic_write(state)
            return state

    def collect_results(self, state: dict[str, Any] | None = None) -> list[dict]:
        """按 batch_index 顺序展平所有结果。已完成 batch 的结果按顺序拼接。"""
        st = state if state is not None else self.load()
        rb = st.get("results_by_batch") or {}
        out: list[dict] = []
        for i in sorted(int(k) for k in rb.keys()):
            out.extend(rb[str(i)])
        return out

    # ---------- helpers ----------

    def _atomic_write(self, state: dict[str, Any]) -> None:
        tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.file_path)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total": 0,
            "batch_size": 0,
            "total_batches": 0,
            "completed_indices": [],
            "results_by_batch": {},
            "sources": [],
            "content_types": [],
            "units": [],
            "orig_to_unique": [],
            "context": {},
            "created_at": None,
            "last_updated": None,
        }


def _normalize_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    return {k: v for k, v in context.items() if v is not None}


def _state_mismatches(state: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if state.get("total") != requested["total"]:
        mismatches.append("total")

    for key in ("batch_size", "total_batches", "sources", "content_types", "units", "orig_to_unique"):
        existing = state.get(key)
        if existing and existing != requested[key]:
            mismatches.append(key)

    existing_context_raw = state.get("context") or {}
    requested_context_raw = requested.get("context") or {}
    if existing_context_raw or requested_context_raw:
        if not existing_context_raw and requested_context_raw:
            legacy_tm_enabled = _context_with_compat_defaults(existing_context_raw)[
                "use_tm_exact_match"
            ]
            requested_tm_enabled = _context_with_compat_defaults(requested_context_raw)[
                "use_tm_exact_match"
            ]
            if legacy_tm_enabled != requested_tm_enabled:
                mismatches.append("context.use_tm_exact_match")
            return mismatches
        existing_context = _context_with_compat_defaults(existing_context_raw)
        requested_context = _context_with_compat_defaults(requested_context_raw)
        if existing_context != requested_context:
            changed = sorted(set(existing_context) | set(requested_context))
            mismatches.extend(f"context.{key}" for key in changed if existing_context.get(key) != requested_context.get(key))
    return mismatches


def _context_with_compat_defaults(context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(context)
    normalized.setdefault("use_tm_exact_match", False)
    return normalized


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["ProgressTracker"]
