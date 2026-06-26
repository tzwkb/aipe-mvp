"""术语服务：加载 / 索引 / 命中匹配。

不再做"前置占位 + 后置校验 + 硬替换"。术语命中结果以参考段形式注入 Prompt，
由 LLM 结合语境决定是否采用，以避免出现僵硬的逐词替换。
"""

from __future__ import annotations

import logging
import re

from app.errors import TerminologyError
from app.schemas.terminology import TermEntry

logger = logging.getLogger(__name__)


class TerminologyService:
    """全局单例，启动时按需 ``load`` 整张术语表。"""

    def __init__(self) -> None:
        self.entries: list[TermEntry] = []
        self.term_dict: dict[str, str] = {}
        self.term_categories: dict[str, list[TermEntry]] = {}
        self._pattern: re.Pattern[str] | None = None
        self._duplicate_count: int = 0

    # ---------- load ----------

    def load(self, entries: list[TermEntry] | list[dict]) -> None:
        """加载术语条目，按 source 长度降序构建正则，确保长术语优先匹配。

        重复 source 保留首条，重复计数对外暴露用于上传响应。
        """
        normalized: list[TermEntry] = []
        seen: set[str] = set()
        dup = 0
        for raw in entries:
            entry = raw if isinstance(raw, TermEntry) else TermEntry(**raw)
            src = entry.source.strip()
            tgt = entry.target.strip()
            if not src or not tgt:
                continue
            if src in seen:
                dup += 1
                continue
            seen.add(src)
            normalized.append(entry.model_copy(update={"source": src, "target": tgt}))

        normalized.sort(key=lambda e: len(e.source), reverse=True)

        self.entries = normalized
        self.term_dict = {e.source: e.target for e in normalized}
        self.term_categories = {}
        for e in normalized:
            if e.category:
                self.term_categories.setdefault(e.category, []).append(e)

        if normalized:
            self._pattern = re.compile(
                "|".join(re.escape(e.source) for e in normalized)
            )
        else:
            self._pattern = None
        self._duplicate_count = dup

        logger.info(
            "术语表加载完成: total=%d, duplicates_skipped=%d, categories=%d",
            len(normalized),
            dup,
            len(self.term_categories),
        )

    @property
    def loaded(self) -> bool:
        return bool(self.entries)

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    # ---------- 命中匹配 ----------

    def find_matches(self, text: str) -> list[TermEntry]:
        """扫描原文中命中的术语，按出现顺序去重返回完整条目。

        长术语优先（``load`` 已按长度降序构建正则），同一术语多次出现只返回一次。
        命中结果会作为"强烈推荐"参考段注入 Prompt，由 LLM 结合语境决定是否采用。
        """
        if not text or self._pattern is None:
            return []

        seen: set[str] = set()
        matches: list[TermEntry] = []
        for m in self._pattern.finditer(text):
            src = m.group(0)
            if src in seen:
                continue
            seen.add(src)
            entry = next((e for e in self.entries if e.source == src), None)
            if entry is not None:
                matches.append(entry)
        return matches

    # ---------- 查询辅助 ----------

    def to_entries(self) -> list[TermEntry]:
        return list(self.entries)

    def lookup(self, source: str) -> str | None:
        return self.term_dict.get(source)


# ---------- 模块级单例 ----------

_singleton: TerminologyService | None = None


def get_terminology_service() -> TerminologyService:
    global _singleton
    if _singleton is None:
        _singleton = TerminologyService()
    return _singleton


def reset_terminology_service() -> None:
    """测试用：清空单例状态。"""
    global _singleton
    _singleton = None


__all__ = [
    "TerminologyService",
    "TerminologyError",
    "get_terminology_service",
    "reset_terminology_service",
]
