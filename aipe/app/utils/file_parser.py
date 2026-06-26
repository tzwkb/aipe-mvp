"""文件解析工具：术语表 / 双语语料 / 文本输入。

术语表（本期实现）：支持 .xlsx / .xls / .csv，固定两列 `中文` 和 `英语`，
也兼容常见英文别名（source / target / zh / en 等），可选 `category` / `notes` 列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from app.errors import StyleGuideError, TerminologyError


_SOURCE_ALIASES = {"中文", "原文", "source", "src", "zh", "zh_cn", "chinese"}
_TARGET_ALIASES = {"英语", "英文", "译文", "target", "tgt", "en", "english"}
_CATEGORY_ALIASES = {"category", "类型", "分类"}
_NOTES_ALIASES = {"notes", "note", "备注", "说明"}
_CONTENT_TYPE_ALIASES = {"content_type", "文本类型", "类型", "type", "category", "分类"}
_DIALOG_ID_ALIASES = {"id", "dialog_id", "对话id", "对话编号", "dialogue_id"}
_SPEAKER_ALIASES = {"说话人", "speaker", "角色", "name", "character", "actor"}
_TIME_ALIASES = {"time", "时间", "时刻", "timestamp", "t"}


def parse_terminology_file(path: str | Path) -> list[dict]:
    """解析术语表文件，返回 ``[{source, target, category?, notes?}]``。

    - 自动忽略空白行
    - 自动去除前后空格
    - 重复 source 保留首次出现，调用方可基于返回值进一步去重 / 报告
    """
    p = Path(path)
    if not p.exists():
        raise TerminologyError(f"文件不存在: {p}")

    suffix = p.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(p, dtype=str)
        elif suffix == ".csv":
            df = pd.read_csv(p, dtype=str)
        else:
            raise TerminologyError(f"不支持的术语表格式: {suffix}（仅支持 .xlsx/.xls/.csv）")
    except TerminologyError:
        raise
    except Exception as exc:  # pandas/openpyxl 抛出的解析异常统一包装
        raise TerminologyError(f"术语表解析失败: {exc}") from exc

    return _normalize_terminology_df(df)


def parse_terminology_bytes(data: bytes, filename: str) -> list[dict]:
    """从内存字节解析术语表（用于上传接口，避免落盘）。"""
    suffix = Path(filename).suffix.lower()
    import io

    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(data), dtype=str)
        elif suffix == ".csv":
            df = pd.read_csv(io.BytesIO(data), dtype=str)
        else:
            raise TerminologyError(f"不支持的术语表格式: {suffix}（仅支持 .xlsx/.xls/.csv）")
    except TerminologyError:
        raise
    except Exception as exc:
        raise TerminologyError(f"术语表解析失败: {exc}") from exc

    return _normalize_terminology_df(df)


def _normalize_terminology_df(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    col_map = _resolve_columns(df.columns)
    src_col = col_map["source"]
    tgt_col = col_map["target"]

    entries: list[dict] = []
    seen_sources: set[str] = set()

    for _, row in df.iterrows():
        source = _clean(row.get(src_col))
        target = _clean(row.get(tgt_col))
        if not source or not target:
            continue
        if source in seen_sources:
            continue
        seen_sources.add(source)

        entry: dict = {"source": source, "target": target}
        if cat_col := col_map.get("category"):
            if v := _clean(row.get(cat_col)):
                entry["category"] = v
        if notes_col := col_map.get("notes"):
            if v := _clean(row.get(notes_col)):
                entry["notes"] = v
        entries.append(entry)

    return entries


def _resolve_columns(columns: Iterable) -> dict[str, str]:
    """根据列名别名解析出 source/target/category/notes 实际列名。"""
    norm = {str(c).strip().lower(): str(c) for c in columns}

    def find(aliases: set[str]) -> str | None:
        for alias in aliases:
            if alias.lower() in norm:
                return norm[alias.lower()]
        return None

    src = find(_SOURCE_ALIASES)
    tgt = find(_TARGET_ALIASES)

    # 退化策略：若严格匹配未命中，按列序取前两列。
    if src is None or tgt is None:
        cols = list(columns)
        if len(cols) < 2:
            raise TerminologyError(
                f"术语表至少需要 2 列（中文/英语），实际列: {cols}"
            )
        src = src or str(cols[0])
        tgt = tgt or str(cols[1])

    out: dict[str, str] = {"source": src, "target": tgt}
    if cat := find(_CATEGORY_ALIASES):
        out["category"] = cat
    if notes := find(_NOTES_ALIASES):
        out["notes"] = notes
    return out


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


_STYLE_GUIDE_SUFFIXES = {".txt", ".md", ".markdown"}
_STYLE_GUIDE_MAX_BYTES = 1 * 1024 * 1024  # 1 MB，足以容纳长篇风格指南


def parse_style_guide_bytes(data: bytes, filename: str) -> str:
    """解析风格指南上传内容：UTF-8 解码、去 BOM、规范化换行、整体 strip。

    返回去除两端空白的文本正文。校验失败抛 ``StyleGuideError``。
    """
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in _STYLE_GUIDE_SUFFIXES:
        raise StyleGuideError(
            f"不支持的风格指南格式: {suffix}（仅支持 .txt/.md/.markdown）"
        )
    if len(data) > _STYLE_GUIDE_MAX_BYTES:
        raise StyleGuideError(
            f"风格指南文件过大: {len(data)} 字节，上限 {_STYLE_GUIDE_MAX_BYTES} 字节"
        )

    try:
        # utf-8-sig 自动剥离可能存在的 BOM；其他编码（GBK 等）暂不支持，避免静默乱码。
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise StyleGuideError(
            f"风格指南需为 UTF-8 编码（检测到非 UTF-8 字节）: {exc}"
        ) from exc

    # 统一换行；保留 markdown 内部空行，仅 strip 整体两端。
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise StyleGuideError("风格指南内容为空")
    return normalized


_CORPUS_SOURCE_ALIASES = _SOURCE_ALIASES | {"原文"}
_CORPUS_TARGET_ALIASES = _TARGET_ALIASES | {"译文"}
_CORPUS_CONTEXT_ALIASES = {"context", "上下文", "场景", "scene", "note", "备注"}
_CORPUS_STATUS_ALIASES = {"status", "状态", "审校状态", "review_status", "state", "级别"}


class CorpusParseError(Exception):
    """双语语料解析失败。"""


def parse_corpus_file(path: str | Path) -> list[dict]:
    """解析双语语料文件 (xlsx/xls/csv/json)，返回 ``[{source, target, context?}]``。"""
    p = Path(path)
    if not p.exists():
        raise CorpusParseError(f"文件不存在: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(p, dtype=str)
            return _normalize_corpus_df(df)
        if suffix == ".csv":
            df = pd.read_csv(p, dtype=str)
            return _normalize_corpus_df(df)
        if suffix == ".json":
            import json

            data = json.loads(p.read_text(encoding="utf-8"))
            return _normalize_corpus_list(data)
    except CorpusParseError:
        raise
    except Exception as exc:
        raise CorpusParseError(f"语料解析失败: {exc}") from exc

    raise CorpusParseError(f"不支持的语料格式: {suffix}（仅支持 .xlsx/.xls/.csv/.json）")


def parse_corpus_bytes(data: bytes, filename: str) -> list[dict]:
    """从内存字节解析双语语料（上传接口用）。"""
    suffix = Path(filename).suffix.lower()
    import io
    import json

    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(data), dtype=str)
            return _normalize_corpus_df(df)
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(data), dtype=str)
            return _normalize_corpus_df(df)
        if suffix == ".json":
            obj = json.loads(data.decode("utf-8-sig"))
            return _normalize_corpus_list(obj)
    except CorpusParseError:
        raise
    except Exception as exc:
        raise CorpusParseError(f"语料解析失败: {exc}") from exc

    raise CorpusParseError(f"不支持的语料格式: {suffix}（仅支持 .xlsx/.xls/.csv/.json）")


def _normalize_corpus_df(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    col_map = _resolve_corpus_columns(df.columns)
    src_col, tgt_col = col_map["source"], col_map["target"]
    ctx_col = col_map.get("context")
    status_col = col_map.get("status")

    out: list[dict] = []
    for _, row in df.iterrows():
        src = _clean(row.get(src_col))
        tgt = _clean(row.get(tgt_col))
        if not src or not tgt:
            continue
        entry: dict = {"source": src, "target": tgt}
        if ctx_col:
            ctx = _clean(row.get(ctx_col))
            if ctx:
                entry["context"] = ctx
        if status_col:
            st = _clean(row.get(status_col))
            if st:
                entry["status"] = st
        out.append(entry)
    return out


def _normalize_corpus_list(obj) -> list[dict]:
    if not isinstance(obj, list):
        raise CorpusParseError("JSON 语料必须是数组：[{source, target, context?}, ...]")
    src_lc = {a.lower() for a in _CORPUS_SOURCE_ALIASES}
    tgt_lc = {a.lower() for a in _CORPUS_TARGET_ALIASES}
    ctx_lc = {a.lower() for a in _CORPUS_CONTEXT_ALIASES}
    status_lc = {a.lower() for a in _CORPUS_STATUS_ALIASES}
    out: list[dict] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        src = ""
        tgt = ""
        ctx: str | None = None
        st: str | None = None
        for key in item.keys():
            kl = str(key).strip().lower()
            if not src and kl in src_lc:
                src = _clean(item[key])
            elif not tgt and kl in tgt_lc:
                tgt = _clean(item[key])
            elif ctx is None and kl in ctx_lc:
                v = _clean(item[key])
                ctx = v or None
            elif st is None and kl in status_lc:
                v = _clean(item[key])
                st = v or None
        if not src or not tgt:
            continue
        entry: dict = {"source": src, "target": tgt}
        if ctx:
            entry["context"] = ctx
        if st:
            entry["status"] = st
        out.append(entry)
    return out


def _resolve_corpus_columns(columns: Iterable) -> dict[str, str]:
    norm = {str(c).strip().lower(): str(c) for c in columns}

    def find(aliases: set[str]) -> str | None:
        for alias in aliases:
            if alias.lower() in norm:
                return norm[alias.lower()]
        return None

    src = find(_CORPUS_SOURCE_ALIASES)
    tgt = find(_CORPUS_TARGET_ALIASES)
    if src is None or tgt is None:
        cols = list(columns)
        if len(cols) < 2:
            raise CorpusParseError(f"语料文件至少需要 2 列（原文/译文），实际列: {cols}")
        src = src or str(cols[0])
        tgt = tgt or str(cols[1])
    out: dict[str, str] = {"source": src, "target": tgt}
    if ctx := find(_CORPUS_CONTEXT_ALIASES):
        out["context"] = ctx
    if st := find(_CORPUS_STATUS_ALIASES):
        out["status"] = st
    return out


def parse_text_file(path: str | Path) -> list[dict]:
    """从磁盘读取待翻译文本，返回 ``[{source, content_type}, ...]``。"""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"文件不存在: {p}")
    return parse_text_bytes(p.read_bytes(), p.name)


def parse_text_bytes(data: bytes, filename: str) -> list[dict]:
    """从内存字节解析待翻译文本（上传接口用）。

    支持：
    - .txt   每行一句，按 ``\\n`` 切分（content_type 始终为 None）
    - .csv   多列。优先取 ``中文/原文/source`` 列；可选 ``content_type/文本类型`` 列
    - .xlsx/.xls  同上
    - .json  字符串数组 ``["...", "..."]`` 或对象数组 ``[{"source": "...", "content_type": "..."}, ...]``

    返回 ``[{source: str, content_type: str | None}, ...]``。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"TXT 文件需为 UTF-8 编码: {exc}") from exc
        lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
        return [{"source": line, "content_type": None} for line in lines]

    if suffix == ".csv":
        import io

        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str)
        except Exception as exc:
            raise ValueError(f"CSV 解析失败: {exc}") from exc
        return _extract_source_with_type(df)

    if suffix in {".xlsx", ".xls"}:
        import io

        try:
            df = pd.read_excel(io.BytesIO(data), dtype=str)
        except Exception as exc:
            raise ValueError(f"Excel 解析失败: {exc}") from exc
        return _extract_source_with_type(df)

    if suffix == ".json":
        import json

        try:
            obj = json.loads(data.decode("utf-8-sig"))
        except Exception as exc:
            raise ValueError(f"JSON 解析失败: {exc}") from exc
        return _extract_source_from_json(obj)

    raise ValueError(
        f"不支持的待译文件格式: {suffix}（仅支持 .txt/.csv/.xlsx/.xls/.json）"
    )


def _extract_source_with_type(df: pd.DataFrame) -> list[dict]:
    """从 DataFrame 提取 source 列和可选的 content_type / 对话相关列。

    当文件中出现 ``id`` / ``说话人`` / ``time`` 中的任意列时，会把它们附加到每个 item
    上（键名固定为 ``dialog_id`` / ``speaker`` / ``time``）。仅在该列存在时附加，避免
    污染普通文本翻译路径的返回结构。
    """
    if df.empty:
        return []
    norm = {str(c).strip().lower(): str(c) for c in df.columns}

    def _find(aliases: set[str]) -> str | None:
        for alias in aliases:
            if alias.lower() in norm:
                return norm[alias.lower()]
        return None

    # 找 source 列
    src_col = _find(_SOURCE_ALIASES) or str(df.columns[0])

    # 找 content_type 列
    type_col = _find(_CONTENT_TYPE_ALIASES)

    # 对话相关列（任一列存在即开启该字段透传）
    id_col = _find(_DIALOG_ID_ALIASES)
    speaker_col = _find(_SPEAKER_ALIASES)
    time_col = _find(_TIME_ALIASES)

    # content_type 别名与 dialog_id/speaker/time 在 type/category/name 上可能冲突；
    # 若 type_col 与对话列指向同一物理列，content_type 让位，避免把"说话人"误读成
    # 文本类型。
    if type_col is not None and type_col in {id_col, speaker_col, time_col}:
        type_col = None

    out: list[dict] = []
    for _, row in df.iterrows():
        src = _clean(row.get(src_col))
        if not src:
            continue
        entry: dict = {"source": src, "content_type": None}
        if type_col:
            ct = _clean(row.get(type_col))
            entry["content_type"] = ct or None
        if id_col:
            entry["dialog_id"] = _clean(row.get(id_col)) or None
        if speaker_col:
            entry["speaker"] = _clean(row.get(speaker_col)) or None
        if time_col:
            entry["time"] = _parse_time(row.get(time_col))
        out.append(entry)
    return out


def _parse_time(value) -> float | None:
    """尝试把单元格解析为浮点时刻；解析失败 / 缺失返回 None。"""
    s = _clean(value)
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _extract_source_from_json(obj) -> list[dict]:
    if not isinstance(obj, list):
        raise ValueError("JSON 输入必须是数组：[\"...\"] 或 [{\"source\": \"...\"}, ...]")

    src_lc = {a.lower() for a in _SOURCE_ALIASES}
    ct_lc = {a.lower() for a in _CONTENT_TYPE_ALIASES}
    id_lc = {a.lower() for a in _DIALOG_ID_ALIASES}
    speaker_lc = {a.lower() for a in _SPEAKER_ALIASES}
    time_lc = {a.lower() for a in _TIME_ALIASES}

    out: list[dict] = []
    for item in obj:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append({"source": s, "content_type": None})
            continue
        if not isinstance(item, dict):
            continue
        src: str | None = None
        ct: str | None = None
        did: str | None = None
        speaker: str | None = None
        ts: float | None = None
        has_id = has_speaker = has_time = False
        for key, value in item.items():
            kl = str(key).strip().lower()
            if src is None and kl in src_lc:
                src = _clean(value)
            elif ct is None and kl in ct_lc and kl not in id_lc | speaker_lc | time_lc:
                ct = _clean(value)
            elif kl in id_lc:
                has_id = True
                did = _clean(value) or None
            elif kl in speaker_lc:
                has_speaker = True
                speaker = _clean(value) or None
            elif kl in time_lc:
                has_time = True
                ts = _parse_time(value)
        if not src:
            continue
        entry: dict = {"source": src, "content_type": ct or None}
        if has_id:
            entry["dialog_id"] = did
        if has_speaker:
            entry["speaker"] = speaker
        if has_time:
            entry["time"] = ts
        out.append(entry)
    return out


def _extract_sources_only(items: list[dict]) -> list[str]:
    """辅助函数：从解析结果中提取纯 source 字符串列表（向后兼容）。"""
    return [item["source"] for item in items]
