"""一次性入库脚本：把 data/corpus/corpus.xlsx 全量灌入 Qdrant。

特性：
- 读取大文件后切分成 chunk，按 ``--concurrency`` 并发分发到 RAGService
- 进度落盘到 ``data/progress/corpus_ingest.json``，中断重启自动跳过已处理偏移
- 支持 ``--reset`` 删除并重建 collection
- 支持 ``--limit`` / ``--offset`` 调试小批量
- 实时打印 done/total/elapsed/eta

⚠️ 升级到 hybrid 检索（dense + BM25 + RRF）后 collection schema 已变更
   （单匿名向量 → 命名向量 dense+sparse），首次升级**必须** ``--reset`` 重灌：

用法：
    conda run -n aipe python -m scripts.ingest_corpus --reset   # 升级后首次执行
    conda run -n aipe python -m scripts.ingest_corpus
    conda run -n aipe python -m scripts.ingest_corpus --limit 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.dependencies import get_llm_service
from app.schemas.rag import CorpusEntry
from app.services.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest_corpus")


DEFAULT_CORPUS = Path("data/corpus/corpus.xlsx")
DEFAULT_PROGRESS = Path("data/progress/corpus_ingest.json")
DEFAULT_CHUNK_SIZE = 500       # 每个并发任务覆盖的语料条数（内部仍按 100/批 embed）
DEFAULT_CONCURRENCY = 4        # 同时进行的 chunk 任务数


_STATUS_ALIASES = {"状态", "status", "审校状态", "review_status", "state", "级别"}


def _read_corpus(path: Path) -> list[CorpusEntry]:
    if not path.exists():
        raise SystemExit(f"语料文件不存在: {path}")
    logger.info("读取语料: %s", path)
    df = pd.read_excel(path, dtype=str)
    if df.shape[1] < 2:
        raise SystemExit(f"语料至少需要 2 列，实际 {df.shape[1]} 列")

    src_col, tgt_col = df.columns[0], df.columns[1]
    if "原文" in df.columns and "译文" in df.columns:
        src_col, tgt_col = "原文", "译文"

    # 找 status 列（大小写不敏感）
    col_norm = {str(c).strip().lower(): str(c) for c in df.columns}
    status_col: str | None = next(
        (col_norm[a.lower()] for a in _STATUS_ALIASES if a.lower() in col_norm), None
    )
    if status_col:
        logger.info("检测到审校状态列: %s", status_col)

    entries: list[CorpusEntry] = []
    skipped = 0
    for _, row in df.iterrows():
        src = _clean(row.get(src_col))
        tgt = _clean(row.get(tgt_col))
        if not src or not tgt:
            skipped += 1
            continue
        st = _clean(row.get(status_col)) if status_col else None
        entries.append(CorpusEntry(source=src, target=tgt, status=st or None))
    logger.info("语料解析完成: total=%d skipped(empty)=%d", len(entries), skipped)
    return entries


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {"completed_offset": 0, "indexed": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("进度文件解析失败，重置: %s", path)
        return {"completed_offset": 0, "indexed": 0}


def _save_progress(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    llm_svc = get_llm_service()
    svc = RAGService(settings, llm_svc)

    if args.reset:
        logger.warning("--reset 启用：将删除并重建 collection=%s", svc.collection)
        await svc.reset_collection()

    entries = _read_corpus(args.corpus)
    if args.offset:
        entries = entries[args.offset :]
    if args.limit:
        entries = entries[: args.limit]
    total = len(entries)
    if total == 0:
        logger.info("无可入库条目，退出")
        await svc.aclose()
        return

    progress_path = args.progress
    state = _load_progress(progress_path)
    if args.reset:
        state = {"completed_offset": 0, "indexed": 0}
    start_offset: int = int(state.get("completed_offset", 0))
    cumulative_indexed: int = int(state.get("indexed", 0))
    if start_offset >= total:
        logger.info("进度文件显示已全部完成 (%d/%d)，使用 --reset 可重跑", start_offset, total)
        await svc.aclose()
        return
    if start_offset:
        logger.info("从断点继续: offset=%d / total=%d", start_offset, total)

    chunk_size = max(1, args.chunk_size)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    progress_lock = asyncio.Lock()

    # chunk_indexed[i]=已成功条数；chunk_done[i]=布尔
    pending = list(range(start_offset, total, chunk_size))
    completed_status: dict[int, int] = {}  # offset -> indexed_count

    t0 = time.monotonic()

    async def _process_chunk(off: int) -> None:
        chunk = entries[off : off + chunk_size]
        async with sem:
            try:
                indexed = await svc.index(chunk)
            except Exception as exc:
                logger.error("chunk@%d 入库失败: %s", off, exc)
                raise
        async with progress_lock:
            completed_status[off] = indexed
            # 推进 watermark：按顺序找到第一个未完成的 offset
            nonlocal start_offset, cumulative_indexed
            wm = start_offset
            while wm in completed_status:
                cumulative_indexed += completed_status.pop(wm)
                wm += chunk_size
            start_offset = wm
            _save_progress(
                progress_path,
                {
                    "completed_offset": min(start_offset, total),
                    "indexed": cumulative_indexed,
                    "total": total,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            done_pct = min(start_offset, total) / total * 100
            elapsed = time.monotonic() - t0
            rate = (start_offset - int(state.get("completed_offset", 0))) / max(elapsed, 1e-6)
            eta_s = max(0, (total - start_offset) / max(rate, 1e-6))
            logger.info(
                "进度 %d/%d (%.1f%%) | indexed=%d | rate=%.1f/s | elapsed=%ds eta=%ds",
                min(start_offset, total),
                total,
                done_pct,
                cumulative_indexed,
                rate,
                int(elapsed),
                int(eta_s),
            )

    # 让 _ensure_collection 先跑一次，避免并发任务里抢 lock 时第一个 embed 探测失败放大错误
    await svc._ensure_collection(svc.collection)  # noqa: SLF001  脚本场景下允许访问

    try:
        await asyncio.gather(*(_process_chunk(off) for off in pending))
    finally:
        await svc.aclose()

    logger.info(
        "全部完成: total=%d indexed=%d elapsed=%ds",
        total,
        cumulative_indexed,
        int(time.monotonic() - t0),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将 corpus.xlsx 灌入 Qdrant")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="语料文件路径")
    p.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS, help="进度文件路径")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="单个并发任务覆盖条数")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发 chunk 数")
    p.add_argument("--limit", type=int, default=0, help="仅入库前 N 条（调试）")
    p.add_argument("--offset", type=int, default=0, help="从第 N 条开始（调试）")
    p.add_argument("--reset", action="store_true", help="清空 collection 与进度后重新入库")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
