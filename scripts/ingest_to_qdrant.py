"""清洗后语料入库 Qdrant —— 异步并发 + 断点续传。

策略：
- batch_size=200, max_concurrent=15
- 失败批次进 failed.jsonl，不阻塞其他批次
- 每完成 N 批 flush 进度到 progress.json，崩了重启自动跳过已完成批次
- ID = uuid5(source||target||status)  （与项目 rag_service.py 一致）
- Dense: text-embedding-3-large, 1024 维（截断）
- Sparse: jieba 分词 + 词频，与项目 _to_sparse 一致
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI, APIError, APIStatusError, APITimeoutError, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
import jieba

# --- 控制台 UTF-8 输出 ---
sys.stdout.reconfigure(encoding='utf-8')

# --- 配置 ---
BASE = Path(r"E:\Langlobal\AIPEMVP_0526")
CORPUS_PATH = BASE / "cleaned_corpus" / "cleaned_corpus.jsonl"
STATE_PATH = BASE / "cleaned_corpus" / "ingest_state.json"
FAILED_PATH = BASE / "cleaned_corpus" / "ingest_failed.jsonl"

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = "https://api.vectorengine.cn/v1"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1024

QDRANT_URL = "http://localhost:6333"
COLLECTION = "yanyun_corpus"

BATCH_SIZE = 200
MAX_CONCURRENT = 15
MAX_RETRIES = 5
FLUSH_EVERY = 10  # 每 10 批 flush 一次 state

_VEC_DENSE = "dense"
_VEC_SPARSE = "sparse"
_ID_NAMESPACE = uuid.UUID("6f3a8b56-2c0d-4d27-8d62-7b3a2f49d6a1")  # 与项目一致

STATUS_RANK = {
    "Designer Reviewed": 1.00,
    "CQA_Done": 0.97,
    "Done_LQA edited": 0.94,
    "Done": 0.90,
}

# --- 日志 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ingest")


def status_rank(status: str | None) -> float:
    if not status:
        return 0.85
    return STATUS_RANK.get(status, 0.85)


def to_sparse(text: str) -> models.SparseVector:
    """jieba 分词 + 词频 → SparseVector（indices=token hash, values=count）。"""
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    if not tokens:
        return models.SparseVector(indices=[0], values=[0.0])
    counts = Counter(tokens)
    indices = []
    values = []
    for tok, cnt in counts.items():
        # 与项目一致：哈希到 31 位正整数范围
        idx = abs(hash(tok)) % (2 ** 31)
        indices.append(idx)
        values.append(float(cnt))
    return models.SparseVector(indices=indices, values=values)


def make_id(source: str, target: str, status: str | None) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{source}\x1f{target}\x1f{status or ''}"))


async def load_corpus() -> list[dict]:
    items = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


async def reset_collection(qdrant: AsyncQdrantClient):
    """删除并重建 collection，与项目原配置一致。"""
    try:
        await qdrant.delete_collection(COLLECTION)
        log.info("已删除旧 collection %s", COLLECTION)
    except Exception as e:
        log.info("删除 collection (可能不存在): %s", e)

    await qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            _VEC_DENSE: models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        },
        sparse_vectors_config={
            _VEC_SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
        on_disk_payload=True,
        hnsw_config=models.HnswConfigDiff(on_disk=True),
    )
    log.info("已创建新 collection %s (dense=%d 维 cosine + sparse BM25 IDF)", COLLECTION, EMBEDDING_DIM)


async def embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """带重试的批量 embedding。"""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIM,
            )
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [list(d.embedding) for d in ordered]
        except RateLimitError as e:
            last_exc = e
            wait = min(2 ** attempt, 32)
            log.warning("rate limited, sleep %ss (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
            await asyncio.sleep(wait)
        except (APITimeoutError, APIError, APIStatusError) as e:
            last_exc = e
            log.warning("embedding error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, str(e)[:200])
            await asyncio.sleep(2)
    raise RuntimeError(f"embed_batch 重试 {MAX_RETRIES} 次仍失败: {last_exc}")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed_batch_ids": [], "total_indexed": 0, "started_at": None}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_failed(batch_id: int, items: list[dict], reason: str):
    with open(FAILED_PATH, "a", encoding="utf-8") as f:
        for it in items:
            rec = {"batch_id": batch_id, "reason": reason, **it}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def process_batch(
    batch_id: int,
    batch: list[dict],
    client: AsyncOpenAI,
    qdrant: AsyncQdrantClient,
    sem: asyncio.Semaphore,
    state: dict,
    state_lock: asyncio.Lock,
    counter: dict,
):
    async with sem:
        sources = [it["source"] for it in batch]
        try:
            t0 = time.monotonic()
            vectors = await embed_batch(client, sources)
            t_embed = time.monotonic() - t0

            points = [
                models.PointStruct(
                    id=make_id(it["source"], it["target"], it.get("status")),
                    vector={
                        _VEC_DENSE: vec,
                        _VEC_SPARSE: to_sparse(it["source"]),
                    },
                    payload={
                        "source": it["source"],
                        "target": it["target"],
                        **({"status": it["status"], "priority_rank": status_rank(it.get("status"))}
                           if it.get("status") else {}),
                        **({"source_db": it["source_db"]} if it.get("source_db") else {}),
                        **({"orig_id": it["id"]} if it.get("id") else {}),
                    },
                )
                for it, vec in zip(batch, vectors)
            ]
            await qdrant.upsert(collection_name=COLLECTION, points=points, wait=False)

            async with state_lock:
                state["completed_batch_ids"].append(batch_id)
                state["total_indexed"] += len(batch)
                counter["done_batches"] += 1
                counter["done_items"] += len(batch)
                if counter["done_batches"] % FLUSH_EVERY == 0:
                    save_state(state)
                    progress_pct = state["total_indexed"] / counter["total_items"] * 100
                    eta = (time.monotonic() - counter["start_time"]) / counter["done_batches"] * (counter["total_batches"] - counter["done_batches"])
                    log.info(
                        "进度 %d/%d 批 (%d items, %.1f%%) 本批 embed=%.1fs ETA=%.1f分钟",
                        counter["done_batches"], counter["total_batches"],
                        state["total_indexed"], progress_pct,
                        t_embed, eta / 60,
                    )
        except Exception as e:
            log.error("批次 %d 失败 (%d items): %s", batch_id, len(batch), str(e)[:200])
            append_failed(batch_id, batch, str(e)[:200])


async def main():
    # 加载数据
    log.info("加载清洗后语料: %s", CORPUS_PATH)
    items = await load_corpus()
    log.info("总条数: %d", len(items))

    # Qdrant 客户端 + 重建 collection
    qdrant = AsyncQdrantClient(url=QDRANT_URL, timeout=60)
    await reset_collection(qdrant)

    # OpenAI 客户端
    client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=60,
        max_retries=0,
    )

    # 切批
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    log.info("切分为 %d 批 (batch_size=%d, concurrent=%d)", len(batches), BATCH_SIZE, MAX_CONCURRENT)

    # 状态加载
    state = load_state()
    completed = set(state["completed_batch_ids"])
    if completed:
        log.info("断点续传：已完成 %d 批", len(completed))
    state["started_at"] = state["started_at"] or time.time()

    counter = {
        "total_items": len(items),
        "total_batches": len(batches),
        "done_batches": len(completed),
        "done_items": state["total_indexed"],
        "start_time": time.monotonic(),
    }

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    state_lock = asyncio.Lock()

    coros = [
        process_batch(i, batch, client, qdrant, sem, state, state_lock, counter)
        for i, batch in enumerate(batches)
        if i not in completed
    ]

    log.info("启动 %d 个待处理批次", len(coros))
    t_all = time.monotonic()
    await asyncio.gather(*coros)
    total_time = time.monotonic() - t_all

    save_state(state)
    await client.close()
    await qdrant.close()

    log.info("=" * 60)
    log.info("完成！总耗时 %.1f 分钟", total_time / 60)
    log.info("入库总数: %d / %d", state["total_indexed"], len(items))
    if FAILED_PATH.exists():
        with open(FAILED_PATH, "r", encoding="utf-8") as f:
            failed_n = sum(1 for _ in f)
        log.info("失败条数: %d（详见 %s）", failed_n, FAILED_PATH)


if __name__ == "__main__":
    asyncio.run(main())
