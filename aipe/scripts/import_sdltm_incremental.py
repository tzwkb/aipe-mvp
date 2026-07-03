"""Incrementally import SDLTM files into the RAG Qdrant collection.

The important property is that existing TM pairs are reconciled before any
embedding call is made. Exact or better existing entries are skipped; lower
quality existing entries are upgraded by reusing their dense vector.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

STATUS_RANK: dict[str, int] = {
    "Designer Reviewed": 1,
    "CQA_Done": 2,
    "Done_LQA edited": 3,
    "Done": 4,
}

_ID_NAMESPACE = uuid.UUID("0a8c1d8c-5e22-4f7a-9a5e-9c6f4a2d3b11")
_DEFAULT_PROGRESS = Path("data/progress/sdltm_incremental_import_20260629.json")
_DEFAULT_BATCH_SIZE = 30
_DEFAULT_EMBED_CONCURRENCY = 2
_DEFAULT_SCROLL_LIMIT = 2048

logger = logging.getLogger("import_sdltm_incremental")


@dataclass(frozen=True)
class Candidate:
    source: str
    target: str
    status: str
    source_file: str


@dataclass(frozen=True)
class ExistingPoint:
    point_id: Any
    status: str | None


@dataclass
class ImportPlan:
    to_embed: list[Candidate]
    to_clone: list[tuple[Candidate, Any, list[Any]]]
    skipped_existing: int

    @property
    def embedding_sources(self) -> set[str]:
        return {c.source for c in self.to_embed}


def status_for_path(path: str | Path, default_status: str | None = None) -> str:
    """Map an SDLTM library file/name to the RAG quality status."""
    name = Path(path).name.lower()
    if "designer" in name:
        return "Designer Reviewed"
    if "cqa" in name:
        return "CQA_Done"
    if "lqa" in name:
        return "Done_LQA edited"
    if "gt" in name or "st" in name:
        return "Done"
    if default_status:
        if default_status not in STATUS_RANK:
            raise ValueError(f"不支持的默认 TM 分级: {default_status}; allowed={list(STATUS_RANK)}")
        return default_status
    raise ValueError(f"无法从文件名判断 TM 分级: {path}")


def status_rank(status: str | None) -> int:
    return STATUS_RANK.get((status or "").strip(), 99)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def extract_segment_text(segment_xml: str) -> str:
    """Extract visible segment text from SDLTM XML.

    SDLTM stores segment text and inline tag metadata in the same XML. Only
    ``Text`` element values are user-visible translation content; tag metadata
    such as Anchor/TagID must not enter RAG text.
    """
    try:
        root = ET.fromstring(segment_xml)
    except ET.ParseError as exc:
        raise ValueError(f"SDLTM segment XML 解析失败: {exc}") from exc

    parts: list[str] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "Text":
            continue
        value_children = [c for c in list(elem) if _local_name(c.tag) == "Value"]
        if value_children:
            parts.extend("".join(v.itertext()) for v in value_children)
        else:
            parts.append("".join(elem.itertext()))
    return _clean_text("".join(parts))


def point_id_for(candidate: Candidate) -> str:
    raw = f"{candidate.source}\x1f{candidate.target}\x1f{candidate.status or ''}"
    return str(uuid.uuid5(_ID_NAMESPACE, raw))


def best_candidate_by_pair(
    candidates: Iterable[Candidate],
) -> tuple[dict[tuple[str, str], Candidate], int]:
    """Deduplicate incoming TM by pair and keep the highest quality status."""
    selected: dict[tuple[str, str], Candidate] = {}
    skipped = 0
    for candidate in candidates:
        key = (candidate.source, candidate.target)
        current = selected.get(key)
        if current is None:
            selected[key] = candidate
            continue
        if status_rank(candidate.status) < status_rank(current.status):
            selected[key] = candidate
        skipped += 1
    return selected, skipped


def plan_incremental_import(
    candidates_by_pair: dict[tuple[str, str], Candidate],
    existing_by_pair: dict[tuple[str, str], list[ExistingPoint]],
) -> ImportPlan:
    """Plan import actions without requiring new embeddings for existing pairs."""
    to_embed: list[Candidate] = []
    to_clone: list[tuple[Candidate, Any, list[Any]]] = []
    skipped_existing = 0

    for key, candidate in candidates_by_pair.items():
        existing_points = existing_by_pair.get(key, [])
        if not existing_points:
            to_embed.append(candidate)
            continue

        candidate_rank = status_rank(candidate.status)
        best_existing_rank = min(status_rank(p.status) for p in existing_points)
        if best_existing_rank <= candidate_rank:
            skipped_existing += 1
            continue

        reusable = min(existing_points, key=lambda p: status_rank(p.status))
        to_clone.append((candidate, reusable.point_id, [p.point_id for p in existing_points]))

    return ImportPlan(to_embed=to_embed, to_clone=to_clone, skipped_existing=skipped_existing)


def iter_sdltm_candidates(path: Path, default_status: str | None = None) -> Iterable[Candidate]:
    status = status_for_path(path, default_status=default_status)
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select source_segment, target_segment
            from translation_units
            where source_segment is not null
              and trim(source_segment) <> ''
              and target_segment is not null
              and trim(target_segment) <> ''
            """
        )
        for row in rows:
            source = extract_segment_text(row["source_segment"])
            target = extract_segment_text(row["target_segment"])
            if source and target:
                yield Candidate(source=source, target=target, status=status, source_file=path.name)


def load_candidates(
    tm_dir: Path,
    limit: int = 0,
    default_status: str | None = None,
) -> tuple[list[Candidate], dict[str, int]]:
    paths = sorted(tm_dir.glob("*.sdltm"))
    if not paths:
        raise SystemExit(f"未找到 SDLTM 文件: {tm_dir}")

    candidates: list[Candidate] = []
    counts: dict[str, int] = {}
    for path in paths:
        count = 0
        for candidate in iter_sdltm_candidates(path, default_status=default_status):
            candidates.append(candidate)
            count += 1
            if limit and len(candidates) >= limit:
                counts[path.name] = count
                return candidates, counts
        counts[path.name] = count
    return candidates, counts


async def scan_existing_pairs(client: Any, collection: str, limit: int) -> dict[tuple[str, str], list[ExistingPoint]]:
    existing: dict[tuple[str, str], list[ExistingPoint]] = defaultdict(list)
    offset = None
    scanned = 0
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=["source", "target", "status"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            source = _clean_text(str(payload.get("source") or ""))
            target = _clean_text(str(payload.get("target") or ""))
            if not source or not target:
                continue
            existing[(source, target)].append(ExistingPoint(point.id, payload.get("status") or None))
        scanned += len(points)
        if scanned and scanned % (limit * 10) == 0:
            logger.info("已扫描 Qdrant payload: %d", scanned)
        if offset is None:
            break
    logger.info("Qdrant payload 扫描完成: points=%d unique_pairs=%d", scanned, len(existing))
    return dict(existing)


def _summary_dict(
    *,
    candidates_total: int,
    incoming_unique: int,
    incoming_duplicates_skipped: int,
    per_file_counts: dict[str, int],
    plan: ImportPlan,
    collection: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "collection": collection,
        "dry_run": dry_run,
        "candidates_total": candidates_total,
        "incoming_unique_pairs": incoming_unique,
        "incoming_duplicates_skipped": incoming_duplicates_skipped,
        "skipped_existing_better_or_equal": plan.skipped_existing,
        "new_pairs_need_embedding": len(plan.to_embed),
        "unique_sources_need_embedding": len(plan.embedding_sources),
        "pairs_upgraded_by_vector_reuse": len(plan.to_clone),
        "per_file_counts": per_file_counts,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _point_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "source": candidate.source,
        "target": candidate.target,
        "status": candidate.status,
        "priority_rank": status_rank(candidate.status),
        "source_tm": candidate.source_file,
    }


async def _upsert_clone_upgrades(client: Any, collection: str, clone_actions: list[tuple[Candidate, Any, list[Any]]]) -> int:
    if not clone_actions:
        return 0

    from qdrant_client import models
    from app.services.rag_service import _VEC_DENSE, _VEC_SPARSE, _to_sparse

    upgraded = 0
    for chunk in _iter_batches(clone_actions, 128):
        reuse_ids = [reuse_id for _, reuse_id, _ in chunk]
        records = await client.retrieve(
            collection_name=collection,
            ids=reuse_ids,
            with_vectors=True,
            with_payload=False,
        )
        vectors_by_id = {r.id: r.vector for r in records}
        points = []
        delete_ids: list[Any] = []
        for candidate, reuse_id, obsolete_ids in chunk:
            vector = vectors_by_id.get(reuse_id)
            if not isinstance(vector, dict) or _VEC_DENSE not in vector:
                raise RuntimeError(f"无法复用 existing vector: point_id={reuse_id}")
            points.append(
                models.PointStruct(
                    id=point_id_for(candidate),
                    vector={
                        _VEC_DENSE: vector[_VEC_DENSE],
                        _VEC_SPARSE: _to_sparse(candidate.source),
                    },
                    payload=_point_payload(candidate),
                )
            )
            delete_ids.extend(obsolete_ids)
        await client.upsert(collection_name=collection, points=points, wait=True)
        await client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(dict.fromkeys(delete_ids))),
            wait=True,
        )
        upgraded += len(points)
        logger.info("复用向量升级完成: %d/%d", upgraded, len(clone_actions))
    return upgraded


async def _embed_and_upsert_new(
    client: Any,
    collection: str,
    candidates: list[Candidate],
    batch_size: int,
    concurrency: int,
    progress_path: Path,
    summary: dict[str, Any],
) -> int:
    if not candidates:
        return 0

    from qdrant_client import models
    from app.config import get_settings
    from app.dependencies import get_llm_service
    from app.services.rag_service import _VEC_DENSE, _VEC_SPARSE, _to_sparse

    settings = get_settings()
    llm_svc = get_llm_service()
    source_to_candidates: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        source_to_candidates[candidate.source].append(candidate)
    sources = list(source_to_candidates)
    chunks = list(_iter_batches(sources, batch_size))
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    inserted = 0

    async def worker(chunk: list[str]) -> None:
        nonlocal inserted
        async with sem:
            vectors = await llm_svc.embed_batch(chunk)
            points = []
            for source, vector in zip(chunk, vectors):
                for candidate in source_to_candidates[source]:
                    points.append(
                        models.PointStruct(
                            id=point_id_for(candidate),
                            vector={
                                _VEC_DENSE: vector,
                                _VEC_SPARSE: _to_sparse(candidate.source),
                            },
                            payload=_point_payload(candidate),
                        )
                    )
            await client.upsert(collection_name=collection, points=points, wait=True)
        async with lock:
            inserted += len(points)
            summary["inserted_new_pairs"] = inserted
            summary["embedding_batches_completed"] = summary.get("embedding_batches_completed", 0) + 1
            summary["embedding_batches_total"] = len(chunks)
            summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _save_json(progress_path, summary)
            logger.info(
                "新增 embedding 入库进度: pairs=%d/%d batches=%d/%d",
                inserted,
                len(candidates),
                summary["embedding_batches_completed"],
                len(chunks),
            )

    await asyncio.gather(*(worker(chunk) for chunk in chunks))
    if hasattr(llm_svc, "aclose"):
        await llm_svc.aclose()
    return inserted


async def _ensure_collection_for_import(client: Any, collection: str, settings: Any, *, commit: bool) -> bool:
    from qdrant_client import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    from app.services.llm_service import LLMService
    from app.services.rag_service import _VEC_DENSE, _VEC_SPARSE

    try:
        await client.get_collection(collection)
        return True
    except (UnexpectedResponse, ValueError):
        if not commit:
            logger.info("collection 不存在，dry-run 按空库规划: %s", collection)
            return False

    llm_svc = LLMService(settings)
    try:
        probe = await llm_svc.embed("probe")
    finally:
        if hasattr(llm_svc, "aclose"):
            await llm_svc.aclose()
    vector_size = len(probe)
    await client.create_collection(
        collection_name=collection,
        vectors_config={
            _VEC_DENSE: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        },
        sparse_vectors_config={
            _VEC_SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
        hnsw_config=models.HnswConfigDiff(on_disk=True),
        on_disk_payload=True,
    )
    logger.info("collection 已创建: %s dim=%d", collection, vector_size)
    return True


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from qdrant_client import AsyncQdrantClient
    from app.config import get_settings

    settings = get_settings()
    collection = args.collection or settings.qdrant_collection
    candidates, per_file_counts = load_candidates(
        args.tm_dir,
        limit=args.limit,
        default_status=args.default_status,
    )
    selected, incoming_duplicates = best_candidate_by_pair(candidates)

    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60.0)
    try:
        collection_exists = await _ensure_collection_for_import(client, collection, settings, commit=args.commit)
        existing = await scan_existing_pairs(client, collection, args.scroll_limit) if collection_exists else {}
        plan = plan_incremental_import(selected, existing)
        summary = _summary_dict(
            candidates_total=len(candidates),
            incoming_unique=len(selected),
            incoming_duplicates_skipped=incoming_duplicates,
            per_file_counts=per_file_counts,
            plan=plan,
            collection=collection,
            dry_run=not args.commit,
        )
        _save_json(args.progress, summary)
        logger.info("导入计划: %s", json.dumps(summary, ensure_ascii=False))

        if not args.commit:
            return summary

        upgraded = await _upsert_clone_upgrades(client, collection, plan.to_clone)
        summary["upgraded_by_vector_reuse"] = upgraded
        _save_json(args.progress, summary)
        inserted = await _embed_and_upsert_new(
            client=client,
            collection=collection,
            candidates=plan.to_embed,
            batch_size=args.batch_size,
            concurrency=args.embedding_concurrency,
            progress_path=args.progress,
            summary=summary,
        )
        summary["inserted_new_pairs"] = inserted
        summary["dry_run"] = False
        summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_json(args.progress, summary)
        return summary
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally import SDLTM TM files into Qdrant RAG.")
    parser.add_argument("tm_dir", type=Path, help="Directory containing .sdltm files")
    parser.add_argument("--collection", default=None, help="Target Qdrant collection; default uses .env")
    parser.add_argument("--progress", type=Path, default=_DEFAULT_PROGRESS, help="Progress/summary JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N candidates for testing")
    parser.add_argument(
        "--default-status",
        choices=sorted(STATUS_RANK),
        default=None,
        help="Fallback quality status for SDLTM files whose names do not encode Designer/CQA/LQA/GT/ST.",
    )
    parser.add_argument("--scroll-limit", type=int, default=_DEFAULT_SCROLL_LIMIT, help="Qdrant scroll page size")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="Unique source texts per embedding batch")
    parser.add_argument(
        "--embedding-concurrency",
        type=int,
        default=_DEFAULT_EMBED_CONCURRENCY,
        help="Concurrent embedding batches",
    )
    parser.add_argument("--commit", action="store_true", help="Actually write to Qdrant and call embedding API")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
