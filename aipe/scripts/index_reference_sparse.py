"""Index monolingual reference chunks in a local sparse-only Qdrant collection."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings
from app.services.rag_service import _to_sparse


ID_NAMESPACE = uuid.UUID("9ca7a9a3-6650-41b6-b99c-7cbe8de22ad8")
VECTOR_NAME = "sparse"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def _collection_exists(client: AsyncQdrantClient, collection: str) -> bool:
    try:
        await client.get_collection(collection)
        return True
    except Exception:
        return False


async def _create(client: AsyncQdrantClient, collection: str) -> None:
    await client.create_collection(
        collection_name=collection,
        vectors_config={},
        sparse_vectors_config={
            VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=True),
            )
        },
        on_disk_payload=True,
    )


async def command_index(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60.0)
    try:
        exists = await _collection_exists(client, args.collection)
        if args.reset and exists:
            if not args.collection.startswith("lom_reference_"):
                raise SystemExit("refusing to reset a collection outside the lom_reference_ namespace")
            await client.delete_collection(args.collection)
            exists = False
        if not exists:
            await _create(client, args.collection)

        chunks = {item["chunk_id"]: item for item in _read_jsonl(args.chunks)}
        aliases = _read_jsonl(args.alias_map)
        if set(chunks) != {item["chunk_id"] for item in aliases}:
            raise SystemExit("chunk and alias-map IDs differ")

        total = len(aliases)
        indexed = 0
        for start in range(0, total, args.batch_size):
            batch = aliases[start:start + args.batch_size]
            points: list[models.PointStruct] = []
            for item in batch:
                chunk = chunks[item["chunk_id"]]
                search_text = str(item["sparse_search_text"])
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(ID_NAMESPACE, item["chunk_id"])),
                        vector={VECTOR_NAME: _to_sparse(search_text)},
                        payload={
                            "entry_kind": "client_reference_corpus",
                            "source_id": "source.lom.lord_txt.v1",
                            "chunk_id": item["chunk_id"],
                            "chapter_number": item["chapter_number"],
                            "chapter_title": chunk["chapter_title"],
                            "line_start": item["line_start"],
                            "line_end": item["line_end"],
                            "text": chunk["text"],
                            "source_aliases": item["source_aliases"],
                            "client_terms": item["client_terms"],
                            "sha256": chunk["sha256"],
                        },
                    )
                )
            await client.upsert(collection_name=args.collection, points=points, wait=True)
            indexed += len(points)
            print(f"indexed={indexed}/{total}", flush=True)
        count = await client.count(collection_name=args.collection, exact=True)
        print(json.dumps({"collection": args.collection, "expected": total, "points": count.count}, ensure_ascii=False))
        if count.count != total:
            raise SystemExit("Qdrant point count mismatch")
    finally:
        await client.close()


async def command_search(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60.0)
    try:
        response = await client.query_points(
            collection_name=args.collection,
            query=_to_sparse(args.query),
            using=VECTOR_NAME,
            limit=args.top_k,
            with_payload=True,
        )
        results = [
            {
                "score": float(point.score),
                "chunk_id": (point.payload or {}).get("chunk_id"),
                "chapter_number": (point.payload or {}).get("chapter_number"),
                "chapter_title": (point.payload or {}).get("chapter_title"),
                "line_start": (point.payload or {}).get("line_start"),
                "line_end": (point.payload or {}).get("line_end"),
                "source_aliases": (point.payload or {}).get("source_aliases"),
                "text_preview": str((point.payload or {}).get("text", ""))[:500],
            }
            for point in response.points
        ]
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser("index")
    index.add_argument("--chunks", type=Path, required=True)
    index.add_argument("--alias-map", type=Path, required=True)
    index.add_argument("--collection", required=True)
    index.add_argument("--batch-size", type=int, default=100)
    index.add_argument("--reset", action="store_true")
    index.set_defaults(func=command_index)
    search = subparsers.add_parser("search")
    search.add_argument("--collection", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.set_defaults(func=command_search)
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
