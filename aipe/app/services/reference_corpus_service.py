"""Local sparse retrieval for monolingual project reference corpora."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.services.rag_service import _to_sparse

VECTOR_NAME = "sparse"


@dataclass(frozen=True)
class ReferenceCorpusHit:
    score: float
    chunk_id: str
    chapter_number: int | None
    chapter_title: str
    line_start: int | None
    line_end: int | None
    text: str
    source_aliases: tuple[str, ...]
    client_terms: tuple[dict[str, Any], ...]


class ReferenceCorpusService:
    """Query a dedicated sparse-only Qdrant collection without embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client_instance: AsyncQdrantClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    @property
    def _client(self) -> AsyncQdrantClient:
        loop = asyncio.get_running_loop()
        if self._client_instance is None or self._client_loop is not loop:
            self._client_instance = AsyncQdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=15.0,
            )
            self._client_loop = loop
        return self._client_instance

    async def search(
        self,
        query: str,
        *,
        collection: str,
        required_source_aliases: Iterable[str] = (),
        chapter_numbers: Iterable[int] = (),
        top_k: int = 3,
    ) -> list[ReferenceCorpusHit]:
        query = str(query or "").strip()
        collection = str(collection or "").strip()
        required = {
            _normalize_alias(alias)
            for alias in required_source_aliases
            if _normalize_alias(alias)
        }
        if not query or not collection or not required or top_k <= 0:
            return []

        chapters = sorted(
            {
                number
                for number in chapter_numbers
                if isinstance(number, int) and number > 0
            }
        )
        query_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="chapter_number",
                        match=models.MatchAny(any=chapters),
                    )
                ]
            )
            if chapters
            else None
        )

        response = await self._client.query_points(
            collection_name=collection,
            query=_to_sparse(query),
            using=VECTOR_NAME,
            limit=max(top_k * 6, 12),
            with_payload=True,
            query_filter=query_filter,
        )
        hits: list[ReferenceCorpusHit] = []
        for point in response.points:
            payload = point.payload or {}
            aliases = tuple(
                str(alias).strip()
                for alias in payload.get("source_aliases") or []
                if str(alias).strip()
            )
            normalized_aliases = {_normalize_alias(alias) for alias in aliases}
            if not required.intersection(normalized_aliases):
                continue
            raw_terms = payload.get("client_terms") or []
            client_terms = tuple(term for term in raw_terms if isinstance(term, dict))
            hits.append(
                ReferenceCorpusHit(
                    score=float(point.score),
                    chunk_id=str(payload.get("chunk_id") or ""),
                    chapter_number=_optional_int(payload.get("chapter_number")),
                    chapter_title=str(payload.get("chapter_title") or "").strip(),
                    line_start=_optional_int(payload.get("line_start")),
                    line_end=_optional_int(payload.get("line_end")),
                    text=str(payload.get("text") or "").strip(),
                    source_aliases=aliases,
                    client_terms=client_terms,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    @staticmethod
    def render(
        hits: list[ReferenceCorpusHit],
        *,
        target_anchors: Iterable[str] = (),
        max_chars: int = 3600,
        excerpt_chars: int = 900,
    ) -> str:
        if not hits or max_chars <= 0:
            return ""
        anchors = [
            str(anchor).strip() for anchor in target_anchors if str(anchor).strip()
        ]
        lines = [
            "项目英文参考语料（Lord.txt，本地稀疏检索；仅供既有英文命名、搭配和语域参考，不代表与当前原文语义等同）："
        ]
        for hit in hits:
            chapter = (
                f"Chapter {hit.chapter_number}"
                if hit.chapter_number is not None
                else "Preamble"
            )
            if hit.chapter_title:
                chapter += f": {hit.chapter_title}"
            coordinate = _coordinate(hit.line_start, hit.line_end)
            excerpt = _extract_excerpt(hit.text, anchors, excerpt_chars)
            lines.append(f"- {chapter}; Lord.txt {coordinate}; score={hit.score:.4f}")
            if excerpt:
                lines.append(f"  {excerpt}")
        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            return (
                rendered[: max_chars - 31].rstrip() + "\n[reference context truncated]"
            )
        return rendered


def _extract_excerpt(text: str, anchors: list[str], limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact or limit <= 0:
        return ""
    positions = [
        position
        for anchor in anchors
        if anchor
        for position in [compact.casefold().find(anchor.casefold())]
        if position >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(compact), start + limit)
    start = max(0, end - limit)
    excerpt = compact[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(compact):
        excerpt += "…"
    return excerpt


def _normalize_alias(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coordinate(start: int | None, end: int | None) -> str:
    if start is None:
        return "(unknown lines)"
    if end is None or end == start:
        return f"L{start}"
    return f"L{start}-L{end}"


__all__ = ["ReferenceCorpusHit", "ReferenceCorpusService"]
