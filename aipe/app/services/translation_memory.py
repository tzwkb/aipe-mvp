"""Translation-memory exact-match resolution for translation pipelines."""

from __future__ import annotations

import logging

from app.schemas.rag import RAGSearchResult
from app.schemas.translate import TranslationResult
from app.services.rag_service import RAGService
from app.services.style_guide_service import ContentType

logger = logging.getLogger(__name__)


class TMExactMatchResolver:
    """Resolve exact TM hits in one batched payload query."""

    def __init__(self, rag_svc: RAGService):
        self.rag_svc = rag_svc

    async def resolve_many(
        self,
        sources: list[str],
        *,
        collection: str | None,
        content_type: ContentType | None,
    ) -> list[TranslationResult | None]:
        if not sources:
            return []
        try:
            matches_by_source = await self.rag_svc.find_exact_source_matches_many(
                sources,
                collection=collection,
                top_k=1,
            )
        except Exception as exc:
            logger.warning("TM 精确匹配查询失败，降级为常规翻译: %s", exc)
            return [None] * len(sources)

        return [
            _exact_result(source, matches_by_source.get(source, []), content_type)
            for source in sources
        ]


def merge_exact_results(
    exact_results: list[TranslationResult | None],
    generated_results: list[TranslationResult],
) -> list[TranslationResult]:
    """Overlay exact TM results without changing source positions or context size."""
    if len(exact_results) != len(generated_results):
        raise ValueError(
            "TM 精确匹配结果与生成结果长度不一致: "
            f"{len(exact_results)} != {len(generated_results)}"
        )
    return [
        exact if exact is not None else generated
        for exact, generated in zip(exact_results, generated_results)
    ]


def format_locked_tm_section(exact_results: list[TranslationResult | None]) -> str | None:
    """Render exact hits as immutable context for group/dialog prompts."""
    lines = [
        f'{index}. 「{result.source}」 → "{result.translation}"'
        for index, result in enumerate(exact_results, 1)
        if result is not None
    ]
    if not lines:
        return None
    return (
        "## 已锁定 TM 译文（必须原样保留；同时作为未命中句的上下文）\n"
        "以下编号已经由精确 TM 命中。仍需输出全部编号；这些行的译文必须原样复制，"
        "并用它们帮助理解未命中句的上下文：\n"
        + "\n".join(lines)
    )


def _exact_result(
    source: str,
    matches: list[RAGSearchResult],
    content_type: ContentType | None,
) -> TranslationResult | None:
    if not matches:
        return None
    match = matches[0]
    reference = {
        "source": match.source,
        "target": match.target,
        "score": round(match.score, 4),
    }
    if match.status:
        reference["status"] = match.status
    return TranslationResult(
        source=source,
        translation=match.target,
        translation_reason=(
            "TM_EXACT_MATCH: source 与 TM 语料完全一致，直接采用最高优先级 TM target，跳过 AI 翻译。"
        ),
        status="success",
        content_type=content_type.value if content_type is not None else None,
        rag_references=[reference],
        tm_exact_match_used=True,
        tm_exact_match_source=match.source,
        tm_exact_match_target=match.target,
        tm_exact_match_status=match.status,
        tm_exact_match_score=match.score,
    )


__all__ = ["TMExactMatchResolver", "format_locked_tm_section", "merge_exact_results"]
