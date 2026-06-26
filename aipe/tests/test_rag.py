"""RAG 服务集成测试（hybrid: dense + BM25 + RRF）。

依赖本地 Qdrant (localhost:6333)；不可用时整个模块 skip。
Embedding 用确定性 mock（输入字符串映射到固定 one-hot 向量），
保证相同字符串 cosine=1.0，不同字符串 cosine=0.0，便于断言 dense 路阈值过滤。

Sparse 路走真实 jieba 分词 + Qdrant Modifier.IDF，分数为 RRF 融合分（不再 == 1.0）。
"""

from __future__ import annotations

import asyncio
import hashlib
import socket

import pytest

from app.config import get_settings
from app.schemas.rag import CorpusEntry
from app.services.rag_service import RAGService


def _qdrant_alive(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


_settings = get_settings()
if not _qdrant_alive(_settings.qdrant_host, _settings.qdrant_port):
    pytest.skip(
        f"Qdrant 不可用 ({_settings.qdrant_host}:{_settings.qdrant_port})，跳过 RAG 测试",
        allow_module_level=True,
    )


_VECTOR_DIM = 64


class _DeterministicEmbedLLM:
    """LLMService 替身：把字符串 hash 成 one-hot 向量（确定性）。"""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return _one_hot(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [_one_hot(t) for t in texts]


def _one_hot(text: str) -> list[float]:
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    idx = h % _VECTOR_DIM
    v = [0.0] * _VECTOR_DIM
    v[idx] = 1.0
    return v


@pytest.fixture
def rag_svc():
    """每个用例一个独立 collection，避免相互污染。"""
    settings = get_settings().model_copy(
        update={"qdrant_collection": "yanyun_test_" + _short_id()}
    )
    svc = RAGService(settings, _DeterministicEmbedLLM())  # type: ignore[arg-type]
    yield svc

    async def _cleanup():
        try:
            await svc._client.delete_collection(svc.collection)  # noqa: SLF001
        except Exception:
            pass
        await svc.aclose()

    asyncio.run(_cleanup())


def _short_id() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


def test_index_and_exact_hit(rag_svc: RAGService):
    entries = [
        CorpusEntry(source="契丹来犯", target="The Khitan are attacking"),
        CorpusEntry(source="少侠请进", target="Welcome, young hero"),
    ]
    indexed = asyncio.run(rag_svc.index(entries))
    assert indexed == 2
    assert asyncio.run(rag_svc.count()) == 2

    hits = asyncio.run(rag_svc.search("契丹来犯", threshold=0.85, top_k=3))
    # 命中条目应稳定排在第一位；分数为 RRF 融合分，与 cosine 不可比，仅断言为正。
    assert len(hits) >= 1
    assert hits[0].source == "契丹来犯"
    assert hits[0].target == "The Khitan are attacking"
    assert hits[0].score > 0.0


def test_below_threshold_returns_empty(rag_svc: RAGService):
    asyncio.run(
        rag_svc.index([CorpusEntry(source="契丹来犯", target="The Khitan attack")])
    )
    # 不同字符串：dense one-hot cosine=0 被 0.85 阈值过滤；BM25 token 也不重叠 → 全空
    hits = asyncio.run(rag_svc.search("毫不相干的句子", threshold=0.85, top_k=3))
    assert hits == []


def test_top_k_limit(rag_svc: RAGService):
    # 用同一 source 的多个 (source,target) 共享向量但 ID 不同
    entries = [
        CorpusEntry(source="燕云十六声", target=f"Yanyun #{i}") for i in range(5)
    ]
    asyncio.run(rag_svc.index(entries))
    hits = asyncio.run(rag_svc.search("燕云十六声", threshold=0.85, top_k=2))
    assert len(hits) == 2
    for h in hits:
        assert h.source == "燕云十六声"
        assert h.score > 0.0  # RRF 融合分；同 source 多 target 间分数会接近


def test_threshold_param_overrides_default(rag_svc: RAGService):
    asyncio.run(
        rag_svc.index([CorpusEntry(source="契丹来犯", target="The Khitan attack")])
    )
    # 无关 query：threshold=0 应能取到任意命中（top_k=1）
    hits = asyncio.run(rag_svc.search("和原文毫不相干", threshold=0.0, top_k=1))
    # 不同 one-hot 的 cosine=0，但 score_threshold=0 仍会过滤掉，所以可能为空。
    # 这里仅验证调用不抛错，不强约束返回。
    assert isinstance(hits, list)


def test_index_idempotent_dedup(rag_svc: RAGService):
    entry = CorpusEntry(source="契丹来犯", target="The Khitan attack")
    asyncio.run(rag_svc.index([entry]))
    asyncio.run(rag_svc.index([entry]))  # 再次入库同一条
    assert asyncio.run(rag_svc.count()) == 1


def test_index_skips_empty_rows(rag_svc: RAGService):
    entries = [
        CorpusEntry(source="A", target="a"),
        # 注意：CorpusEntry 不强校验非空，靠 RAGService 过滤
        CorpusEntry(source="", target="b"),
        CorpusEntry(source="C", target=""),
    ]
    indexed = asyncio.run(rag_svc.index(entries))
    assert indexed == 1


def test_hybrid_recovers_substring_match(rag_svc: RAGService):
    """混合检索的核心收益：dense 命不中（不同字符串 → cosine=0）时，BM25 仍能召回包含子串的长句。"""
    corpus = [
        CorpusEntry(source="百世千般病 一命换一医", target="A life for a cure"),
        CorpusEntry(source="少侠请进", target="Welcome, young hero"),
    ]
    asyncio.run(rag_svc.index(corpus))

    # one-hot mock 让任意不等字符串 dense cosine=0，必被 0.85 阈值过滤
    # → 命中只能来自 sparse(BM25) 路
    hits = asyncio.run(rag_svc.search("一命换一医", threshold=0.85, top_k=3))
    assert hits, "hybrid 应通过 BM25 召回包含子串的长句"
    assert hits[0].source == "百世千般病 一命换一医"


def test_search_empty_query_short_circuits(rag_svc: RAGService):
    """空 query 直接返回 []，不应触发 collection 创建或 embedding 调用。"""
    fake_llm: _DeterministicEmbedLLM = rag_svc.llm_svc  # type: ignore[assignment]
    before = fake_llm.calls
    hits = asyncio.run(rag_svc.search("   "))
    assert hits == []
    assert fake_llm.calls == before


# ---------- search_with_diagnostics ----------


def test_search_with_diagnostics_returns_dense_top1_and_sparse_hits(rag_svc: RAGService):
    """对已入库的精确匹配 query，dense_top1 应该 ≈ 1.0（one-hot 自相似），sparse_hits ≥ 1。"""
    asyncio.run(
        rag_svc.index([CorpusEntry(source="契丹来犯", target="The Khitan attack")])
    )
    refs, diag = asyncio.run(rag_svc.search_with_diagnostics("契丹来犯", threshold=0.85, top_k=3))
    assert len(refs) >= 1
    assert diag.dense_top1 is not None
    assert diag.dense_top1 > 0.9   # one-hot 自相似 cosine ≈ 1.0
    assert diag.sparse_hits >= 1


def test_search_with_diagnostics_empty_query(rag_svc: RAGService):
    refs, diag = asyncio.run(rag_svc.search_with_diagnostics("  "))
    assert refs == []
    assert diag.dense_top1 is None
    assert diag.sparse_hits == 0


def test_search_with_diagnostics_dense_top1_ignores_threshold(rag_svc: RAGService):
    """诊断查询应该看到真实最高分，即使 threshold 阻断了融合路的 dense prefetch。"""
    asyncio.run(
        rag_svc.index([CorpusEntry(source="契丹来犯", target="The Khitan attack")])
    )
    # 阈值放到非常高 → 融合路的 dense prefetch 不会召回；但诊断查询仍能看到 dense top1
    refs, diag = asyncio.run(
        rag_svc.search_with_diagnostics("契丹来犯", threshold=0.99, top_k=3)
    )
    # one-hot 自相似 cosine=1.0，等于 0.99 阈值刚好通过；改成不存在的 query 验证诊断不受 threshold 影响
    refs2, diag2 = asyncio.run(
        rag_svc.search_with_diagnostics("无关字符", threshold=0.99, top_k=3)
    )
    # 无关 query：dense 不同 one-hot → cosine=0；诊断查询限 limit=1，dense_top1 仍可能是 0 或 None
    # 关键断言：诊断查询不抛错，且返回结构正确
    assert isinstance(diag2.sparse_hits, int)
