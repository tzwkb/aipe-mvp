"""RAG 服务：双语语料入库 + 混合检索（dense + BM25 + RRF）+ 阈值过滤。

设计参考技术设计文档 §5.3，Day-N 升级为 hybrid：

- 向量库：Qdrant（async client），命名向量 ``dense``（cosine）+ ``sparse``（IDF/BM25）
- Dense embedding：复用 ``LLMService.embed_batch``（OpenAI 兼容协议）
- Sparse 端：jieba 中文分词 → token 哈希为 u32 索引；Qdrant 服务端按 ``Modifier.IDF``
  计算 IDF / BM25，无需自维护词表
- 检索：``query_points`` + ``Prefetch`` 双路召回 + ``FusionQuery(RRF)`` 融合
- ID 策略：``uuid5(source||target)``，重复入库幂等去重
- 阈值：cosine 阈值仅作用于 dense 路召回；融合后只用 ``top_k`` 控规模
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import jieba
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings
from app.errors import TranslationError
from app.schemas.rag import CorpusEntry, RAGSearchResult, status_to_rank, status_to_weight
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


# 与 (source, target) 一一对应的命名空间，保证不同部署/语料版本之间 ID 稳定。
_ID_NAMESPACE = uuid.UUID("0a8c1d8c-5e22-4f7a-9a5e-9c6f4a2d3b11")

# 入库批次大小：每批向 LLM 发一次 embedding 请求 + 一次 Qdrant upsert。
_INGEST_BATCH_SIZE = 30

# 命名向量名（与 collection schema 绑定，改名需重建 collection）
_VEC_DENSE = "dense"
_VEC_SPARSE = "sparse"
_SOURCE_PAYLOAD_FIELD = "source"
_EXACT_SCROLL_PAGE_SIZE = 256

# 仅保留 CJK / 英文 / 数字 token，过滤标点空白
_TOKEN_KEEP = re.compile(r"[一-鿿0-9A-Za-z]+")


def _tokenize(text: str) -> list[str]:
    """中英混排分词；CJK 走 jieba，英文/数字保留原 token。"""
    if not text:
        return []
    raw = jieba.lcut(text, cut_all=False)
    return [t.lower() for t in raw if _TOKEN_KEEP.fullmatch(t)]


def _hash_token(token: str) -> int:
    """token → u32 稀疏向量索引（md5 截断 4 字节）。"""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big")


def _to_sparse(text: str) -> models.SparseVector:
    """文本 → SparseVector(indices=u32 hash, values=tf)。

    Qdrant 端开启 ``Modifier.IDF`` 时会自动按集合统计计算 IDF/BM25，
    客户端只需提供 token 频次（query 侧 tf 取 1 即可）。
    """
    tokens = _tokenize(text)
    if not tokens:
        # 空 sparse 向量 Qdrant 会拒绝；返回单一 dummy 索引避免 upsert/search 失败
        return models.SparseVector(indices=[0], values=[0.0])
    counter = Counter(_hash_token(t) for t in tokens)
    indices = list(counter.keys())
    values = [float(v) for v in counter.values()]
    return models.SparseVector(indices=indices, values=values)


@dataclass(frozen=True)
class RAGDiagnostics:
    """RAG 检索诊断信息，用于"弱召回兜底 web search"等触发判断。

    - ``dense_top1``: dense（cosine）路 top1 原始相似度；无召回时为 None
    - ``sparse_hits``: sparse（BM25）路返回的命中数（受 ``sparse_prefetch`` 上限）

    这两个诊断查询是独立于 RRF 融合路径的，因此 ``dense_top1`` 不受
    ``rag_threshold`` 影响（诊断查询不带 score_threshold）。
    """

    dense_top1: float | None
    sparse_hits: int


def _dedup_by_source_target(results: list[RAGSearchResult], top_k: int) -> list[RAGSearchResult]:
    """对 RRF 结果按 (source, target) 去重，再按加权分排序。

    去重规则（相同 source + target）：
    - 只保留 priority_rank 最小（质量最高）的那条；rank 相同时保留 RRF score 较高者。

    排序规则（不同 target 之间）：
    - adjusted_score = raw_score × status_weight
    - 高质量语料（Designer Reviewed=1.00 … 其他=0.85）在分数接近时自然排前，
      但不会完全压制相似度显著更高的低质量条目。
    - 对外返回的 score 仍是 RRF 原始分，加权仅用于排序，保持可解释性。
    """
    best: dict[tuple[str, str], RAGSearchResult] = {}
    for r in results:
        key = (r.source.strip(), r.target.strip())
        existing = best.get(key)
        if existing is None:
            best[key] = r
            continue
        r_rank = status_to_rank(r.status)
        e_rank = status_to_rank(existing.status)
        if r_rank < e_rank or (r_rank == e_rank and r.score > existing.score):
            best[key] = r
    return sorted(
        best.values(),
        key=lambda x: x.score * status_to_weight(x.status),
        reverse=True,
    )[:top_k]


def _dedup_exact_matches(results: list[RAGSearchResult], top_k: int) -> list[RAGSearchResult]:
    """精确 source 命中结果去重排序。

    精确采用 TM 时，语义相似度已经是 100% source match；排序只看 TM 质量等级，
    同等级按 target 稳定排序，保证每次选择一致。
    """
    best_by_target: dict[str, RAGSearchResult] = {}
    for r in results:
        target_key = r.target.strip()
        existing = best_by_target.get(target_key)
        if existing is None or status_to_rank(r.status) < status_to_rank(existing.status):
            best_by_target[target_key] = r
    return sorted(
        best_by_target.values(),
        key=lambda x: (status_to_rank(x.status), x.target.strip()),
    )[:top_k]


class RAGService:
    """双语语料 RAG 服务（hybrid 检索）。

    生命周期：进程内单例，``__init__`` 不做 IO；``index`` / ``search`` 首次调用前
    会 lazy 建好 collection（dense 维度从 embedding 探测得到）。
    """

    def __init__(self, settings: Settings, llm_svc: LLMService):
        self.settings = settings
        self.llm_svc = llm_svc
        self.collection = settings.qdrant_collection
        self.dense_threshold = settings.rag_threshold
        self.top_k = settings.rag_top_k
        self.dense_prefetch = settings.rag_dense_prefetch
        self.sparse_prefetch = settings.rag_sparse_prefetch

        self._client_instance: AsyncQdrantClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._vector_size: int | None = None
        self._collection_ready: set[str] = set()
        self._source_index_ready: set[str] = set()
        self._init_lock = asyncio.Lock()
        self._source_index_lock = asyncio.Lock()

    @property
    def _client(self) -> AsyncQdrantClient:
        loop = asyncio.get_running_loop()
        if self._client_instance is None or self._client_loop is not loop:
            self._client_instance = AsyncQdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=30.0,
            )
            self._client_loop = loop
            self._init_lock = asyncio.Lock()
        return self._client_instance

    # ---------- collection 初始化 ----------

    async def _ensure_collection(self, collection: str) -> None:
        """探测 embedding 维度并按需创建 hybrid collection（幂等）。

        若检测到旧版（单匿名向量）schema，抛错提示需 ``--reset`` 重灌。
        """
        if collection in self._collection_ready:
            return
        _ = self._client
        async with self._init_lock:
            if collection in self._collection_ready:
                return

            if self._vector_size is None:
                probe = await self.llm_svc.embed("probe")
                self._vector_size = len(probe)
                logger.info(
                    "RAG embedding 维度探测: model=%s dim=%d",
                    self.settings.embedding_model,
                    self._vector_size,
                )

            try:
                info = await self._client.get_collection(collection)
            except (UnexpectedResponse, ValueError):
                await self._create_hybrid_collection(collection)
                self._collection_ready.add(collection)
                return

            # 校验已有 collection 是否为 hybrid schema
            vectors_cfg = info.config.params.vectors
            sparse_cfg = info.config.params.sparse_vectors
            if not isinstance(vectors_cfg, dict) or _VEC_DENSE not in vectors_cfg:
                raise TranslationError(
                    f"Qdrant collection '{collection}' 是旧版 schema（单匿名向量），"
                    "已不兼容 hybrid 检索；请用 `python -m scripts.ingest_corpus --reset` "
                    "重建 collection 并重新灌入语料。"
                )
            if not sparse_cfg or _VEC_SPARSE not in sparse_cfg:
                raise TranslationError(
                    f"Qdrant collection '{collection}' 缺少 sparse 向量 '{_VEC_SPARSE}'；"
                    "请用 `--reset` 重建 collection。"
                )
            existing_size = vectors_cfg[_VEC_DENSE].size
            if existing_size != self._vector_size:
                raise TranslationError(
                    f"Qdrant collection '{collection}' dense 维度 {existing_size} "
                    f"≠ 当前 embedding 维度 {self._vector_size}，请先 --reset 重建。"
                )
            logger.info("RAG hybrid collection 已存在: %s", collection)
            self._collection_ready.add(collection)

    async def _ensure_source_payload_index(self, collection: str) -> None:
        if collection in self._source_index_ready:
            return
        async with self._source_index_lock:
            if collection in self._source_index_ready:
                return
            info = await self._client.get_collection(collection)
            payload_schema = getattr(info, "payload_schema", None) or {}
            if _SOURCE_PAYLOAD_FIELD not in payload_schema:
                await self._client.create_payload_index(
                    collection_name=collection,
                    field_name=_SOURCE_PAYLOAD_FIELD,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
                logger.info(
                    "RAG payload index 已创建: collection=%s field=%s type=keyword",
                    collection,
                    _SOURCE_PAYLOAD_FIELD,
                )
            self._source_index_ready.add(collection)

    async def _create_hybrid_collection(self, collection: str) -> None:
        await self._client.create_collection(
            collection_name=collection,
            vectors_config={
                _VEC_DENSE: models.VectorParams(
                    size=self._vector_size,
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
        logger.info(
            "RAG hybrid collection 已创建: %s (dense_dim=%d cosine, sparse=BM25/IDF, on_disk=True)",
            collection,
            self._vector_size,
        )

    # ---------- 入库 ----------

    async def index(
        self,
        entries: list[CorpusEntry] | Iterable[CorpusEntry],
        *,
        collection: str | None = None,
        batch_size: int = _INGEST_BATCH_SIZE,
        progress_cb=None,
    ) -> int:
        """批量入库双语语料，返回成功入库条数。

        每批同时写入 dense 向量与 sparse 向量。
        ``collection`` 不填则用默认 collection。
        """
        coll = collection or self.collection
        items = [e for e in entries if e.source and e.target]
        if not items:
            return 0

        await self._ensure_collection(coll)

        total = len(items)
        done = 0
        for start in range(0, total, batch_size):
            batch = items[start : start + batch_size]
            sources = [e.source for e in batch]
            try:
                vectors = await self.llm_svc.embed_batch(sources)
            except TranslationError:
                raise
            except Exception as exc:
                raise TranslationError(f"[ERROR: AI_FAIL] embed_batch: {exc}") from exc

            points = [
                models.PointStruct(
                    # UUID 三元组：同一 (source, target) 允许因 status 不同而共存；
                    # 相同 (source, target, status) 重复上传幂等覆盖。
                    id=str(uuid.uuid5(_ID_NAMESPACE, f"{e.source}\x1f{e.target}\x1f{e.status or ''}")),
                    vector={
                        _VEC_DENSE: vec,
                        _VEC_SPARSE: _to_sparse(e.source),
                    },
                    payload={
                        "source": e.source,
                        "target": e.target,
                        **({"context": e.context} if e.context else {}),
                        **({"status": e.status, "priority_rank": status_to_rank(e.status)} if e.status else {}),
                    },
                )
                for e, vec in zip(batch, vectors)
            ]
            await self._client.upsert(
                collection_name=coll,
                points=points,
                wait=False,
            )
            done += len(batch)
            if progress_cb is not None:
                progress_cb(done, total)

        logger.info("RAG 入库完成: total=%d collection=%s", done, coll)
        return done

    # ---------- 检索 ----------

    async def find_exact_source_matches(
        self,
        source: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> list[RAGSearchResult]:
        """按 payload.source 做完全一致匹配，不调用 embedding。

        这个路径用于“直接采用 TM 100% match”开关：如果命中，pipeline 会直接使用
        target 并跳过 LLM 翻译；如果 collection 不存在或 Qdrant 异常，异常交给调用方
        降级处理。
        """
        exact_source = (source or "").strip()
        if not exact_source:
            return []
        matches = await self.find_exact_source_matches_many(
            [exact_source],
            collection=collection,
            top_k=top_k,
        )
        return matches[exact_source]

    async def find_exact_source_matches_many(
        self,
        sources: Iterable[str],
        *,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[RAGSearchResult]]:
        """批量查询完全一致的 payload.source；分页收齐后再按 TM 质量排序。"""
        exact_sources = list(
            dict.fromkeys((source or "").strip() for source in sources if (source or "").strip())
        )
        if not exact_sources:
            return {}

        coll = collection or self.collection
        k = self.top_k if top_k is None else top_k
        if k <= 0:
            return {source: [] for source in exact_sources}

        await self._ensure_source_payload_index(coll)
        source_set = set(exact_sources)
        raw_by_source: dict[str, list[RAGSearchResult]] = {
            source: [] for source in exact_sources
        }
        offset = None
        while True:
            points, next_offset = await self._client.scroll(
                collection_name=coll,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=_SOURCE_PAYLOAD_FIELD,
                            match=models.MatchAny(any=exact_sources),
                        )
                    ]
                ),
                limit=_EXACT_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                matched_source = str(payload.get(_SOURCE_PAYLOAD_FIELD, ""))
                target = payload.get("target")
                if matched_source not in source_set or not target:
                    continue
                raw_by_source[matched_source].append(
                    RAGSearchResult(
                        source=matched_source,
                        target=str(target),
                        score=1.0,
                        status=payload.get("status") or None,
                    )
                )
            if next_offset is None:
                break
            if next_offset == offset:
                raise TranslationError(
                    f"Qdrant exact-match scroll offset 未推进: collection={coll} offset={offset}"
                )
            offset = next_offset

        return {
            source: _dedup_exact_matches(raw_by_source[source], k)
            for source in exact_sources
        }

    async def search(
        self,
        query: str,
        threshold: float | None = None,
        top_k: int | None = None,
        collection: str | None = None,
    ) -> list[RAGSearchResult]:
        """混合检索：dense 召回 + BM25 召回 → RRF 融合 → 返回 Top-K。

        - ``threshold`` 仅作用于 dense 路（cosine 下限）；融合后不再过滤。
        - ``collection`` 不填则用默认 collection。
        """
        if not query or not query.strip():
            return []

        coll = collection or self.collection
        await self._ensure_collection(coll)

        thr = self.dense_threshold if threshold is None else threshold
        k = self.top_k if top_k is None else top_k

        try:
            dense_vec = await self.llm_svc.embed(query)
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(f"[ERROR: AI_FAIL] embed: {exc}") from exc

        sparse_vec = _to_sparse(query)

        # 过量召回：同一 (source, target) 可能因 status 不同存有多个点；
        # 多取 top_k * 3 候选，客户端去重后再返回 top_k。
        fetch_limit = max(k * 3, k + 10)

        resp = await self._client.query_points(
            collection_name=coll,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=_VEC_DENSE,
                    limit=self.dense_prefetch,
                    score_threshold=thr,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=_VEC_SPARSE,
                    limit=self.sparse_prefetch,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit,
            with_payload=True,
        )

        raw = [
            RAGSearchResult(
                source=str((h.payload or {}).get("source", "")),
                target=str((h.payload or {}).get("target", "")),
                score=float(h.score),
                status=(h.payload or {}).get("status") or None,
            )
            for h in resp.points
        ]
        return _dedup_by_source_target(raw, k)

    async def search_with_diagnostics(
        self,
        query: str,
        threshold: float | None = None,
        top_k: int | None = None,
        collection: str | None = None,
    ) -> tuple[list[RAGSearchResult], RAGDiagnostics]:
        """混合检索 + 诊断信息（dense top1 / sparse 命中数）。

        在原 RRF 融合查询基础上，并发再发两个独立的诊断查询：
        - dense-only top1（**不带** score_threshold，要看真实最高分）
        - sparse-only（按 ``sparse_prefetch`` 上限取 N，统计命中数）

        三路查询用 ``asyncio.gather`` 并发；诊断查询 ``with_payload=False``
        网络开销可忽略。失败时（任一查询出错）整体抛出 ``TranslationError``。
        """
        if not query or not query.strip():
            return [], RAGDiagnostics(dense_top1=None, sparse_hits=0)

        coll = collection or self.collection
        await self._ensure_collection(coll)

        thr = self.dense_threshold if threshold is None else threshold
        k = self.top_k if top_k is None else top_k

        try:
            dense_vec = await self.llm_svc.embed(query)
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(f"[ERROR: AI_FAIL] embed: {exc}") from exc

        sparse_vec = _to_sparse(query)
        fetch_limit = max(k * 3, k + 10)

        fusion_task = self._client.query_points(
            collection_name=coll,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=_VEC_DENSE,
                    limit=self.dense_prefetch,
                    score_threshold=thr,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=_VEC_SPARSE,
                    limit=self.sparse_prefetch,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit,
            with_payload=True,
        )
        dense_diag_task = self._client.query_points(
            collection_name=coll,
            query=dense_vec,
            using=_VEC_DENSE,
            limit=1,
            with_payload=False,
        )
        sparse_diag_task = self._client.query_points(
            collection_name=coll,
            query=sparse_vec,
            using=_VEC_SPARSE,
            limit=self.sparse_prefetch,
            with_payload=False,
        )

        fusion_resp, dense_resp, sparse_resp = await asyncio.gather(
            fusion_task, dense_diag_task, sparse_diag_task
        )

        raw = [
            RAGSearchResult(
                source=str((h.payload or {}).get("source", "")),
                target=str((h.payload or {}).get("target", "")),
                score=float(h.score),
                status=(h.payload or {}).get("status") or None,
            )
            for h in fusion_resp.points
        ]
        refs = _dedup_by_source_target(raw, k)

        dense_top1 = float(dense_resp.points[0].score) if dense_resp.points else None
        sparse_hits = len(sparse_resp.points)
        return refs, RAGDiagnostics(dense_top1=dense_top1, sparse_hits=sparse_hits)

    # ---------- 运维 / 调试辅助 ----------

    async def count(self, collection: str | None = None) -> int:
        """返回指定 collection 内的点数（人工抽查用）。"""
        coll = collection or self.collection
        await self._ensure_collection(coll)
        res = await self._client.count(collection_name=coll, exact=True)
        return int(res.count)

    async def reset_collection(self, collection: str | None = None) -> None:
        """删除并重建 collection（CLI 入库脚本可选调用，谨慎使用）。"""
        coll = collection or self.collection
        try:
            await self._client.delete_collection(coll)
            logger.warning("RAG collection 已删除: %s", coll)
        except (UnexpectedResponse, ValueError):
            pass
        self._collection_ready.discard(coll)
        self._source_index_ready.discard(coll)
        await self._ensure_collection(coll)

    async def aclose(self) -> None:
        if self._client_instance is None:
            return
        try:
            await self._client_instance.close()
        finally:
            self._client_instance = None
            self._client_loop = None
            self._init_lock = asyncio.Lock()
            self._source_index_lock = asyncio.Lock()


# ---------- 模块级单例 ----------

_singleton: RAGService | None = None


def get_rag_service() -> RAGService:
    """全局单例。配置变更需重启服务。"""
    global _singleton
    if _singleton is None:
        from app.config import get_settings
        from app.dependencies import get_llm_service

        _singleton = RAGService(get_settings(), get_llm_service())
    return _singleton


def reset_rag_service() -> None:
    """测试用：清空单例状态。"""
    global _singleton
    _singleton = None


__all__ = [
    "RAGDiagnostics",
    "RAGService",
    "get_rag_service",
    "reset_rag_service",
]
