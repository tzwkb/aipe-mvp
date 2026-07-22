"""Shared test doubles and project fixtures for translation-pipeline tests."""

from __future__ import annotations

import asyncio
import json

from app.schemas.rag import RAGSearchResult
from app.schemas.terminology import TermEntry
from app.schemas.web_search import WebSearchResult
from app.services.rag_service import RAGDiagnostics
from app.services.style_guide_service import StyleGuideService
from app.services.terminology_service import TerminologyService
from app.services.translation_pipeline import TranslationPipeline


class FakeLLM:
    def __init__(self, response_fn=None, classify_fn=None):
        self.calls: list[list[dict]] = []
        self.classify_calls: list[list[dict]] = []
        self.response_fn = response_fn or (lambda msgs: "stubbed translation")
        self.classify_fn = classify_fn or (lambda msgs: "未知")

    async def translate(self, prompt, *, temperature=None, max_retries=None):
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        self.calls.append(messages)
        output = self.response_fn(messages)
        if asyncio.iscoroutine(output):
            return await output
        return output

    async def classify(self, messages):
        self.classify_calls.append(messages)
        output = self.classify_fn(messages)
        if asyncio.iscoroutine(output):
            return await output
        return output

    async def embed(self, text):
        return [0.0]

    async def embed_batch(self, texts):
        return [[0.0] for _ in texts]


class FakeRAG:
    def __init__(
        self,
        results: list[RAGSearchResult] | None = None,
        raise_exc: Exception | None = None,
        results_by_query: dict[str, list[RAGSearchResult]] | None = None,
        exact_results_by_query: dict[str, list[RAGSearchResult]] | None = None,
        top_k: int = 3,
        diagnostics: RAGDiagnostics | None = None,
        diagnostics_by_query: dict[str, RAGDiagnostics] | None = None,
    ):
        self._results = results or []
        self._raise = raise_exc
        self._by_query = results_by_query or {}
        self._exact_by_query = exact_results_by_query or {}
        self.top_k = top_k
        self.calls: list[tuple[str, float | None, int | None]] = []
        self.exact_calls: list[tuple[str, str | None, int | None]] = []
        self.exact_batch_calls: list[tuple[list[str], str | None, int | None]] = []
        self.diag_calls: list[str] = []
        self._diag_default = diagnostics or RAGDiagnostics(dense_top1=0.9, sparse_hits=1)
        self._diag_by_query = diagnostics_by_query or {}
        self.collection_calls: list[str | None] = []

    async def search(self, query, threshold=None, top_k=None, collection=None):
        self.calls.append((query, threshold, top_k))
        self.collection_calls.append(collection)
        if self._raise:
            raise self._raise
        if query in self._by_query:
            return list(self._by_query[query])
        return list(self._results)

    async def search_with_diagnostics(self, query, threshold=None, top_k=None, collection=None):
        self.calls.append((query, threshold, top_k))
        self.collection_calls.append(collection)
        self.diag_calls.append(query)
        if self._raise:
            raise self._raise
        results = list(self._by_query.get(query, self._results))
        diagnostics = self._diag_by_query.get(query, self._diag_default)
        return results, diagnostics

    async def find_exact_source_matches(self, source, collection=None, top_k=None):
        self.exact_calls.append((source, collection, top_k))
        self.collection_calls.append(collection)
        if self._raise:
            raise self._raise
        return list(self._exact_by_query.get(source, []))[: top_k or self.top_k]

    async def find_exact_source_matches_many(self, sources, collection=None, top_k=None):
        source_list = list(sources)
        self.exact_batch_calls.append((source_list, collection, top_k))
        self.collection_calls.append(collection)
        if self._raise:
            raise self._raise
        return {
            source: list(self._exact_by_query.get(source, []))[: top_k or self.top_k]
            for source in dict.fromkeys(source_list)
        }


class FakeWebSearch:
    def __init__(
        self,
        results: list[WebSearchResult] | None = None,
        raise_exc: Exception | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._results = results or []
        self._raise = raise_exc
        self.calls: list[str] = []

    async def search(self, query: str, *, prefix: str | None = None):
        self.calls.append(query)
        if self._raise:
            raise self._raise
        return list(self._results)


def make_pipeline(
    *,
    terms: list[TermEntry] | None = None,
    style_rules: str = "保持武侠调性。",
    rag_results: list[RAGSearchResult] | None = None,
    llm_response_fn=None,
    rag_exc: Exception | None = None,
    diagnostics: RAGDiagnostics | None = None,
    web_search: FakeWebSearch | None = None,
    web_search_dense_threshold: float = 0.6,
):
    term_svc = TerminologyService()
    if terms:
        term_svc.load(terms)

    style_svc = StyleGuideService()
    if style_rules:
        style_svc.load(style_rules, filename="test.md")

    rag = FakeRAG(rag_results, raise_exc=rag_exc, diagnostics=diagnostics)
    llm = FakeLLM(response_fn=llm_response_fn)
    pipeline = TranslationPipeline(
        term_svc,
        rag,
        style_svc,
        llm,
        web_search_svc=web_search,
        web_search_dense_threshold=web_search_dense_threshold,
    )
    return pipeline, term_svc, rag, llm


def write_project(
    root,
    name: str,
    *,
    language_pair: str = "ZH-EN",
    source_lang: str = "zh",
    target_lang: str = "en",
    term_source: str,
    term_target: str,
    style: str,
    collection: str,
) -> None:
    project_dir = root / name
    project_dir.mkdir(parents=True)
    (project_dir / "terms.json").write_text(
        json.dumps([{"source": term_source, "target": term_target}], ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "style.md").write_text(style, encoding="utf-8")
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": name,
                "language_pair": language_pair,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "game": name,
                "style_guide": "style.md",
                "terminology": "terms.json",
                "qdrant_collection": collection,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_min_project(
    root,
    name: str,
    *,
    language_pair: str,
    source_lang: str,
    target_lang: str,
    collection: str,
) -> None:
    project_dir = root / name
    project_dir.mkdir(parents=True)
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": name,
                "language_pair": language_pair,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "game": "Isekai",
                "qdrant_collection": collection,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


__all__ = [
    "FakeLLM",
    "FakeRAG",
    "FakeWebSearch",
    "make_pipeline",
    "write_min_project",
    "write_project",
]
