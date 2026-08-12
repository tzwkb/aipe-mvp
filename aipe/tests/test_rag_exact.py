from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.config import get_settings
from app.services.rag_service import RAGService


class _NoEmbedLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        raise AssertionError(f"exact lookup must not embed: {text}")


class _FakeExactClient:
    def __init__(
        self,
        pages: dict[object, tuple[list[SimpleNamespace], object]],
        *,
        source_indexed: bool = True,
    ) -> None:
        self.pages = pages
        self.source_indexed = source_indexed
        self.scroll_calls: list[dict] = []
        self.payload_index_calls: list[dict] = []

    async def get_collection(self, collection_name: str):
        schema = {"source": object()} if self.source_indexed else {}
        return SimpleNamespace(payload_schema=schema)

    async def create_payload_index(self, **kwargs):
        self.payload_index_calls.append(kwargs)
        self.source_indexed = True

    async def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return self.pages[kwargs.get("offset")]


class _ClientBackedRAGService(RAGService):
    def __init__(self, client: _FakeExactClient, llm: _NoEmbedLLM) -> None:
        settings = get_settings().model_copy(update={"qdrant_collection": "exact_test"})
        super().__init__(settings, llm)  # type: ignore[arg-type]
        self.fake_client = client

    @property
    def _client(self):
        return self.fake_client


class _ReadOnlySearchClient:
    def __init__(self) -> None:
        self.payload_index_calls: list[dict] = []

    async def get_collection(self, collection_name: str):
        params = SimpleNamespace(
            vectors={"dense": SimpleNamespace(size=1)},
            sparse_vectors={"sparse": object()},
        )
        return SimpleNamespace(
            config=SimpleNamespace(params=params),
            payload_schema={},
        )

    async def create_payload_index(self, **kwargs):
        self.payload_index_calls.append(kwargs)
        raise PermissionError("read-only Qdrant credentials")

    async def query_points(self, **kwargs):
        return SimpleNamespace(points=[])


class _SearchLLM:
    async def embed(self, text: str) -> list[float]:
        return [1.0]


def _point(source: str, target: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(payload={"source": source, "target": target, "status": status})


def test_exact_lookup_pages_before_selecting_highest_priority_match():
    first_page = [_point("同源", f"Done {i}", "Done") for i in range(5)]
    second_page = [_point("同源", "Designer target", "Designer Reviewed")]
    client = _FakeExactClient(
        {
            None: (first_page, "page-2"),
            "page-2": (second_page, None),
        }
    )
    llm = _NoEmbedLLM()
    svc = _ClientBackedRAGService(client, llm)

    hits = asyncio.run(svc.find_exact_source_matches("同源", top_k=1))

    assert [(hit.target, hit.status) for hit in hits] == [
        ("Designer target", "Designer Reviewed")
    ]
    assert len(client.scroll_calls) == 2
    assert llm.calls == 0


def test_exact_lookup_batches_multiple_sources_into_one_scroll_sequence():
    client = _FakeExactClient(
        {
            None: (
                [
                    _point("源一", "Target one", "Done"),
                    _point("源二", "Target two", "Designer Reviewed"),
                ],
                None,
            )
        }
    )
    llm = _NoEmbedLLM()
    svc = _ClientBackedRAGService(client, llm)
    find_many = getattr(svc, "find_exact_source_matches_many", None)

    assert callable(find_many)
    matches = asyncio.run(find_many(["源一", "源二", "源一"], top_k=1))

    assert matches["源一"][0].target == "Target one"
    assert matches["源二"][0].target == "Target two"
    assert len(client.scroll_calls) == 1
    condition = client.scroll_calls[0]["scroll_filter"].must[0]
    assert set(condition.match.any) == {"源一", "源二"}
    assert llm.calls == 0


def test_exact_lookup_lazily_creates_source_payload_index():
    client = _FakeExactClient({None: ([], None)}, source_indexed=False)
    svc = _ClientBackedRAGService(client, _NoEmbedLLM())

    assert asyncio.run(svc.find_exact_source_matches("源文")) == []

    assert len(client.payload_index_calls) == 1
    assert client.payload_index_calls[0]["field_name"] == "source"
    assert client.payload_index_calls[0]["wait"] is True


def test_regular_search_does_not_require_payload_index_write_access():
    client = _ReadOnlySearchClient()
    settings = get_settings().model_copy(update={"qdrant_collection": "readonly_test"})

    class _ReadOnlyClientService(RAGService):
        @property
        def _client(self):
            return client

    svc = _ReadOnlyClientService(settings, _SearchLLM())  # type: ignore[arg-type]

    assert asyncio.run(svc.search("query")) == []
    assert client.payload_index_calls == []
