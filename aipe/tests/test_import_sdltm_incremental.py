import asyncio
from types import SimpleNamespace

from scripts.import_sdltm_incremental import (
    Candidate,
    ExistingPoint,
    best_candidate_by_pair,
    extract_segment_text,
    plan_incremental_import,
    status_for_path,
)


def test_status_for_path_maps_tm_tiers():
    assert status_for_path("Designer库0629.sdltm") == "Designer Reviewed"
    assert status_for_path("CQA库0629.sdltm") == "CQA_Done"
    assert status_for_path("LQA库0629.sdltm") == "Done_LQA edited"
    assert status_for_path("GT库0629.sdltm") == "Done"
    assert status_for_path("ST库0629.sdltm") == "Done"


def test_status_for_path_accepts_explicit_default_for_unclassified_tm():
    assert status_for_path("Isekai_EN-DE_TM_20260702.sdltm", default_status="Done") == "Done"


def test_status_for_path_still_rejects_unclassified_tm_without_default():
    try:
        status_for_path("Isekai_EN-DE_TM_20260702.sdltm")
    except ValueError as exc:
        assert "无法从文件名判断 TM 分级" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_extract_segment_text_uses_text_values_and_ignores_tag_metadata():
    xml = """
    <Segment>
      <Elements>
        <Text><Value>问</Value></Text>
        <Tag>
          <Type>Start</Type>
          <Anchor>7</Anchor>
          <TagID>not visible</TagID>
        </Tag>
        <Text><Value>剑</Value></Text>
      </Elements>
      <CultureName>zh-CN</CultureName>
    </Segment>
    """

    assert extract_segment_text(xml) == "问剑"


def test_best_candidate_by_pair_keeps_highest_quality_status():
    candidates = [
        Candidate("同源", "Same target", "Done", "GT库0629.sdltm"),
        Candidate("同源", "Same target", "Designer Reviewed", "Designer库0629.sdltm"),
        Candidate("另一条", "Another", "CQA_Done", "CQA库0629.sdltm"),
    ]

    selected, skipped = best_candidate_by_pair(candidates)

    assert selected[("同源", "Same target")].status == "Designer Reviewed"
    assert selected[("另一条", "Another")].status == "CQA_Done"
    assert skipped == 1


def test_plan_incremental_import_skips_existing_and_reuses_vectors_for_upgrades():
    new_best = Candidate("新句", "New target", "Done", "GT库0629.sdltm")
    exact_existing = Candidate("旧句", "Old target", "CQA_Done", "CQA库0629.sdltm")
    upgrade = Candidate("升级句", "Upgrade target", "Designer Reviewed", "Designer库0629.sdltm")
    worse_than_existing = Candidate("已更好", "Better existing", "Done", "GT库0629.sdltm")

    existing = {
        ("旧句", "Old target"): [
            ExistingPoint("old-exact", "CQA_Done"),
        ],
        ("升级句", "Upgrade target"): [
            ExistingPoint("old-lqa", "Done_LQA edited"),
        ],
        ("已更好", "Better existing"): [
            ExistingPoint("old-designer", "Designer Reviewed"),
        ],
    }

    plan = plan_incremental_import(
        {
            (new_best.source, new_best.target): new_best,
            (exact_existing.source, exact_existing.target): exact_existing,
            (upgrade.source, upgrade.target): upgrade,
            (worse_than_existing.source, worse_than_existing.target): worse_than_existing,
        },
        existing,
    )

    assert plan.to_embed == [new_best]
    assert plan.to_clone == [(upgrade, "old-lqa", ["old-lqa"])]
    assert plan.skipped_existing >= 2
    assert plan.embedding_sources == {"新句"}


def test_run_creates_missing_collection_when_committing(monkeypatch, tmp_path):
    import scripts.import_sdltm_incremental as mod

    class FakeClient:
        instances = []

        def __init__(self, *args, **kwargs):
            self.created = []
            FakeClient.instances.append(self)

        async def get_collection(self, collection_name):
            raise ValueError(f"missing: {collection_name}")

        async def create_collection(self, **kwargs):
            self.created.append(kwargs)

        async def close(self):
            pass

    class FakeLLMService:
        def __init__(self, settings):
            pass

        async def embed(self, text):
            return [0.0, 0.0, 0.0]

        async def aclose(self):
            pass

    async def fake_scan_existing_pairs(client, collection, scroll_limit):
        return {}

    monkeypatch.setattr("qdrant_client.AsyncQdrantClient", FakeClient)
    monkeypatch.setattr("app.services.llm_service.LLMService", FakeLLMService)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(
            qdrant_collection="new_corpus",
            qdrant_host="localhost",
            qdrant_port=6333,
            embedding_model="fake-embed",
        ),
    )
    monkeypatch.setattr(mod, "load_candidates", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(mod, "scan_existing_pairs", fake_scan_existing_pairs)

    args = SimpleNamespace(
        tm_dir=tmp_path,
        collection="new_corpus",
        limit=0,
        default_status=None,
        scroll_limit=10,
        progress=tmp_path / "summary.json",
        commit=True,
        batch_size=30,
        embedding_concurrency=1,
    )

    summary = asyncio.run(mod.run(args))

    assert summary["collection"] == "new_corpus"
    assert FakeClient.instances[0].created
