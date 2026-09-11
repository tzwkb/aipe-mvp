from app.services.reference_corpus_service import (
    ReferenceCorpusHit,
    ReferenceCorpusService,
)


def test_render_reference_hits_keeps_provenance_and_caps_excerpt():
    hit = ReferenceCorpusHit(
        score=7.25,
        chunk_id="lom.chapter.0034.chunk.001",
        chapter_number=34,
        chapter_title="Advance Payment",
        line_start=5624,
        line_end=5700,
        text=("opening " * 300)
        + "The Fool answered Audrey calmly."
        + (" ending" * 300),
        source_aliases=("愚者", "奥黛丽"),
        client_terms=(),
    )

    rendered = ReferenceCorpusService.render(
        [hit],
        target_anchors=["The Fool", "Audrey"],
        max_chars=1500,
        excerpt_chars=240,
    )

    assert "Chapter 34: Advance Payment" in rendered
    assert "Lord.txt L5624-L5700" in rendered
    assert "The Fool answered Audrey calmly" in rendered
    assert len(rendered) <= 1500
    assert ("opening " * 100) not in rendered


def test_render_reference_hits_does_not_present_hit_as_semantic_equivalence():
    hit = ReferenceCorpusHit(
        score=1.0,
        chunk_id="chunk",
        chapter_number=1,
        chapter_title="Crimson",
        line_start=4,
        line_end=20,
        text="Klein opened his eyes.",
        source_aliases=("克莱恩",),
        client_terms=(),
    )

    rendered = ReferenceCorpusService.render([hit], target_anchors=["Klein"])

    assert "不代表与当前原文语义等同" in rendered
    assert "Klein opened his eyes" in rendered
