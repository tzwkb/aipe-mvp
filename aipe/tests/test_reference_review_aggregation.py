from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.aggregate_reference_reviews import aggregate


def _write_jsonl(path: Path, values: list[dict], marker: str | None = None) -> None:
    lines = [json.dumps(value, ensure_ascii=False) for value in values]
    if marker:
        lines.append(marker)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_chapters() -> list[dict]:
    return [
        {
            "chapter_id": f"chapter.{number:04d}",
            "chapter_number": number,
            "title": f"Chapter {number}",
            "line_start": number * 10,
            "line_end": number * 10 + 9,
            "sha256": f"sha-{number}",
        }
        for number in range(1, 1429)
    ]


def _review(source: dict) -> dict:
    return {
        "chapter_number": source["chapter_number"],
        "title": source["title"],
        "line_start": source["line_start"],
        "line_end": source["line_end"],
        "summary_zh": "逐章摘要",
        "entities": [],
        "relations": [],
        "events": [],
        "terms": [],
        "voices": [],
        "narrative": {
            "pov": "第三人称限知",
            "evidence_lines": [source["line_start"]],
        },
        "uncertainties": [],
    }


def test_aggregate_accepts_write_marker_and_validates_knowledge(tmp_path: Path):
    source = _source_chapters()
    source_path = tmp_path / "chapters.jsonl"
    _write_jsonl(source_path, source)
    parts = tmp_path / "parts"
    parts.mkdir()
    _write_jsonl(
        parts / "chapters_0001_0002.jsonl",
        [_review(source[0]), _review(source[1])],
        marker="__AIPE_CHAPTERS_0001_0002_CONTINUE__",
    )
    knowledge = {
        "schema": "aipe.manual-knowledge",
        "version": 1,
        "range": {
            "chapter_start": 1,
            "chapter_end": 2,
            "chapter_count": 2,
            "line_start": 10,
            "line_end": 29,
        },
        "counts": {"entities": 0, "relations": 0, "terms": 0, "voices": 0},
        "entities": [],
        "relations": [],
        "terms": [],
        "voices": [],
        "style_evidence": [
            {"chapter": 1, "evidence_lines": [10], "pov": "第三人称限知"}
        ],
    }
    (parts / "knowledge_0001_0002.json").write_text(
        json.dumps(knowledge, ensure_ascii=False), encoding="utf-8"
    )
    output = tmp_path / "aggregate"

    result = aggregate(
        argparse.Namespace(
            chapters=source_path,
            parts_dir=parts,
            output_dir=output,
            require_complete=False,
        )
    )

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert result == 0
    assert audit["reviewed_chapters"] == 2
    assert audit["validation_error_count"] == 0
    assert audit["knowledge_parts"][0]["validation_error_count"] == 0


def test_aggregate_rejects_knowledge_evidence_outside_referenced_chapter(
    tmp_path: Path,
):
    source = _source_chapters()
    source_path = tmp_path / "chapters.jsonl"
    _write_jsonl(source_path, source)
    parts = tmp_path / "parts"
    parts.mkdir()
    _write_jsonl(
        parts / "chapters_0001_0002.jsonl",
        [_review(source[0]), _review(source[1])],
    )
    knowledge = {
        "schema": "aipe.manual-knowledge",
        "version": 1,
        "range": {"chapter_start": 1, "chapter_end": 2, "chapter_count": 2},
        "entities": [
            {
                "name": "Example",
                "chapters": [1],
                "evidence_lines": [20],
            }
        ],
        "relations": [],
        "terms": [],
        "voices": [],
    }
    (parts / "knowledge_0001_0002.json").write_text(
        json.dumps(knowledge, ensure_ascii=False), encoding="utf-8"
    )
    output = tmp_path / "aggregate"

    result = aggregate(
        argparse.Namespace(
            chapters=source_path,
            parts_dir=parts,
            output_dir=output,
            require_complete=False,
        )
    )

    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert result == 1
    assert any(
        "evidence outside referenced chapters" in error
        for error in audit["validation_errors"]
    )
