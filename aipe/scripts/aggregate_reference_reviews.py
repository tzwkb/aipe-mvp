"""Validate and aggregate local Codex subagent chapter reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PART_RE = re.compile(r"^chapters_(\d{4})_(\d{4})\.jsonl$")
KNOWLEDGE_RE = re.compile(r"^knowledge_(\d{4})_(\d{4})\.json$")
WRITE_MARKER_RE = re.compile(r"^__AIPE_CHAPTERS_\d{4}_\d{4}_CONTINUE__$")
ARRAY_FIELDS = ("entities", "relations", "events", "terms", "voices", "uncertainties")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        if WRITE_MARKER_RE.fullmatch(raw.strip()):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}: line {line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise TypeError(f"{path}: line {line_number}: top level must be an object")
        records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in values
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direct_evidence_lines(value: dict[str, Any]) -> tuple[list[int], list[str]]:
    lines: list[int] = []
    errors: list[str] = []
    for key in ("line", "line_start", "line_end"):
        item = value.get(key)
        if item is None:
            continue
        if isinstance(item, int):
            lines.append(item)
        else:
            errors.append(f"{key} must be an integer")
    for key in ("lines", "evidence_lines"):
        item = value.get(key)
        if item is None:
            continue
        if not isinstance(item, list) or any(
            not isinstance(number, int) for number in item
        ):
            errors.append(f"{key} must be an integer array")
            continue
        lines.extend(item)
    return lines, errors


def _validate_knowledge_evidence(
    value: Any,
    *,
    source_by_number: dict[int, dict[str, Any]],
    range_start: int,
    range_end: int,
    path: str = "$",
    inherited_chapters: tuple[int, ...] = (),
) -> list[str]:
    errors: list[str] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(
                _validate_knowledge_evidence(
                    item,
                    source_by_number=source_by_number,
                    range_start=range_start,
                    range_end=range_end,
                    path=f"{path}[{index}]",
                    inherited_chapters=inherited_chapters,
                )
            )
        return errors
    if not isinstance(value, dict):
        return errors

    local_chapters: tuple[int, ...] = inherited_chapters
    for key in ("chapter", "chapter_number"):
        chapter = value.get(key)
        if isinstance(chapter, int):
            local_chapters = (chapter,)
            break
    else:
        chapters = value.get("chapters")
        if isinstance(chapters, list) and chapters and all(
            isinstance(chapter, int) for chapter in chapters
        ):
            local_chapters = tuple(chapters)
        else:
            mentions = value.get("chapter_mentions")
            if isinstance(mentions, list) and mentions and all(
                isinstance(chapter, int) for chapter in mentions
            ):
                local_chapters = tuple(mentions)

    invalid_chapters = sorted(
        {
            chapter
            for chapter in local_chapters
            if chapter not in source_by_number
            or not range_start <= chapter <= range_end
        }
    )
    if invalid_chapters:
        errors.append(f"{path}: chapter references outside declared range: {invalid_chapters[:8]}")

    lines, line_errors = _direct_evidence_lines(value)
    errors.extend(f"{path}: {error}" for error in line_errors)
    if lines:
        overall_start = source_by_number[range_start]["line_start"]
        overall_end = source_by_number[range_end]["line_end"]
        outside_range = sorted(
            {line for line in lines if not overall_start <= line <= overall_end}
        )
        if outside_range:
            errors.append(
                f"{path}: evidence outside knowledge range: {outside_range[:8]}"
            )
        valid_context_chapters = tuple(
            chapter
            for chapter in local_chapters
            if chapter in source_by_number and range_start <= chapter <= range_end
        )
        if valid_context_chapters:
            outside_context = sorted(
                {
                    line
                    for line in lines
                    if not any(
                        source_by_number[chapter]["line_start"]
                        <= line
                        <= source_by_number[chapter]["line_end"]
                        for chapter in valid_context_chapters
                    )
                }
            )
            if outside_context:
                errors.append(
                    f"{path}: evidence outside referenced chapters: {outside_context[:8]}"
                )

    evidence_keys = {"line", "line_start", "line_end", "lines", "evidence_lines"}
    for key, item in value.items():
        if key in evidence_keys:
            continue
        errors.extend(
            _validate_knowledge_evidence(
                item,
                source_by_number=source_by_number,
                range_start=range_start,
                range_end=range_end,
                path=f"{path}.{key}",
                inherited_chapters=local_chapters,
            )
        )
    return errors


def _validate_knowledge_file(
    path: Path,
    *,
    range_start: int,
    range_end: int,
    source_by_number: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]
    if not isinstance(value, dict):
        return None, ["top level must be an object"]
    if not str(value.get("schema") or "").startswith("aipe.manual-knowledge"):
        errors.append(f"unexpected schema {value.get('schema')!r}")
    declared_range = value.get("range")
    if not isinstance(declared_range, dict):
        errors.append("range must be an object")
    else:
        if declared_range.get("chapter_start") != range_start:
            errors.append("range.chapter_start mismatch")
        if declared_range.get("chapter_end") != range_end:
            errors.append("range.chapter_end mismatch")
        expected_count = range_end - range_start + 1
        if declared_range.get("chapter_count") != expected_count:
            errors.append("range.chapter_count mismatch")
    counts = value.get("counts")
    if isinstance(counts, dict):
        for field in ("entities", "relations", "terms", "voices", "uncertainties"):
            declared = counts.get(field)
            actual = value.get(field)
            if isinstance(declared, int) and isinstance(actual, list) and declared != len(actual):
                errors.append(f"counts.{field}={declared} but {field} has {len(actual)} items")
    errors.extend(
        _validate_knowledge_evidence(
            value,
            source_by_number=source_by_number,
            range_start=range_start,
            range_end=range_end,
        )
    )
    return value, errors


def _evidence_lines(value: Any) -> list[int]:
    found: list[int] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"line", "line_start", "line_end"} and isinstance(item, int):
                found.append(item)
            elif key in {"lines", "evidence_lines"} and isinstance(item, list):
                found.extend(number for number in item if isinstance(number, int))
            elif key == "ev" and isinstance(item, list):
                found.extend(
                    evidence[0]
                    for evidence in item
                    if isinstance(evidence, list)
                    and evidence
                    and isinstance(evidence[0], int)
                )
            found.extend(_evidence_lines(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_evidence_lines(item))
    return found


def _validate_record(
    value: dict[str, Any],
    source: dict[str, Any],
    *,
    part_path: Path,
) -> list[str]:
    chapter_number = source["chapter_number"]
    errors: list[str] = []
    if value.get("chapter_number") != chapter_number:
        errors.append(f"chapter_number must be {chapter_number}")
    if value.get("title") != source.get("title"):
        errors.append(
            f"title mismatch: {value.get('title')!r} != {source.get('title')!r}"
        )
    for field in ("line_start", "line_end"):
        if value.get(field) != source.get(field):
            errors.append(
                f"{field} mismatch: {value.get(field)!r} != {source.get(field)!r}"
            )
    if not str(value.get("summary_zh") or "").strip():
        errors.append("summary_zh is empty")
    for field in ARRAY_FIELDS:
        items = value.get(field)
        if not isinstance(items, list):
            errors.append(f"{field} must be an array")
            continue
        if field == "uncertainties":
            required_evidence_items = items
        else:
            required_evidence_items = items
        for index, item in enumerate(required_evidence_items):
            if not isinstance(item, dict):
                errors.append(f"{field}[{index}] must be an object")
                continue
            lines = _evidence_lines(item)
            if not lines:
                errors.append(f"{field}[{index}] has no evidence line")
                continue
            invalid = sorted(
                {
                    line
                    for line in lines
                    if not source["line_start"] <= line <= source["line_end"]
                }
            )
            if invalid:
                errors.append(
                    f"{field}[{index}] evidence outside chapter range: {invalid[:8]}"
                )
    narrative = (
        value.get("narrative") or value.get("narrative_style") or value.get("style")
    )
    if not isinstance(narrative, dict):
        errors.append("narrative/narrative_style/style must be an object")
    elif not _evidence_lines(narrative):
        errors.append("narrative has no evidence line")
    return [f"{part_path.name}: chapter {chapter_number}: {error}" for error in errors]


def _normalize_record(
    value: dict[str, Any],
    source: dict[str, Any],
    part_path: Path,
) -> dict[str, Any]:
    normalized = {
        "schema": "aipe.reference-chapter-review",
        "version": 1,
        "review_method": "codex_subagent_full_chapter_read",
        "chapter_id": source["chapter_id"],
        "chapter_number": source["chapter_number"],
        "title": source["title"],
        "line_start": source["line_start"],
        "line_end": source["line_end"],
        "source_sha256": source["sha256"],
        "review_part": part_path.name,
        "summary_zh": value["summary_zh"],
        "entities": value["entities"],
        "relations": value["relations"],
        "events": value["events"],
        "terms": value["terms"],
        "voices": value["voices"],
        "narrative": value.get("narrative")
        or value.get("narrative_style")
        or value.get("style"),
        "uncertainties": value["uncertainties"],
    }
    normalized["review_sha256"] = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return normalized


def _flatten(
    chapters: list[dict[str, Any]],
    field: str,
    record_kind: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chapter in chapters:
        for index, item in enumerate(chapter[field], 1):
            output.append(
                {
                    "schema": f"aipe.reference-{record_kind}-mention",
                    "version": 1,
                    "mention_id": (
                        f"{chapter['chapter_id']}.{record_kind}.{index:03d}"
                    ),
                    "chapter_number": chapter["chapter_number"],
                    "line_start": chapter["line_start"],
                    "line_end": chapter["line_end"],
                    "source_sha256": chapter["source_sha256"],
                    "record": item,
                }
            )
    return output


def aggregate(args: argparse.Namespace) -> int:
    source_chapters = _read_jsonl(args.chapters)
    source_by_number = {item["chapter_number"]: item for item in source_chapters}
    if sorted(source_by_number) != list(range(1, 1429)):
        raise SystemExit("source chapter index must contain exactly Chapters 1-1428")

    part_paths = sorted(
        path
        for path in args.parts_dir.glob("chapters_*.jsonl")
        if PART_RE.fullmatch(path.name)
    )
    errors: list[str] = []
    records_by_number: dict[int, dict[str, Any]] = {}
    provenance: dict[int, Path] = {}
    part_manifest: list[dict[str, Any]] = []
    completed_part_ranges: set[tuple[int, int]] = set()
    for part_path in part_paths:
        match = PART_RE.fullmatch(part_path.name)
        assert match is not None
        range_start, range_end = map(int, match.groups())
        try:
            part_records = _read_jsonl(part_path)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            part_manifest.append(
                {
                    "path": str(part_path.resolve()),
                    "sha256": _sha256(part_path),
                    "declared_range": [range_start, range_end],
                    "record_count": None,
                    "parse_error": str(exc),
                }
            )
            continue
        append_markers = [
            record for record in part_records if record.get("_append_marker") is True
        ]
        review_records = [
            record for record in part_records if record.get("_append_marker") is not True
        ]
        part_numbers = [
            chapter_number
            for record in review_records
            if isinstance((chapter_number := record.get("chapter_number")), int)
        ]
        if set(part_numbers) == set(range(range_start, range_end + 1)):
            completed_part_ranges.add((range_start, range_end))
        part_manifest.append(
            {
                "path": str(part_path.resolve()),
                "sha256": _sha256(part_path),
                "declared_range": [range_start, range_end],
                "record_count": len(review_records),
                "append_marker_count": len(append_markers),
                "first_chapter": min(part_numbers) if part_numbers else None,
                "last_chapter": max(part_numbers) if part_numbers else None,
            }
        )
        for value in review_records:
            chapter_number = value.get("chapter_number")
            if (
                not isinstance(chapter_number, int)
                or chapter_number not in source_by_number
            ):
                errors.append(
                    f"{part_path.name}: invalid chapter_number {chapter_number!r}"
                )
                continue
            if not range_start <= chapter_number <= range_end:
                errors.append(
                    f"{part_path.name}: chapter {chapter_number} outside declared range"
                )
            if chapter_number in records_by_number:
                errors.append(
                    f"duplicate chapter {chapter_number}: "
                    f"{provenance[chapter_number].name} and {part_path.name}"
                )
                continue
            errors.extend(
                _validate_record(
                    value,
                    source_by_number[chapter_number],
                    part_path=part_path,
                )
            )
            records_by_number[chapter_number] = value
            provenance[chapter_number] = part_path

    knowledge_manifest: list[dict[str, Any]] = []
    knowledge_ranges: set[tuple[int, int]] = set()
    knowledge_paths = sorted(
        path
        for path in args.parts_dir.glob("knowledge_*.json")
        if KNOWLEDGE_RE.fullmatch(path.name)
    )
    for knowledge_path in knowledge_paths:
        match = KNOWLEDGE_RE.fullmatch(knowledge_path.name)
        assert match is not None
        range_start, range_end = map(int, match.groups())
        declared_range = (range_start, range_end)
        if declared_range in knowledge_ranges:
            errors.append(f"duplicate knowledge range {range_start}-{range_end}")
        knowledge_ranges.add(declared_range)
        if (
            range_start not in source_by_number
            or range_end not in source_by_number
            or range_start > range_end
        ):
            knowledge_errors = ["filename declares an invalid chapter range"]
            knowledge_value = None
        else:
            knowledge_value, knowledge_errors = _validate_knowledge_file(
                knowledge_path,
                range_start=range_start,
                range_end=range_end,
                source_by_number=source_by_number,
            )
        errors.extend(
            f"{knowledge_path.name}: {error}" for error in knowledge_errors
        )
        knowledge_manifest.append(
            {
                "path": str(knowledge_path.resolve()),
                "sha256": _sha256(knowledge_path),
                "declared_range": [range_start, range_end],
                "schema": (
                    knowledge_value.get("schema")
                    if isinstance(knowledge_value, dict)
                    else None
                ),
                "validation_error_count": len(knowledge_errors),
            }
        )
    for range_start, range_end in sorted(completed_part_ranges - knowledge_ranges):
        errors.append(
            f"completed chapters_{range_start:04d}_{range_end:04d}.jsonl "
            f"is missing knowledge_{range_start:04d}_{range_end:04d}.json"
        )

    present = sorted(records_by_number)
    missing = sorted(set(source_by_number) - set(present))
    normalized = [
        _normalize_record(
            records_by_number[number], source_by_number[number], provenance[number]
        )
        for number in present
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path] = {
        "chapters": output_dir / "chapter_reviews.jsonl",
        "entities": output_dir / "entity_mentions.jsonl",
        "relations": output_dir / "relation_mentions.jsonl",
        "events": output_dir / "event_mentions.jsonl",
        "terms": output_dir / "term_mentions.jsonl",
        "voices": output_dir / "voice_mentions.jsonl",
        "uncertainties": output_dir / "uncertainty_mentions.jsonl",
    }
    _write_jsonl(output_files["chapters"], normalized)
    for field, kind in (
        ("entities", "entity"),
        ("relations", "relation"),
        ("events", "event"),
        ("terms", "term"),
        ("voices", "voice"),
        ("uncertainties", "uncertainty"),
    ):
        _write_jsonl(output_files[field], _flatten(normalized, field, kind))

    counts = {
        field: sum(len(chapter[field]) for chapter in normalized)
        for field in ARRAY_FIELDS
    }
    narrative_pov = Counter(
        str(
            chapter["narrative"].get("pov")
            or chapter["narrative"].get("viewpoint")
            or ""
        )
        for chapter in normalized
        if isinstance(chapter["narrative"], dict)
    )
    audit = {
        "schema": "aipe.reference-review-audit",
        "version": 1,
        "review_method": "codex_subagent_full_chapter_read",
        "source_chapters": len(source_chapters),
        "reviewed_chapters": len(normalized),
        "coverage_percent": round(len(normalized) / len(source_chapters) * 100, 6),
        "first_reviewed": present[0] if present else None,
        "last_reviewed": present[-1] if present else None,
        "missing_chapters": missing,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "record_counts": counts,
        "narrative_pov_counts": dict(narrative_pov.most_common()),
        "parts": part_manifest,
        "knowledge_parts": knowledge_manifest,
        "outputs": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in output_files.items()
        },
    }
    _write_json(output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors or (args.require_complete and missing):
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapters", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(aggregate(parse_args()))
