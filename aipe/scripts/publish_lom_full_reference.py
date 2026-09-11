"""Publish audited local-only LOM full-reference assets into the AIPE profile."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


COPY_SPECS = (
    (
        "entities_runtime_registry",
        "synthesis/entities.runtime.draft.json",
        "entities_runtime_registry.json",
    ),
    ("review_audit", "manual-aggregate/audit.json", "review_audit.json"),
    (
        "entity_synthesis_audit",
        "synthesis/entity_synthesis_audit.json",
        "entity_synthesis_audit.json",
    ),
    (
        "entities_runtime_build_audit",
        "synthesis/entities_runtime_build_audit.json",
        "entities_runtime_build_audit.json",
    ),
    (
        "entities_runtime_independent_audit",
        "synthesis/entities_runtime_independent_audit.json",
        "entities_runtime_independent_audit.json",
    ),
    (
        "voice_profiles",
        "synthesis/voice_profiles.draft.json",
        "voice_profiles.json",
    ),
    (
        "narrative_style_patterns",
        "synthesis/narrative_style_patterns.draft.json",
        "narrative_style_patterns.json",
    ),
    (
        "context_rules_fullbook",
        "synthesis/context_rules_fullbook.draft.json",
        "context_rules_fullbook.json",
    ),
    (
        "voice_style_synthesis_audit",
        "synthesis/voice_style_synthesis_audit.json",
        "voice_style_synthesis_audit.json",
    ),
    (
        "style_runtime_build_audit",
        "synthesis/style_runtime_build_audit.json",
        "style_runtime_build_audit.json",
    ),
    (
        "lore_systems",
        "synthesis/lore_systems.draft.json",
        "lore_systems.json",
    ),
    (
        "terminology_fullbook",
        "synthesis/terminology_fullbook.draft.json",
        "terminology_fullbook.json",
    ),
    (
        "scene_arcs",
        "synthesis/scene_arcs.draft.json",
        "scene_arcs.json",
    ),
    (
        "lore_terminology_synthesis_audit",
        "synthesis/lore_terminology_synthesis_audit.json",
        "lore_terminology_synthesis_audit.json",
    ),
    (
        "deterministic_asset_summary",
        "deterministic-assets/deterministic_asset_summary.json",
        "deterministic_asset_summary.json",
    ),
    (
        "reference_job_manifest",
        "reference-job/manifest.json",
        "reference_job_manifest.json",
    ),
)

GZIP_SPECS = (
    (
        "chapter_reviews",
        "manual-aggregate/chapter_reviews.jsonl",
        "chapter_reviews.jsonl.gz",
    ),
    (
        "entity_arcs",
        "synthesis/entity_arcs.final.json",
        "entity_arcs.json.gz",
    ),
    (
        "relationship_arcs",
        "synthesis/relationship_arcs.final.json",
        "relationship_arcs.json.gz",
    ),
    (
        "entity_alias_map",
        "synthesis/entity_alias_map.final.json",
        "entity_alias_map.json.gz",
    ),
    (
        "terminology_evidence",
        "deterministic-assets/terminology_evidence.json",
        "terminology_evidence.json.gz",
    ),
    (
        "style_statistics",
        "deterministic-assets/style_statistics.json",
        "style_statistics.json.gz",
    ),
    (
        "chunk_alias_map",
        "deterministic-assets/chunk_alias_map.jsonl",
        "chunk_alias_map.jsonl.gz",
    ),
    (
        "source_chapter_index",
        "reference-job/chapters.jsonl",
        "source_chapter_index.jsonl.gz",
    ),
    (
        "reference_chunks",
        "reference-job/chunks.jsonl",
        "reference_chunks.jsonl.gz",
    ),
)

AUDIT_IDS = {
    "entity_synthesis_audit",
    "entities_runtime_build_audit",
    "entities_runtime_independent_audit",
    "voice_style_synthesis_audit",
    "style_runtime_build_audit",
    "lore_terminology_synthesis_audit",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _atomic_gzip(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            shutil.copyfileobj(source_handle, compressed, length=1024 * 1024)
    temporary.replace(destination)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象: {path}")
    return value


def _numeric_validation_errors(value: Any, trail: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{trail}.{key}" if trail else key
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and (
                    key in {"validation_error_count", "final_error_count"}
                    or key.endswith("_parse_errors")
                    or key.endswith("_boundary_errors")
                )
                and item != 0
            ):
                errors.append(f"{current}={item}")
            errors.extend(_numeric_validation_errors(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_numeric_validation_errors(item, f"{trail}[{index}]"))
    return errors


def _validate_audit(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("status") != "final":
        raise ValueError(f"audit 尚未 final: {path}: {value.get('status')!r}")
    errors = _numeric_validation_errors(value)
    if errors:
        raise ValueError(f"audit validation 未清零: {path}: {', '.join(errors)}")
    return value


def _write_knowledge_archive(work_dir: Path, destination: Path) -> int:
    files = sorted((work_dir / "manual-parts").glob("knowledge_*.json"))
    if len(files) != 18:
        raise ValueError(f"knowledge parts 应为 18，实际 {len(files)}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source in files:
                    info = archive.gettarinfo(str(source), arcname=source.name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as source_handle:
                        archive.addfile(info, source_handle)
    temporary.replace(destination)
    return len(files)


def _asset_record(asset_id: str, path: Path, project_dir: Path) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "path": path.relative_to(project_dir).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def publish(work_dir: Path, project_dir: Path, generated_at: str) -> dict[str, Any]:
    output_dir = project_dir / "sources" / "canonical" / "reference-review"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_audit_path = work_dir / "manual-aggregate" / "audit.json"
    aggregate_audit = _load_json(aggregate_audit_path)
    if (
        aggregate_audit.get("review_method") != "codex_subagent_full_chapter_read"
        or aggregate_audit.get("source_chapters") != 1428
        or aggregate_audit.get("reviewed_chapters") != 1428
        or aggregate_audit.get("coverage_percent") != 100.0
        or aggregate_audit.get("missing_chapters") != []
        or aggregate_audit.get("validation_error_count") != 0
    ):
        raise ValueError("全书审阅 audit 未达到 Codex subagent 1428/1428 严格完成门槛")

    runtime_registry_path = work_dir / "synthesis" / "entities.runtime.draft.json"
    runtime_registry = _load_json(runtime_registry_path)
    entities = runtime_registry.get("entities")
    relations = runtime_registry.get("relations")
    if (
        runtime_registry.get("schema") != "aipe.entities"
        or not isinstance(entities, list)
        or not isinstance(relations, list)
    ):
        raise ValueError("runtime entity registry 结构无效")

    independent_audit_path = (
        work_dir / "synthesis" / "entities_runtime_independent_audit.json"
    )
    independent_audit = _validate_audit(independent_audit_path)
    audited_registry = (
        independent_audit.get("input_snapshots", {}).get("registry", {})
    )
    coverage = independent_audit.get("coverage", {})
    summary = independent_audit.get("summary", {})
    runtime_registry_sha256 = _sha256(runtime_registry_path)
    if audited_registry.get("sha256") != runtime_registry_sha256:
        raise ValueError("独立二审绑定的 registry SHA 与待发布快照不一致")
    if (
        coverage.get("entities_expected") != len(entities)
        or coverage.get("entities_reviewed") != len(entities)
        or coverage.get("relations_expected") != len(relations)
        or coverage.get("relations_reviewed") != len(relations)
        or summary.get("coverage_complete") is not True
    ):
        raise ValueError("独立二审未完整覆盖待发布 entity registry")
    if (
        summary.get("validation_error_count") != 0
        or summary.get("finding_error_count") != 0
        or summary.get("runtime_readiness") != "ready_for_runtime"
    ):
        raise ValueError("独立二审尚未清零语义错误并批准 runtime 使用")

    records: list[dict[str, Any]] = []
    audit_statuses: dict[str, str] = {}
    for asset_id, relative_source, output_name in COPY_SPECS:
        source = work_dir / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        if asset_id in AUDIT_IDS:
            audit_statuses[asset_id] = str(_validate_audit(source).get("status"))
        destination = output_dir / output_name
        _atomic_copy(source, destination)
        records.append(_asset_record(asset_id, destination, project_dir))

    for asset_id, relative_source, output_name in GZIP_SPECS:
        source = work_dir / relative_source
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / output_name
        _atomic_gzip(source, destination)
        record = _asset_record(asset_id, destination, project_dir)
        record["source_sha256"] = _sha256(source)
        record["source_bytes"] = source.stat().st_size
        records.append(record)

    knowledge_archive = output_dir / "knowledge_parts.tar.gz"
    knowledge_parts = _write_knowledge_archive(work_dir, knowledge_archive)
    knowledge_record = _asset_record("knowledge_parts", knowledge_archive, project_dir)
    knowledge_record["members"] = knowledge_parts
    records.append(knowledge_record)

    records.sort(key=lambda item: item["asset_id"])
    manifest = {
        "schema": "aipe.full-reference-manifest",
        "version": 1,
        "project": "lom/zh-en",
        "generated_at": generated_at,
        "distribution": "internal_only",
        "execution": {
            "review_method": "codex_subagent_full_chapter_read",
            "external_llm_api": False,
            "embedding_api": False,
            "web_network": False,
        },
        "coverage": {
            "source_chapters": aggregate_audit["source_chapters"],
            "reviewed_chapters": aggregate_audit["reviewed_chapters"],
            "coverage_percent": aggregate_audit["coverage_percent"],
            "missing_chapters": aggregate_audit["missing_chapters"],
            "validation_error_count": aggregate_audit["validation_error_count"],
            "knowledge_parts": knowledge_parts,
        },
        "record_counts": aggregate_audit["record_counts"],
        "audit_statuses": audit_statuses,
        "runtime_boundary": {
            "archive_loaded_into_translation_prompt": False,
            "reviewed_runtime_entity_sha256": runtime_registry_sha256,
            "runtime_assets": [
                "sources/canonical/entities.json",
                "sources/canonical/scenes.json",
                "sources/canonical/context_rules.json",
            ],
            "reference_retrieval": "local sparse Qdrant; anchor-gated; bounded excerpts",
        },
        "assets": records,
    }
    manifest_path = output_dir / "full_reference_manifest.json"
    _atomic_json(manifest_path, manifest)
    canonical_entities_path = (
        project_dir / "sources" / "canonical" / "entities.json"
    )
    _atomic_copy(runtime_registry_path, canonical_entities_path)
    return {
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "canonical_entities": canonical_entities_path.as_posix(),
        "canonical_entities_sha256": _sha256(canonical_entities_path),
        "asset_count": len(records),
        "coverage": manifest["coverage"],
        "audit_statuses": audit_statuses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument(
        "--generated-at",
        default=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    args = parser.parse_args()
    result = publish(args.work_dir.resolve(), args.project_dir.resolve(), args.generated_at)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
