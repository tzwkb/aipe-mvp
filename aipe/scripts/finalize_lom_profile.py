"""Finalize the LOM AIPE profile from the independently audited local archive."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ID = "lom/zh-en"
REFERENCE_ASSETS = {
    "full_reference_manifest": (
        "full_reference_manifest",
        "sources/canonical/reference-review/full_reference_manifest.json",
        "application/json",
        True,
    ),
    "full_chapter_reviews": (
        "chapter_review_archive",
        "sources/canonical/reference-review/chapter_reviews.jsonl.gz",
        "application/gzip",
        False,
    ),
    "full_knowledge_parts": (
        "knowledge_archive",
        "sources/canonical/reference-review/knowledge_parts.tar.gz",
        "application/gzip",
        False,
    ),
    "full_entity_arcs": (
        "entity_arc_archive",
        "sources/canonical/reference-review/entity_arcs.json.gz",
        "application/gzip",
        False,
    ),
    "full_relationship_arcs": (
        "relationship_arc_archive",
        "sources/canonical/reference-review/relationship_arcs.json.gz",
        "application/gzip",
        False,
    ),
    "full_entity_alias_map": (
        "entity_alias_archive",
        "sources/canonical/reference-review/entity_alias_map.json.gz",
        "application/gzip",
        False,
    ),
    "full_voice_profiles": (
        "voice_profile_registry",
        "sources/canonical/reference-review/voice_profiles.json",
        "application/json",
        False,
    ),
    "full_narrative_style_patterns": (
        "narrative_style_registry",
        "sources/canonical/reference-review/narrative_style_patterns.json",
        "application/json",
        False,
    ),
    "full_lore_systems": (
        "lore_system_registry",
        "sources/canonical/reference-review/lore_systems.json",
        "application/json",
        False,
    ),
    "full_terminology": (
        "fullbook_terminology_registry",
        "sources/canonical/reference-review/terminology_fullbook.json",
        "application/json",
        False,
    ),
    "full_scene_arcs": (
        "scene_arc_registry",
        "sources/canonical/reference-review/scene_arcs.json",
        "application/json",
        False,
    ),
    "full_reference_review_audit": (
        "full_reference_review_audit",
        "sources/canonical/reference-review/review_audit.json",
        "application/json",
        True,
    ),
    "runtime_entity_build_audit": (
        "runtime_entity_build_audit",
        "sources/canonical/reference-review/entities_runtime_build_audit.json",
        "application/json",
        True,
    ),
    "runtime_entity_independent_audit": (
        "runtime_entity_independent_audit",
        "sources/canonical/reference-review/entities_runtime_independent_audit.json",
        "application/json",
        True,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON top level is not an object: {path}")
    return value


def write_object(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_gate(project_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    review_dir = project_dir / "sources" / "canonical" / "reference-review"
    manifest = load_object(review_dir / "full_reference_manifest.json")
    if manifest.get("project") != PROJECT_ID:
        raise ValueError("full reference manifest project mismatch")
    coverage = manifest.get("coverage", {})
    if (
        coverage.get("source_chapters") != 1428
        or coverage.get("reviewed_chapters") != 1428
        or coverage.get("coverage_percent") != 100.0
        or coverage.get("missing_chapters") != []
        or coverage.get("validation_error_count") != 0
    ):
        raise ValueError("full reference coverage gate failed")
    execution = manifest.get("execution", {})
    if execution != {
        "review_method": "codex_subagent_full_chapter_read",
        "external_llm_api": False,
        "embedding_api": False,
        "web_network": False,
    }:
        raise ValueError("full reference execution provenance gate failed")

    entities_path = project_dir / "sources" / "canonical" / "entities.json"
    entities = load_object(entities_path)
    entity_count = len(entities.get("entities", []))
    relation_count = len(entities.get("relations", []))
    fact_count = sum(
        len(entity.get("facts", [])) for entity in entities.get("entities", [])
    )
    runtime_boundary = manifest.get("runtime_boundary", {})
    if (
        sha256(entities_path)
        != runtime_boundary.get("reviewed_runtime_entity_sha256")
        or entity_count != 193
        or relation_count != 102
        or fact_count != 660
    ):
        raise ValueError("canonical runtime entity gate failed")

    independent = load_object(
        review_dir / "entities_runtime_independent_audit.json"
    )
    summary = independent.get("summary", {})
    independent_coverage = independent.get("coverage", {})
    if (
        independent.get("status") != "final"
        or summary.get("runtime_readiness") != "ready_for_runtime"
        or summary.get("finding_error_count") != 0
        or summary.get("unresolved_error_count") != 0
        or summary.get("validation_error_count") != 0
        or summary.get("coverage_complete") is not True
        or independent_coverage.get("entities_reviewed") != entity_count
        or independent_coverage.get("relations_reviewed") != relation_count
        or independent_coverage.get("facts_checked") != fact_count
    ):
        raise ValueError("independent runtime entity audit gate failed")
    return manifest, entities


def reference_asset(
    project_dir: Path,
    *,
    kind: str,
    relative_path: str,
    media_type: str,
    required: bool,
) -> dict[str, Any]:
    path = project_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "kind": kind,
        "path": relative_path,
        "required": required,
        "authority": {
            "issuer": "profile_maintainer",
            "level": "derived_from_complete_client_reference_review",
        },
        "provenance": {
            "kind": "codex_subagent_full_chapter_read",
            "sources": ["raw/Lord.txt", "raw/诡秘之主术语表.xlsx"],
            "sha256": sha256(path),
            "runtime_use": False,
        },
        "distribution": "internal_only",
        "availability": "included",
        "media_type": media_type,
    }


def finalize_profile(
    project_dir: Path,
    manifest: dict[str, Any],
    entities: dict[str, Any],
) -> None:
    profile_path = project_dir / "profile.json"
    profile = load_object(profile_path)
    profile["profile_status"] = "ready_with_gaps"
    profile["authority_policy"] = [
        "client_locked_terminology",
        "client_trial_workbook",
        "client_reference_corpus",
        "profile_maintainer",
        "local_sparse_reference_retrieval",
    ]
    profile["terminology_status"].update(
        {
            "unique_source_target_category_records": 404,
            "corpus_attested_records": 400,
            "corpus_unattested_records": 4,
            "corpus_surface_occurrences": 118957,
            "fullbook_asset": "sources/canonical/reference-review/terminology_fullbook.json",
        }
    )

    assets = profile.setdefault("assets", {})
    for asset_id, (kind, relative_path, media_type, required) in REFERENCE_ASSETS.items():
        assets[asset_id] = reference_asset(
            project_dir,
            kind=kind,
            relative_path=relative_path,
            media_type=media_type,
            required=required,
        )

    entity_count = len(entities["entities"])
    relation_count = len(entities["relations"])
    fact_count = sum(len(entity.get("facts", [])) for entity in entities["entities"])
    assets["entities"]["provenance"].update(
        {
            "runtime_counts": {
                "entities": entity_count,
                "relations": relation_count,
                "facts": fact_count,
            },
            "independent_audit": (
                "sources/canonical/reference-review/"
                "entities_runtime_independent_audit.json"
            ),
            "sha256": sha256(
                project_dir / "sources" / "canonical" / "entities.json"
            ),
        }
    )
    assets["lord_reference"]["provenance"].update(
        {
            "direct_prompt_use": False,
            "full_chapter_prompt_use": False,
            "full_review_manifest": (
                "sources/canonical/reference-review/full_reference_manifest.json"
            ),
        }
    )

    profile.setdefault("content_scope", {})["full_reference_review"] = {
        "review_method": "codex_subagent_full_chapter_read",
        "execution": {
            "local_only": True,
            "external_llm_api": False,
            "embedding_api": False,
            "web_network": False,
        },
        "source_chapters": 1428,
        "reviewed_chapters": 1428,
        "coverage_percent": 100.0,
        "missing_chapters": [],
        "validation_error_count": 0,
        "record_counts": manifest["record_counts"],
        "synthesis_counts": {
            "entities": 903,
            "relationship_arcs": 2320,
            "reusable_aliases": 1396,
            "lore_rules": 72,
            "objects_and_abilities": 34,
            "book_phases": 10,
            "scene_arcs": 20,
            "narrative_modes": 15,
        },
        "runtime_entity_counts": {
            "entities": entity_count,
            "relations": relation_count,
            "facts": fact_count,
        },
        "archive_manifest": (
            "sources/canonical/reference-review/full_reference_manifest.json"
        ),
        "archive_loaded_into_translation_prompt": False,
    }

    workflow = profile.setdefault("workflow", {})
    workflow.update(
        {
            "rag_collection_status": "bilingual_tm_declared_not_ingested",
            "reference_collection_status": (
                "ingested_and_verified_sparse_local_4243_points"
            ),
            "current_mtpe_execution": {
                "engine": "codex_subagents",
                "local_files_only": True,
                "external_llm_api": False,
                "embedding_api": False,
                "full_chapter_prompting": False,
            },
            "full_reference_archive_prompt_use": False,
        }
    )

    for gap in profile.get("gaps", []):
        if gap.get("id") == "gap.lom.full_reference_review":
            gap.update(
                {
                    "severity": "resolved",
                    "status": "resolved",
                    "impact": (
                        "已由 Codex subagent 本地完成 1428/1428 章全量通读、"
                        "结构化综合与独立实体语义二审；runtime 仅加载有界资产。"
                    ),
                }
            )
        elif gap.get("id") == "gap.lom.source_aliases":
            gap.update(
                {
                    "status": "partial_nonblocking",
                    "expected": "所有客户资料中可证实的中文名称与别名",
                    "impact": (
                        "客户术语表与中文试译中可证实的 source-target 配对均已绑定；"
                        "仅见于英文 Lord.txt 的实体仍保留 target-only 名称，未凭记忆补造中文名。"
                    ),
                }
            )
        elif gap.get("id") == "gap.lom.bilingual_tm_ingestion":
            gap["impact"] = (
                "双语 TM collection 尚未灌入且不是本次 MTPE 的前置条件；"
                "Lord.txt 已由独立本地 sparse Qdrant collection 提供受锚点限制的短摘录。"
            )

    write_object(profile_path, profile)


def finalize_source_manifest(
    project_dir: Path,
    manifest: dict[str, Any],
    entities: dict[str, Any],
) -> None:
    path = project_dir / "sources" / "canonical" / "project_sources.json"
    source_manifest = load_object(path)
    source_manifest["generated_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    for source in source_manifest.get("sources", []):
        if source.get("source_id") == "source.lom.lord_txt.v1":
            source["verification_status"] = "verified_full_chapter_review"
            source["full_review"] = {
                "method": "codex_subagent_full_chapter_read",
                "chapters": 1428,
                "coverage_percent": 100.0,
                "validation_error_count": 0,
                "manifest": (
                    "sources/canonical/reference-review/full_reference_manifest.json"
                ),
                "external_llm_api": False,
                "embedding_api": False,
                "web_network": False,
            }

    generated_assets = source_manifest.setdefault("generated_assets", [])
    by_id = {item.get("asset_id"): item for item in generated_assets}
    for item in generated_assets:
        relative_path = item.get("path")
        if not isinstance(relative_path, str):
            continue
        asset_path = project_dir / relative_path
        if asset_path.is_file():
            item["sha256"] = sha256(asset_path)

    entities_record = by_id.get("entities")
    if entities_record is not None:
        entities_record["counts"] = {
            "entities": len(entities["entities"]),
            "relations": len(entities["relations"]),
            "facts": sum(
                len(entity.get("facts", [])) for entity in entities["entities"]
            ),
        }
        entities_record["generator"] = {
            "name": "codex_subagent_fullbook_runtime_entity_synthesis",
            "version": "2",
            "independent_audit": "ready_for_runtime",
        }

    full_manifest_path = (
        project_dir
        / "sources"
        / "canonical"
        / "reference-review"
        / "full_reference_manifest.json"
    )
    full_manifest_record = {
        "asset_id": "full_reference_manifest",
        "path": "sources/canonical/reference-review/full_reference_manifest.json",
        "sha256": sha256(full_manifest_path),
        "derived_from": [
            "source.lom.lord_txt.v1",
            "source.lom.client_terminology.v1",
        ],
        "generator": {
            "name": "codex_subagent_full_chapter_read",
            "version": "2026-08-27",
            "asset_count": len(manifest["assets"]),
            "external_llm_api": False,
            "embedding_api": False,
        },
    }
    if "full_reference_manifest" in by_id:
        by_id["full_reference_manifest"].clear()
        by_id["full_reference_manifest"].update(full_manifest_record)
    else:
        generated_assets.append(full_manifest_record)

    coverage = source_manifest.setdefault("coverage", {})
    coverage["reference_corpus"] = {
        "source_chapters": 1428,
        "reviewed_chapters": 1428,
        "coverage_percent": 100.0,
        "missing_chapters": [],
        "validation_error_count": 0,
        "record_counts": manifest["record_counts"],
        "synthesized_entities": 903,
        "synthesized_relationship_arcs": 2320,
        "synthesized_reusable_aliases": 1396,
        "runtime_entities": len(entities["entities"]),
        "runtime_relations": len(entities["relations"]),
        "runtime_facts": sum(
            len(entity.get("facts", [])) for entity in entities["entities"]
        ),
        "canonical_scenes": 1,
        "local_sparse_qdrant_points": 4243,
    }
    coverage["source_language_name_coverage"] = {
        "status": "complete_for_available_client_sources",
        "reason": (
            "All source-target pairs attested in the client termbase and Chinese trial workbook "
            "are bound. Lord-only English entities remain target-only because no Chinese source "
            "form is present; no names were invented."
        ),
    }
    coverage["terminology"].update(
        {
            "unique_source_target_category_records": 404,
            "corpus_attested_records": 400,
            "corpus_unattested_records": 4,
            "corpus_surface_occurrences": 118957,
        }
    )
    write_object(path, source_manifest)


def main() -> int:
    project_dir = (
        Path(__file__).resolve().parents[1] / "data" / "projects" / "lom" / "zh-en"
    )
    manifest, entities = validate_gate(project_dir)
    finalize_profile(project_dir, manifest, entities)
    finalize_source_manifest(project_dir, manifest, entities)
    print(
        json.dumps(
            {
                "profile": str(project_dir / "profile.json"),
                "profile_sha256": sha256(project_dir / "profile.json"),
                "project_sources_sha256": sha256(
                    project_dir / "sources" / "canonical" / "project_sources.json"
                ),
                "status": "ready_with_gaps",
                "runtime_entities": 193,
                "runtime_relations": 102,
                "runtime_facts": 660,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
