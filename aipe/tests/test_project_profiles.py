from __future__ import annotations

import json

import pytest

from app.services.project_context_service import ProjectContextError
from app.services.project_service import (
    ProjectProfileError,
    ProjectRegistry,
    ProjectResourceManager,
)


def _write_project(
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
        json.dumps(
            [{"source": term_source, "target": term_target}], ensure_ascii=False
        ),
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
                "background": f"{name} background",
                "style_guide": "style.md",
                "terminology": "terms.json",
                "qdrant_collection": collection,
                "web_search_prefix": name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_registry_resolves_default_and_relative_assets(tmp_path):
    projects = tmp_path / "projects"
    _write_project(
        projects,
        "wwm/zh-en",
        term_source="燕云",
        term_target="Where Winds Meet",
        style="WWM style",
        collection="wwm_corpus",
    )

    registry = ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en")
    profile = registry.get(None)

    assert profile.name == "wwm/zh-en"
    assert profile.source_lang == "zh"
    assert profile.target_lang == "en"
    assert profile.qdrant_collection == "wwm_corpus"
    assert profile.reference_qdrant_collection is None
    assert profile.web_search_prefix == "wwm/zh-en"
    assert profile.style_guide_path.name == "style.md"
    assert profile.terminology_path.name == "terms.json"
    assert profile.style_guide_path.read_text(encoding="utf-8") == "WWM style"


def test_registry_loads_separate_monolingual_reference_collection(tmp_path):
    projects = tmp_path / "projects"
    _write_project(
        projects,
        "lom/zh-en",
        term_source="克莱恩",
        term_target="Klein",
        style="LOM style",
        collection="lom_zh_en_corpus",
    )
    profile_path = projects / "lom/zh-en/profile.json"
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["reference_qdrant_collection"] = "lom_reference_en_sparse_v1"
    raw["reference_context"] = {"scene_chapter_map": {"scene.lom.chapter34": [34]}}
    profile_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    profile = ProjectRegistry(projects_dir=projects, default_project="lom/zh-en").get()

    assert profile.qdrant_collection == "lom_zh_en_corpus"
    assert profile.reference_qdrant_collection == "lom_reference_en_sparse_v1"
    assert profile.reference_context["scene_chapter_map"]["scene.lom.chapter34"] == [34]


def test_project_resources_keep_terminology_and_style_isolated(tmp_path):
    projects = tmp_path / "projects"
    _write_project(
        projects,
        "wwm/zh-en",
        term_source="燕云",
        term_target="Where Winds Meet",
        style="WWM style",
        collection="wwm_corpus",
    )
    _write_project(
        projects,
        "nrc/zh-en",
        term_source="洛克",
        term_target="Roco",
        style="NRC style",
        collection="nrc_zh_en_corpus",
    )
    manager = ProjectResourceManager(
        ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en")
    )

    wwm_terms = manager.terminology("wwm/zh-en")
    nrc_terms = manager.terminology("nrc/zh-en")
    wwm_style = manager.style_guide("wwm/zh-en")
    nrc_style = manager.style_guide("nrc/zh-en")

    assert [m.target for m in wwm_terms.find_matches("燕云江湖")] == [
        "Where Winds Meet"
    ]
    assert nrc_terms.find_matches("燕云江湖") == []
    assert [m.target for m in nrc_terms.find_matches("洛克王国")] == ["Roco"]
    assert wwm_style.get_rules() == "WWM style"
    assert nrc_style.get_rules() == "NRC style"


def test_project_resource_replacements_persist_to_profile_assets(tmp_path):
    projects = tmp_path / "projects"
    _write_project(
        projects,
        "wwm/zh-en",
        term_source="燕云",
        term_target="Where Winds Meet",
        style="WWM style",
        collection="wwm_corpus",
    )
    manager = ProjectResourceManager(
        ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en")
    )

    manager.replace_terminology(
        "wwm/zh-en",
        [{"source": "问剑", "target": "Ask the Sword", "category": "skill"}],
    )
    manager.replace_style_guide("wwm/zh-en", "Persisted style", filename="uploaded.md")

    restarted = ProjectResourceManager(
        ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en")
    )
    terms = restarted.terminology("wwm/zh-en")
    style = restarted.style_guide("wwm/zh-en")

    assert [m.target for m in terms.find_matches("问剑")] == ["Ask the Sword"]
    assert style.get_rules() == "Persisted style"


def test_registry_rejects_unknown_project(tmp_path):
    registry = ProjectRegistry(
        projects_dir=tmp_path / "projects", default_project="wwm/zh-en"
    )
    with pytest.raises(ProjectProfileError):
        registry.get("missing/en")


def test_registry_maps_legacy_target_only_ids_to_language_pair_profiles(tmp_path):
    projects = tmp_path / "projects"
    _write_project(
        projects,
        "isekai/en-de",
        language_pair="EN-DE",
        source_lang="en",
        target_lang="de",
        term_source="Village Elder",
        term_target="Dorfältester",
        style="DE style",
        collection="isekai_de_corpus",
    )
    registry = ProjectRegistry(projects_dir=projects, default_project="isekai/de")

    profile = registry.get(None)
    alias_profile = registry.get("isekai/de")

    assert profile is alias_profile
    assert profile.name == "isekai/en-de"
    assert profile.source_lang == "en"
    assert profile.target_lang == "de"


def _write_context_project(root, *, mode: str = "enforce") -> None:
    project_dir = root / "demo" / "zh-en"
    canonical_dir = project_dir / "sources" / "canonical"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "entities.json").write_text(
        json.dumps(
            {
                "schema": "aipe.entities",
                "version": 1,
                "entities": [
                    {
                        "id": "entity.alice",
                        "entity_type": "character",
                        "names": {"source": ["甲", "甲的化名"], "target": ["Alice"]},
                        "facts": [
                            {
                                "text": "甲说话冷静。",
                                "verification_status": "source_backed",
                                "provenance": {
                                    "source_id": "source.demo",
                                    "coordinate": "L1",
                                },
                            }
                        ],
                    },
                    {
                        "id": "entity.bob",
                        "entity_type": "character",
                        "names": {"source": ["乙"], "target": ["Bob"]},
                        "facts": [
                            {
                                "text": "乙是甲的联络人。",
                                "verification_status": "source_backed",
                                "provenance": {
                                    "source_id": "source.demo",
                                    "coordinate": "L2",
                                },
                            }
                        ],
                    },
                    {
                        "id": "entity.carol",
                        "entity_type": "character",
                        "names": {"source": ["丙"], "target": ["Carol"]},
                        "facts": [
                            {
                                "text": "不相关的秘密事实。",
                                "verification_status": "source_backed",
                                "provenance": {
                                    "source_id": "source.demo",
                                    "coordinate": "L3",
                                },
                            }
                        ],
                    },
                ],
                "relations": [
                    {
                        "id": "relation.alice.bob",
                        "from": "entity.alice",
                        "to": "entity.bob",
                        "relation_type": "trusted_contact_of",
                        "attributes": {"relationship_stage": "early"},
                        "verification_status": "source_backed",
                        "provenance": {"source_id": "source.demo", "coordinate": "L4"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (canonical_dir / "scenes.json").write_text(
        json.dumps(
            {
                "schema": "aipe.scenes",
                "version": 1,
                "scenes": [
                    {
                        "id": "scene.demo.handoff",
                        "title": "Secret handoff",
                        "participants": ["entity.alice", "entity.bob"],
                        "relationship_stage": "early",
                        "scene_tone": "tense",
                        "triggers": ["密令"],
                        "facts": ["甲向乙交付密令。"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (canonical_dir / "context_rules.json").write_text(
        json.dumps(
            {
                "schema": "aipe.context-rules",
                "version": 1,
                "rules": [
                    {
                        "id": "rule.alice.voice",
                        "target_lang": "en",
                        "rule_status": "draft",
                        "priority": 90,
                        "when": {"speaker_entity_ids": ["entity.alice"]},
                        "expect": {"voice": "calm Alice voice"},
                        "provenance": {"source_id": "source.demo", "coordinate": "L1"},
                    },
                    {
                        "id": "rule.alice.alias.voice",
                        "target_lang": "en",
                        "rule_status": "draft",
                        "priority": 95,
                        "when": {
                            "speaker_entity_ids": ["entity.alice"],
                            "contains_any": ["甲的化名"],
                        },
                        "expect": {"voice": "regal alias voice"},
                        "provenance": {"source_id": "source.demo", "coordinate": "L1"},
                    },
                    {
                        "id": "rule.bob.voice",
                        "target_lang": "en",
                        "rule_status": "draft",
                        "priority": 89,
                        "when": {"speaker_entity_ids": ["entity.bob"]},
                        "expect": {"voice": "rough Bob voice"},
                        "provenance": {"source_id": "source.demo", "coordinate": "L2"},
                    },
                    {
                        "id": "rule.stage.early",
                        "target_lang": "en",
                        "rule_status": "confirmed",
                        "priority": 80,
                        "when": {
                            "relationship_stages": ["early"],
                            "scene_tones": ["tense"],
                        },
                        "expect": {"register": "guarded"},
                        "provenance": {"source_id": "source.demo", "coordinate": "L4"},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assets = {
        "entities": {
            "kind": "entity_registry",
            "path": "sources/canonical/entities.json",
            "required": True,
            "availability": "included",
        },
        "scenes": {
            "kind": "scene_registry",
            "path": "sources/canonical/scenes.json",
            "required": True,
            "availability": "included",
        },
        "rules": {
            "kind": "context_rules",
            "path": "sources/canonical/context_rules.json",
            "required": True,
            "availability": "included",
        },
        "missing_terms": {
            "kind": "terminology",
            "path": "raw/client-terms.xlsx",
            "required": True,
            "availability": "missing",
        },
    }
    capabilities = {
        "assets.entity_registry@1": {"required": True, "asset": "entities"},
        "assets.scene_registry@1": {"required": True, "asset": "scenes"},
        "language_policy.context_rules@1": {"required": True, "asset": "rules"},
        "terminology.client_locked@1": {"required": False, "asset": "missing_terms"},
    }
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "profile_contract_version": 2,
                "name": "demo/zh-en",
                "language_pair": "ZH-EN",
                "source_lang": "zh",
                "target_lang": "en",
                "game": "Demo",
                "profile_status": "ready_with_gaps",
                "authority_policy": ["client", "profile_maintainer"],
                "terminology_status": {"state": "missing_client_asset"},
                "assets": assets,
                "context_pipeline": {"mode": mode},
                "capabilities": capabilities,
                "module_context_views": {
                    "translation": {
                        "capabilities": [
                            "assets.entity_registry@1",
                            "assets.scene_registry@1",
                            "language_policy.context_rules@1",
                        ],
                        "limits": {
                            "max_entities": 2,
                            "max_facts_per_entity": 1,
                            "max_relations": 1,
                            "max_rules": 3,
                            "max_chars": 3000,
                        },
                    }
                },
                "content_scope": {"units": 2},
                "workflow": {"review": "human"},
                "gaps": [{"id": "gap.terms", "status": "missing"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_v2_profile_selects_only_relevant_context_and_honors_speaker_rules(tmp_path):
    projects = tmp_path / "projects"
    _write_context_project(projects)
    manager = ProjectResourceManager(ProjectRegistry(projects_dir=projects))

    profile = manager.profile("demo/zh-en")
    rendered = manager.project_context("demo/zh-en").render(
        ["把密令交给乙。"],
        content_type="话术",
        speakers=["甲"],
        addressees=["乙"],
        dialog_id="scene.demo.handoff",
    )

    assert profile.profile_contract_version == 2
    assert profile.profile_status == "ready_with_gaps"
    assert profile.context_pipeline_mode == "enforce"
    assert profile.authority_policy == ("client", "profile_maintainer")
    assert profile.terminology_status == {"state": "missing_client_asset"}
    assert profile.assets["missing_terms"].availability == "missing"
    assert profile.content_scope == {"units": 2}
    assert "Alice / 甲" in rendered and "Bob / 乙" in rendered
    assert "trusted_contact_of" in rendered
    assert "Secret handoff" in rendered
    assert "calm Alice voice" in rendered
    assert "rough Bob voice" not in rendered
    assert "register=guarded" in rendered
    assert "不相关的秘密事实" not in rendered


def test_v2_shadow_mode_validates_assets_without_injecting_context(tmp_path):
    projects = tmp_path / "projects"
    _write_context_project(projects, mode="shadow")
    manager = ProjectResourceManager(ProjectRegistry(projects_dir=projects))

    assert (
        manager.project_context("demo/zh-en").render(["甲拿到密令。"], speakers=["甲"])
        == ""
    )


def test_v2_context_rule_contains_any_matches_speaker_metadata_only(tmp_path):
    projects = tmp_path / "projects"
    _write_context_project(projects)
    manager = ProjectResourceManager(ProjectRegistry(projects_dir=projects))

    with_alias = manager.project_context("demo/zh-en").render(
        ["继续。"],
        speakers=["甲的化名"],
    )
    without_alias = manager.project_context("demo/zh-en").render(
        ["继续。"],
        speakers=["甲"],
    )

    assert "regal alias voice" in with_alias
    assert "regal alias voice" not in without_alias


def test_v2_profile_rejects_missing_included_asset(tmp_path):
    projects = tmp_path / "projects"
    _write_context_project(projects)
    profile_path = projects / "demo" / "zh-en" / "profile.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["assets"]["entities"]["path"] = "sources/canonical/not-there.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectProfileError, match="asset 不存在"):
        ProjectRegistry(projects_dir=projects).get("demo/zh-en")


def test_v2_context_rules_reject_unknown_when_condition(tmp_path):
    projects = tmp_path / "projects"
    _write_context_project(projects)
    rules_path = projects / "demo" / "zh-en" / "sources" / "canonical" / "context_rules.json"
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    payload["rules"][0]["when"]["persona_ids"] = ["persona.alice"]
    rules_path.write_text(json.dumps(payload), encoding="utf-8")
    manager = ProjectResourceManager(ProjectRegistry(projects_dir=projects))

    with pytest.raises(ProjectContextError, match="未知 when 条件.*persona_ids"):
        manager.project_context("demo/zh-en").render(["甲继续说。"], speakers=["甲"])
