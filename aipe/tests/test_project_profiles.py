from __future__ import annotations

import json

import pytest

from app.services.project_service import ProjectProfileError, ProjectRegistry, ProjectResourceManager


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
    _write_project(projects, "wwm/zh-en", term_source="燕云", term_target="Where Winds Meet", style="WWM style", collection="wwm_corpus")

    registry = ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en")
    profile = registry.get(None)

    assert profile.name == "wwm/zh-en"
    assert profile.source_lang == "zh"
    assert profile.target_lang == "en"
    assert profile.qdrant_collection == "wwm_corpus"
    assert profile.web_search_prefix == "wwm/zh-en"
    assert profile.style_guide_path.name == "style.md"
    assert profile.terminology_path.name == "terms.json"
    assert profile.style_guide_path.read_text(encoding="utf-8") == "WWM style"


def test_project_resources_keep_terminology_and_style_isolated(tmp_path):
    projects = tmp_path / "projects"
    _write_project(projects, "wwm/zh-en", term_source="燕云", term_target="Where Winds Meet", style="WWM style", collection="wwm_corpus")
    _write_project(projects, "nrc/zh-en", term_source="洛克", term_target="Roco", style="NRC style", collection="nrc_zh_en_corpus")
    manager = ProjectResourceManager(ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en"))

    wwm_terms = manager.terminology("wwm/zh-en")
    nrc_terms = manager.terminology("nrc/zh-en")
    wwm_style = manager.style_guide("wwm/zh-en")
    nrc_style = manager.style_guide("nrc/zh-en")

    assert [m.target for m in wwm_terms.find_matches("燕云江湖")] == ["Where Winds Meet"]
    assert nrc_terms.find_matches("燕云江湖") == []
    assert [m.target for m in nrc_terms.find_matches("洛克王国")] == ["Roco"]
    assert wwm_style.get_rules() == "WWM style"
    assert nrc_style.get_rules() == "NRC style"


def test_project_resource_replacements_persist_to_profile_assets(tmp_path):
    projects = tmp_path / "projects"
    _write_project(projects, "wwm/zh-en", term_source="燕云", term_target="Where Winds Meet", style="WWM style", collection="wwm_corpus")
    manager = ProjectResourceManager(ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en"))

    manager.replace_terminology(
        "wwm/zh-en",
        [{"source": "问剑", "target": "Ask the Sword", "category": "skill"}],
    )
    manager.replace_style_guide("wwm/zh-en", "Persisted style", filename="uploaded.md")

    restarted = ProjectResourceManager(ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en"))
    terms = restarted.terminology("wwm/zh-en")
    style = restarted.style_guide("wwm/zh-en")

    assert [m.target for m in terms.find_matches("问剑")] == ["Ask the Sword"]
    assert style.get_rules() == "Persisted style"


def test_registry_rejects_unknown_project(tmp_path):
    registry = ProjectRegistry(projects_dir=tmp_path / "projects", default_project="wwm/zh-en")
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
