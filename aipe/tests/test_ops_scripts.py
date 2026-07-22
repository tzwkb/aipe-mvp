from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_create_project_profile_writes_relative_assets(tmp_path):
    from scripts.create_project_profile import create_project_profile

    projects_dir = tmp_path / "projects"
    style = tmp_path / "data" / "style_guide" / "guide.md"
    terms = tmp_path / "data" / "terminology" / "terms.xlsx"
    style.parent.mkdir(parents=True)
    terms.parent.mkdir(parents=True)
    style.write_text("style", encoding="utf-8")
    terms.write_bytes(b"terms")

    profile_path = create_project_profile(
        projects_dir=projects_dir,
        project_id="game/zh-en",
        language_pair="ZH-EN",
        game="Game",
        background="Project background",
        style_guide=style,
        terminology=terms,
        qdrant_collection="game_en_corpus",
        web_search_prefix="Game CN",
        prompt_notes="Keep UI concise.",
        vision_system_prompt="vision prompt",
    )

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile_path == projects_dir / "game" / "zh-en" / "profile.json"
    assert payload["source_lang"] == "zh"
    assert payload["target_lang"] == "en"
    assert payload["style_guide"] == "../../../data/style_guide/guide.md"
    assert payload["terminology"] == "../../../data/terminology/terms.xlsx"
    assert payload["qdrant_collection"] == "game_en_corpus"
    assert payload["prompt_notes"] == "prompt_notes.md"
    assert (profile_path.parent / "prompt_notes.md").read_text(encoding="utf-8") == "Keep UI concise."


def test_create_project_profile_rejects_existing_without_force(tmp_path):
    from scripts.create_project_profile import create_project_profile

    projects_dir = tmp_path / "projects"
    project_dir = projects_dir / "game" / "zh-en"
    project_dir.mkdir(parents=True)
    (project_dir / "profile.json").write_text("{}", encoding="utf-8")

    try:
        create_project_profile(
            projects_dir=projects_dir,
            project_id="game/zh-en",
            language_pair="ZH-EN",
            game="Game",
            background="",
            style_guide=None,
            terminology=None,
            qdrant_collection=None,
            web_search_prefix=None,
            prompt_notes=None,
            vision_system_prompt=None,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")


def test_create_project_profile_requires_project_id_language_pair_suffix(tmp_path):
    from scripts.create_project_profile import create_project_profile

    try:
        create_project_profile(
            projects_dir=tmp_path / "projects",
            project_id="game/en",
            language_pair="ZH-EN",
            game="Game",
            background="",
            style_guide=None,
            terminology=None,
            qdrant_collection=None,
            web_search_prefix=None,
            prompt_notes=None,
            vision_system_prompt=None,
        )
    except ValueError as exc:
        assert "project_id" in str(exc)
        assert "zh-en" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cleanup_commands_do_not_include_volumes_by_default():
    from scripts.cleanup_dev_stack import build_cleanup_commands

    commands = build_cleanup_commands(include_build_cache=True, include_volumes=False)
    flattened = [" ".join(cmd) for cmd in commands]

    assert any("docker compose rm -f" in cmd for cmd in flattened)
    assert any("docker image prune -f" in cmd for cmd in flattened)
    assert any("docker builder prune -f" in cmd for cmd in flattened)
    assert not any("--volumes" in cmd for cmd in flattened)


def test_cleanup_commands_require_explicit_volumes():
    from scripts.cleanup_dev_stack import build_cleanup_commands

    commands = build_cleanup_commands(include_build_cache=False, include_volumes=True)
    flattened = [" ".join(cmd) for cmd in commands]

    assert any("docker compose down --volumes --remove-orphans" in cmd for cmd in flattened)


def test_run_checks_builds_docker_command_for_selected_tests():
    from scripts.run_checks import build_docker_check_command

    cmd = build_docker_check_command(Path("/repo/aipe"), ["tests/test_project_profiles.py"])
    assert cmd[:4] == ["docker", "compose", "run", "--rm"]
    assert "/repo/aipe:/work" in cmd
    assert cmd[-2:] == ["-c", "python -m pip install pytest -q && python -m compileall -q app scripts tests && python -m pytest -q tests/test_project_profiles.py"]


def test_batch_translate_defaults_to_project_profile_collection():
    from scripts.batch_translate_files import build_translate_params

    args = SimpleNamespace(
        batch_size=20,
        project_id="wwm/zh-en",
        rag_collection=None,
        rag_threshold=0.85,
        rag_top_k=3,
        enable_rag=True,
        enable_cluster=True,
        dialog_mode=False,
        enable_web_search=True,
        web_search_dense_threshold=0.85,
        enable_vision=False,
        use_tm_exact_match=False,
    )

    params = build_translate_params(args, task_id="task_1")

    assert params["project_id"] == "wwm/zh-en"
    assert params["task_id"] == "task_1"
    assert params["use_tm_exact_match"] == "false"
    assert "rag_collection" not in params


def test_batch_translate_allows_explicit_collection_override():
    from scripts.batch_translate_files import build_translate_params

    args = SimpleNamespace(
        batch_size=20,
        project_id="wwm/zh-en",
        rag_collection="custom_corpus",
        rag_threshold=0.85,
        rag_top_k=3,
        enable_rag=True,
        enable_cluster=True,
        dialog_mode=False,
        enable_web_search=True,
        web_search_dense_threshold=0.85,
        enable_vision=False,
        use_tm_exact_match=True,
    )

    params = build_translate_params(args, task_id="task_1")

    assert params["rag_collection"] == "custom_corpus"
    assert params["use_tm_exact_match"] == "true"


def test_batch_translate_recursive_jobs_are_unique_for_duplicate_stems(tmp_path):
    from scripts.batch_translate_files import build_file_job

    src_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    first = src_dir / "chapter_a" / "dialog.xlsx"
    second = src_dir / "chapter_b" / "dialog.xlsx"

    first_task, first_output = build_file_job(
        first,
        src_dir=src_dir,
        out_dir=out_dir,
        task_prefix="run_",
        recursive=True,
    )
    second_task, second_output = build_file_job(
        second,
        src_dir=src_dir,
        out_dir=out_dir,
        task_prefix="run_",
        recursive=True,
    )

    assert first_task != second_task
    assert first_output != second_output
    assert first_task.startswith("run_chapter_a__dialog-")
    assert second_task.startswith("run_chapter_b__dialog-")
    assert first_output == out_dir / f"{first_task}.csv"
    assert second_output == out_dir / f"{second_task}.csv"


def test_batch_translate_non_recursive_job_keeps_legacy_task_id(tmp_path):
    from scripts.batch_translate_files import build_file_job

    src_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    task_id, out_path = build_file_job(
        src_dir / "dialog.xlsx",
        src_dir=src_dir,
        out_dir=out_dir,
        task_prefix="run_",
        recursive=False,
    )

    assert task_id == "run_dialog"
    assert out_path == out_dir / "run_dialog.csv"


def test_verify_stack_builds_health_url():
    from scripts.verify_stack import build_health_url

    assert build_health_url("http://localhost:8000/") == "http://localhost:8000/api/v1/health"


def test_dockerfile_copies_scripts_for_ops():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY scripts ./scripts" in dockerfile
