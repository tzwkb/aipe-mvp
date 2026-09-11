"""Create a reusable AIPE project profile.

Usage:
    python3 -m scripts.create_project_profile wwm/zh-en \
      --game "Where Winds Meet" \
      --language-pair ZH-EN \
      --style-guide data/style_guide/style_guide_0701_combined.md \
      --terminology data/terminology/术语0701更新.xlsx \
      --qdrant-collection yanyun_corpus \
      --web-search-prefix 燕云十六声 \
      --prompt-notes "Avoid casual internet-style English unless required."
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def _safe_project_id(project_id: str) -> Path:
    path = Path(project_id)
    if path.is_absolute() or ".." in path.parts or not project_id.strip():
        raise ValueError(f"非法 project_id: {project_id!r}")
    return path


def _asset_relative_to_profile(profile_dir: Path, asset: Path | None) -> str | None:
    if asset is None:
        return None
    if not asset.exists():
        raise FileNotFoundError(f"资源文件不存在: {asset}")
    rel = os.path.relpath(asset.resolve(), start=profile_dir.resolve())
    return Path(rel).as_posix()


def _split_language_pair_codes(language_pair: str) -> tuple[str | None, str | None]:
    parts = [p for p in re.split(r"[-_>/→]+", language_pair.strip()) if p]
    if len(parts) < 2:
        return None, None
    return parts[0].lower(), parts[-1].lower()


def _validate_language_pair_suffix(project_id: str, source_lang: str, target_lang: str) -> None:
    expected = f"{source_lang}-{target_lang}"
    actual = Path(project_id).name.lower()
    if actual != expected:
        raise ValueError(
            f"project_id 最后一级必须写明语言对 {expected!r}，当前为 {project_id!r}"
        )


def create_project_profile(
    *,
    projects_dir: Path,
    project_id: str,
    language_pair: str,
    source_lang: str | None = None,
    target_lang: str | None = None,
    game: str,
    background: str,
    style_guide: Path | None,
    terminology: Path | None,
    qdrant_collection: str | None,
    web_search_prefix: str | None,
    prompt_notes: str | None,
    vision_system_prompt: str | None,
    allow_web_search: bool = True,
    force: bool = False,
    profile_contract_version: int = 1,
    profile_status: str = "active",
    distribution: str = "internal_only",
    authority_policy: list[str] | None = None,
    terminology_status: dict[str, Any] | None = None,
    assets: dict[str, Any] | None = None,
    context_pipeline_mode: str = "off",
    capabilities: dict[str, Any] | None = None,
    module_context_views: dict[str, Any] | None = None,
    content_scope: dict[str, Any] | None = None,
    workflow: dict[str, Any] | None = None,
    gaps: list[dict[str, Any]] | None = None,
) -> Path:
    project_dir = projects_dir / _safe_project_id(project_id)
    profile_path = project_dir / "profile.json"
    if profile_path.exists() and not force:
        raise FileExistsError(f"profile 已存在，使用 --force 覆盖: {profile_path}")

    project_dir.mkdir(parents=True, exist_ok=True)
    pair_source, pair_target = _split_language_pair_codes(language_pair)
    resolved_source_lang = (source_lang or pair_source or "").strip().lower()
    resolved_target_lang = (target_lang or pair_target or "").strip().lower()
    if not resolved_source_lang or not resolved_target_lang:
        raise ValueError(f"无法从 language_pair/source_lang/target_lang 解析语言对: {language_pair!r}")
    _validate_language_pair_suffix(project_id, resolved_source_lang, resolved_target_lang)
    if profile_contract_version not in {1, 2}:
        raise ValueError("profile_contract_version 仅支持 1 或 2")
    if context_pipeline_mode not in {"off", "shadow", "enforce"}:
        raise ValueError("context_pipeline_mode 仅支持 off/shadow/enforce")
    if not profile_status.strip() or not distribution.strip():
        raise ValueError("profile_status / distribution 不能为空")

    payload: dict[str, Any] = {
        "name": project_id,
        "language_pair": language_pair,
        "source_lang": resolved_source_lang,
        "target_lang": resolved_target_lang,
        "game": game,
        "background": background,
        "allow_web_search": allow_web_search,
    }
    if profile_contract_version >= 2:
        payload = {
            "profile_contract_version": profile_contract_version,
            **payload,
            "profile_status": profile_status.strip(),
            "distribution": distribution.strip(),
            "authority_policy": authority_policy or [],
            "terminology_status": terminology_status or {},
            "assets": assets or {},
            "context_pipeline": {"mode": context_pipeline_mode},
            "capabilities": capabilities or {},
        }
        if module_context_views:
            payload["module_context_views"] = module_context_views
    if content_scope:
        payload["content_scope"] = content_scope
    if workflow:
        payload["workflow"] = workflow
    if gaps:
        payload["gaps"] = gaps

    style_rel = _asset_relative_to_profile(project_dir, style_guide)
    term_rel = _asset_relative_to_profile(project_dir, terminology)
    if style_rel:
        payload["style_guide"] = style_rel
    if term_rel:
        payload["terminology"] = term_rel
    if qdrant_collection:
        payload["qdrant_collection"] = qdrant_collection
    if web_search_prefix:
        payload["web_search_prefix"] = web_search_prefix
    if vision_system_prompt:
        payload["vision_system_prompt"] = vision_system_prompt
    if prompt_notes:
        notes_path = project_dir / "prompt_notes.md"
        notes_path.write_text(prompt_notes.strip(), encoding="utf-8")
        payload["prompt_notes"] = notes_path.name

    tmp = profile_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(profile_path)
    return profile_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/projects/<game>/<source-target>/profile.json.")
    parser.add_argument("project_id", help="Project id, e.g. wwm/zh-en or nrc/zh-th")
    parser.add_argument("--projects-dir", type=Path, default=Path("data/projects"))
    parser.add_argument("--language-pair", default="ZH-EN")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--game", required=True)
    parser.add_argument("--background", default="")
    parser.add_argument("--style-guide", type=Path, default=None)
    parser.add_argument("--terminology", type=Path, default=None)
    parser.add_argument("--qdrant-collection", default=None)
    parser.add_argument("--web-search-prefix", default=None)
    parser.add_argument("--prompt-notes", default=None)
    parser.add_argument("--vision-system-prompt", default=None)
    parser.add_argument("--profile-contract-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--profile-status", default="active")
    parser.add_argument("--distribution", default="internal_only")
    parser.add_argument("--authority-policy-json", type=Path, default=None)
    parser.add_argument("--terminology-status-json", type=Path, default=None)
    parser.add_argument("--assets-json", type=Path, default=None)
    parser.add_argument(
        "--context-pipeline-mode",
        choices=("off", "shadow", "enforce"),
        default="off",
    )
    parser.add_argument("--capabilities-json", type=Path, default=None)
    parser.add_argument("--module-context-views-json", type=Path, default=None)
    parser.add_argument("--content-scope-json", type=Path, default=None)
    parser.add_argument("--workflow-json", type=Path, default=None)
    parser.add_argument("--gaps-json", type=Path, default=None)
    parser.add_argument(
        "--disable-web-search",
        action="store_true",
        help="Persist allow_web_search=false for confidential projects",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing profile.json")
    return parser.parse_args()


def _read_json(path: Path | None, expected_type: type, label: str):
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 读取失败: {path}: {exc}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{label} 顶层类型必须是 {expected_type.__name__}: {path}")
    return value


def main() -> int:
    args = parse_args()
    path = create_project_profile(
        projects_dir=args.projects_dir,
        project_id=args.project_id,
        language_pair=args.language_pair,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        game=args.game,
        background=args.background,
        style_guide=args.style_guide,
        terminology=args.terminology,
        qdrant_collection=args.qdrant_collection,
        web_search_prefix=args.web_search_prefix,
        prompt_notes=args.prompt_notes,
        vision_system_prompt=args.vision_system_prompt,
        allow_web_search=not args.disable_web_search,
        force=args.force,
        profile_contract_version=args.profile_contract_version,
        profile_status=args.profile_status,
        distribution=args.distribution,
        authority_policy=_read_json(args.authority_policy_json, list, "authority policy JSON"),
        terminology_status=_read_json(
            args.terminology_status_json,
            dict,
            "terminology status JSON",
        ),
        assets=_read_json(args.assets_json, dict, "assets JSON"),
        context_pipeline_mode=args.context_pipeline_mode,
        capabilities=_read_json(args.capabilities_json, dict, "capabilities JSON"),
        module_context_views=_read_json(
            args.module_context_views_json,
            dict,
            "module context views JSON",
        ),
        content_scope=_read_json(args.content_scope_json, dict, "content scope JSON"),
        workflow=_read_json(args.workflow_json, dict, "workflow JSON"),
        gaps=_read_json(args.gaps_json, list, "gaps JSON"),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
