"""Render the OpenAI-compatible chat.completions payload without calling an API.

The snapshot includes profile, style-guide, terminology and structured project context.
RAG, web and TM sections are absent because this command intentionally performs no
external or vector-database calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.services.project_service import ProjectRegistry, ProjectResourceManager
from app.services.style_guide_service import ContentType
from app.services.translation_prompts import (
    base_system_for_project,
    build_dialog_messages,
    build_single_messages,
)


def _expand(values: list[str] | None, size: int, field_name: str) -> list[str | None]:
    if not values:
        return [None] * size
    if len(values) == 1:
        return values * size
    if len(values) != size:
        raise ValueError(f"{field_name} 必须提供 1 个值或与 source 等长的 {size} 个值")
    return list(values)


def build_payload(args: argparse.Namespace) -> dict:
    settings = Settings()
    registry = ProjectRegistry(Path(args.projects_dir), default_project=args.project_id)
    resources = ProjectResourceManager(registry)
    profile = resources.profile(args.project_id)
    context_service = resources.project_context(args.project_id)
    content_type = ContentType(args.content_type)
    sources = [source.strip() for source in args.source]
    if any(not source for source in sources):
        raise ValueError("source 不能为空")

    speakers = _expand(args.speaker, len(sources), "speaker")
    addressees = _expand(args.addressee, len(sources), "addressee")
    scene_ids = _expand(args.scene_id, len(sources), "scene_id")
    relationship_stages = _expand(
        args.relationship_stage,
        len(sources),
        "relationship_stage",
    )
    scene_tones = _expand(args.scene_tone, len(sources), "scene_tone")
    context_notes = _expand(args.context_note, len(sources), "context_note")

    term_matches = []
    seen_terms: set[str] = set()
    terminology = resources.terminology(args.project_id)
    for source in sources:
        for entry in terminology.find_matches(source):
            if entry.source not in seen_terms:
                seen_terms.add(entry.source)
                term_matches.append(entry)

    project_context = context_service.render(
        sources,
        content_type=content_type.value,
        speakers=speakers,
        addressees=addressees,
        dialog_id=args.dialog_id,
        scene_ids=scene_ids,
        relationship_stages=relationship_stages,
        scene_tones=scene_tones,
        context_notes=context_notes,
    )
    prompt_args = {
        "style_guide_svc": resources.style_guide(args.project_id),
        "base_system": base_system_for_project(
            profile,
            resources.prompt_notes(args.project_id),
        ),
        "target_lang": profile.target_lang,
        "project_context": project_context,
    }
    if len(sources) == 1:
        messages = build_single_messages(
            sources[0],
            term_matches,
            [],
            content_type,
            **prompt_args,
        )
    else:
        messages = build_dialog_messages(
            sources,
            speakers,
            term_matches,
            [],
            content_type,
            args.dialog_id,
            **prompt_args,
        )

    return {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
    }


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Render the exact local chat.completions request body without sending it."
    )
    parser.add_argument("--projects-dir", default=settings.projects_dir)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--speaker", action="append")
    parser.add_argument("--addressee", action="append")
    parser.add_argument("--dialog-id")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--relationship-stage", action="append")
    parser.add_argument("--scene-tone", action="append")
    parser.add_argument("--context-note", action="append")
    parser.add_argument(
        "--content-type",
        choices=[content_type.value for content_type in ContentType],
        default=ContentType.STORY.value,
    )
    return parser.parse_args()


def main() -> int:
    payload = build_payload(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
