"""Verify the running AIPE stack and active project profile.

Usage:
    python3 -m scripts.verify_stack
    python3 -m scripts.verify_stack --project-id wwm/zh-en --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from typing import Any


def build_health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/health"


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local/dev verification helper
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


async def qdrant_count(host: str, port: int, collection: str) -> int:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(host=host, port=port, timeout=30.0)
    try:
        result = await client.count(collection_name=collection, exact=True)
        return int(result.count)
    finally:
        await client.close()


async def collect_report(
    *,
    base_url: str,
    project_id: str | None,
    include_health: bool,
    include_qdrant: bool,
    timeout: float,
) -> dict[str, Any]:
    from app.config import get_settings
    from app.services.project_service import get_project_resource_manager

    settings = get_settings()
    manager = get_project_resource_manager()
    profile = manager.profile(project_id or settings.default_project)

    report: dict[str, Any] = {
        "model": settings.llm_model,
        "default_project": settings.default_project,
        "project": {
            "name": profile.name,
            "language_pair": profile.language_pair,
            "source_lang": profile.source_lang,
            "target_lang": profile.target_lang,
            "game": profile.game,
            "qdrant_collection": profile.qdrant_collection,
            "web_search_prefix": profile.web_search_prefix,
            "profile_dir": str(profile.profile_dir),
        },
        "terminology_total": len(manager.terminology(profile.name).entries),
        "style_guide": manager.style_guide(profile.name).info(),
    }

    if include_health:
        report["health"] = fetch_json(build_health_url(base_url), timeout=timeout)

    if include_qdrant and profile.qdrant_collection:
        report["qdrant_points"] = await qdrant_count(
            settings.qdrant_host,
            settings.qdrant_port,
            profile.qdrant_collection,
        )
    return report


def print_text_report(report: dict[str, Any]) -> None:
    project = report["project"]
    style = report["style_guide"]
    print(f"model: {report['model']}")
    print(f"default_project: {report['default_project']}")
    if "health" in report:
        print(f"health: {report['health']}")
    print(
        f"project: {project['name']} ({project['language_pair']}, "
        f"{project['source_lang']}->{project['target_lang']})"
    )
    print(f"game: {project['game']}")
    print(f"collection: {project['qdrant_collection']}")
    print(f"web_search_prefix: {project['web_search_prefix']}")
    print(f"terminology_total: {report['terminology_total']}")
    print(
        "style_guide: "
        f"loaded={style['loaded']} filename={style['filename']} "
        f"chars={style['char_count']} lines={style['line_count']}"
    )
    if "qdrant_points" in report:
        print(f"qdrant_points: {report['qdrant_points']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify AIPE API, project resources, and Qdrant collection.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(
        collect_report(
            base_url=args.base_url,
            project_id=args.project_id,
            include_health=not args.skip_health,
            include_qdrant=not args.skip_qdrant,
            timeout=args.timeout,
        )
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print_text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
