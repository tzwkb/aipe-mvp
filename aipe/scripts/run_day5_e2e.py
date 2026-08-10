"""Run the isolated Langlobal Day 5 integration matrix.

This script never uses an existing project profile or collection. The AIPE
process must be restarted for each run with these non-secret settings:

    PROJECTS_DIR=<repo>/aipe/data/day5-e2e/projects
    DEFAULT_PROJECT=langlobal-day5/zh-en
    QDRANT_COLLECTION=langlobal_day5_project_20260810_v1
    RAG_GLOBAL_COLLECTION=langlobal_day5_global_20260810_v1
    RAG_SPECIAL_COLLECTION=langlobal_day5_special_20260810_v1

Provider credentials stay in the service processes. This runner neither reads
nor accepts credential values. By default it removes its three exact synthetic
collections after the matrix and retains the synthetic profile as a fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

PROJECT_ID = "langlobal-day5/zh-en"
FIXTURE_MARKER = "Langlobal Day5 Synthetic E2E"
COLLECTIONS = {
    "global": "langlobal_day5_global_20260810_v1",
    "project": "langlobal_day5_project_20260810_v1",
    "special": "langlobal_day5_special_20260810_v1",
}
RAG_CASES = {
    "global": (
        "全局合成信标已点亮。",
        "The global synthetic beacon is lit.",
    ),
    "project": (
        "项目合成信标已点亮。",
        "The project synthetic beacon is lit.",
    ),
    "special": (
        "专项合成信标已点亮。",
        "The special synthetic beacon is lit.",
    ),
}
TERMINOLOGY = [
    {
        "source": "合成信标",
        "target": "Synthetic Beacon",
        "category": "test-only",
        "notes": "Day5 isolated E2E fixture",
    },
    {
        "source": "隔离测试",
        "target": "Isolated Test",
        "category": "test-only",
        "notes": "Day5 isolated E2E fixture",
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aipe-base-url",
        default="http://127.0.0.1:18000/api/v1",
    )
    parser.add_argument(
        "--langlobal-base-url",
        default=None,
        help="Optional Langlobal API base URL, for example http://127.0.0.1:18001/api",
    )
    parser.add_argument(
        "--qdrant-base-url",
        default="http://127.0.0.1:6333",
    )
    parser.add_argument(
        "--projects-dir",
        type=Path,
        default=repo_root / "data" / "day5-e2e" / "projects",
    )
    parser.add_argument(
        "--keep-collections",
        action="store_true",
        help="Retain only the three exact synthetic collections after the run",
    )
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_synthetic_profile(projects_dir: Path) -> Path:
    resolved = projects_dir.resolve()
    if "day5-e2e" not in resolved.parts:
        raise RuntimeError("projects_dir 必须位于独立的 day5-e2e 目录")
    profile_dir = resolved / "langlobal-day5" / "zh-en"
    profile_path = profile_dir / "profile.json"
    if profile_path.exists():
        current = json.loads(profile_path.read_text(encoding="utf-8"))
        if current.get("game") != FIXTURE_MARKER:
            raise RuntimeError(f"拒绝覆盖非 Day5 合成 profile: {profile_path}")
    profile = {
        "name": PROJECT_ID,
        "language_pair": "ZH-EN",
        "source_lang": "zh",
        "target_lang": "en",
        "game": FIXTURE_MARKER,
        "background": "Isolated synthetic integration test data only.",
        "terminology": "terminology.json",
        "qdrant_collection": COLLECTIONS["project"],
        "web_search_prefix": "Langlobal synthetic localization test",
        "allow_web_search": True,
    }
    _atomic_json(profile_path, profile)
    _atomic_json(profile_dir / "terminology.json", TERMINOLOGY[:1])
    return profile_path


def _error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return detail["code"]
    if isinstance(payload.get("error"), str):
        return payload["error"]
    return None


def _record(
    name: str,
    response: httpx.Response,
    *,
    expected_status: int = 200,
    semantic_ok: bool = True,
    **details: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": response.status_code == expected_status and semantic_ok,
        "http_status": response.status_code,
        "error_code": _error_code(response),
        **details,
    }


def _csv(rows: list[tuple[str, ...]]) -> bytes:
    return ("\n".join(",".join(row) for row in rows) + "\n").encode()


async def _upload_rag(
    client: httpx.AsyncClient,
    scope: str,
) -> dict[str, Any]:
    source, target = RAG_CASES[scope]
    content = _csv(
        [
            ("source", "target", "status"),
            (source, target, "Designer Reviewed"),
        ]
    )
    response = await client.post(
        "/rag/corpus/upload",
        files={"file": (f"rag_{scope}.csv", content, "text/csv")},
        data={"collection": COLLECTIONS[scope]},
    )
    payload = response.json() if response.status_code == 200 else {}
    indexed = payload.get("indexed") if isinstance(payload, dict) else None
    return _record(
        f"aipe.rag_upload.{scope}",
        response,
        semantic_ok=isinstance(indexed, int)
        and not isinstance(indexed, bool)
        and indexed >= 1,
        indexed=indexed,
    )


async def _search_aipe_rag(
    client: httpx.AsyncClient,
    scope: str,
) -> dict[str, Any]:
    source, target = RAG_CASES[scope]
    request: dict[str, Any] = {
        "query": source,
        "scope": scope,
        "threshold": 0.0,
        "top_k": 3,
    }
    if scope == "project":
        request["project_id"] = PROJECT_ID
    response = await client.post("/rag/search", json=request)
    payload = response.json() if response.status_code == 200 else {}
    results = payload.get("results", []) if isinstance(payload, dict) else []
    exact = bool(
        results
        and results[0].get("source") == source
        and results[0].get("target") == target
    )
    return _record(
        f"aipe.rag_search.{scope}",
        response,
        semantic_ok=(
            exact
            and payload.get("scope") == scope
            and payload.get("project_id")
            == (PROJECT_ID if scope == "project" else None)
        ),
        total=payload.get("total") if isinstance(payload, dict) else None,
        exact_synthetic_match=exact,
    )


async def run_aipe(aipe_base_url: str) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(
        base_url=aipe_base_url.rstrip("/"),
        timeout=timeout,
        trust_env=False,
    ) as client:
        results: list[dict[str, Any]] = []
        health = await client.get("/health")
        health_payload = health.json() if health.status_code == 200 else {}
        results.append(
            _record(
                "aipe.health",
                health,
                semantic_ok=isinstance(health_payload, dict)
                and health_payload.get("status") == "ok",
            )
        )

        before = await client.get(
            "/terminology",
            params={"project_id": PROJECT_ID, "limit": 10},
        )
        before_payload = before.json() if before.status_code == 200 else {}
        before_total = (
            before_payload.get("total") if isinstance(before_payload, dict) else None
        )
        results.append(
            _record(
                "aipe.terminology_read",
                before,
                semantic_ok=isinstance(before_total, int)
                and not isinstance(before_total, bool)
                and before_total >= 1,
                total=before_total,
            )
        )

        terminology_csv = _csv(
            [
                ("source", "target", "category", "notes"),
                *[
                    (
                        item["source"],
                        item["target"],
                        item["category"],
                        item["notes"],
                    )
                    for item in TERMINOLOGY
                ],
            ]
        )
        written = await client.post(
            "/terminology/upload",
            params={"project_id": PROJECT_ID},
            files={"file": ("terminology_upload.csv", terminology_csv, "text/csv")},
        )
        written_payload = written.json() if written.status_code == 200 else {}
        written_total = (
            written_payload.get("total") if isinstance(written_payload, dict) else None
        )
        results.append(
            _record(
                "aipe.terminology_write",
                written,
                semantic_ok=written_total == len(TERMINOLOGY),
                total=written_total,
            )
        )

        semaphore = asyncio.Semaphore(3)

        async def bounded_upload(scope: str) -> dict[str, Any]:
            async with semaphore:
                return await _upload_rag(client, scope)

        results.extend(
            await asyncio.gather(*(bounded_upload(scope) for scope in COLLECTIONS))
        )
        results.extend(
            await asyncio.gather(
                *(_search_aipe_rag(client, scope) for scope in COLLECTIONS)
            )
        )

        translated = await client.post(
            "/translate",
            json={
                "texts": ["合成信标已点亮。"],
                "source_lang": "zh",
                "target_lang": "en",
                "batch_size": 1,
                "enable_rag": True,
                "rag_threshold": 0.0,
                "rag_top_k": 3,
                "project_id": PROJECT_ID,
                "task_id": f"langlobal-day5-{uuid.uuid4().hex}",
                "content_types": ["ui"],
                "enable_web_search": False,
                "enable_vision": False,
                "use_tm_exact_match": False,
            },
        )
        translated_payload = translated.json() if translated.status_code == 200 else {}
        translation_results = (
            translated_payload.get("results", [])
            if isinstance(translated_payload, dict)
            else []
        )
        completed = (
            translated_payload.get("completed")
            if isinstance(translated_payload, dict)
            else None
        )
        terminology_used = bool(
            translation_results and translation_results[0].get("terminology_used")
        )
        rag_used = bool(
            translation_results and translation_results[0].get("rag_references")
        )
        results.append(
            _record(
                "aipe.translate",
                translated,
                semantic_ok=completed == 1 and terminology_used and rag_used,
                completed=completed,
                terminology_used=terminology_used,
                rag_used=rag_used,
            )
        )

        web = await client.post(
            "/web-search",
            json={
                "query": "localization engineering terminology management",
                "top_k": 1,
                "project_id": PROJECT_ID,
                "provider": "auto",
            },
        )
        web_payload = web.json() if web.status_code == 200 else {}
        results.append(
            _record(
                "aipe.web_search",
                web,
                total=web_payload.get("total")
                if isinstance(web_payload, dict)
                else None,
            )
        )
        return results


async def _langlobal_rag(
    client: httpx.AsyncClient,
    scope: str,
) -> dict[str, Any]:
    source, _ = RAG_CASES[scope]
    request: dict[str, Any] = {"source_text": source, "scope": scope, "top_k": 3}
    if scope == "project":
        request["project_id"] = PROJECT_ID
    response = await client.post("/rag/search", json=request)
    payload = response.json() if response.status_code == 200 else {}
    results = payload.get("results", []) if isinstance(payload, dict) else []
    exact = bool(
        results
        and results[0].get("source") == source
        and results[0].get("target") == RAG_CASES[scope][1]
    )
    return _record(
        f"langlobal.rag_search.{scope}",
        response,
        semantic_ok=(
            payload.get("service") == "aipe" and payload.get("scope") == scope and exact
        ),
        service=payload.get("service") if isinstance(payload, dict) else None,
        total=payload.get("total") if isinstance(payload, dict) else None,
        exact_synthetic_match=exact,
    )


async def _chat_sse(
    client: httpx.AsyncClient,
    request: dict[str, Any],
    *,
    name: str,
    expected_mode: str,
    expected_replayed: bool,
) -> dict[str, Any]:
    event_types: list[str] = []
    chunk_count = 0
    chunk_chars = 0
    replayed = False
    meta_mode = None
    async with client.stream("POST", "/chat/send", json=request) as response:
        if response.status_code != 200:
            await response.aread()
            return _record(name, response)
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            event_type = event.get("type")
            event_types.append(event_type)
            replayed = replayed or bool(event.get("replayed"))
            if event_type == "meta":
                meta_mode = event.get("mode")
            if event_type == "chunk":
                chunk_count += 1
                chunk_chars += len(event.get("content", ""))
    valid = (
        event_types[:1] == ["meta"]
        and event_types[-1:] == ["done"]
        and chunk_count > 0
        and meta_mode == expected_mode
        and replayed is expected_replayed
    )
    return {
        "name": name,
        "ok": valid,
        "http_status": 200,
        "error_code": None,
        "event_types": event_types,
        "chunk_count": chunk_count,
        "chunk_chars": chunk_chars,
        "replayed": replayed,
        "mode": meta_mode,
    }


async def run_langlobal(langlobal_base_url: str) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(
        base_url=langlobal_base_url.rstrip("/"),
        timeout=timeout,
        trust_env=False,
    ) as client:
        results: list[dict[str, Any]] = []
        health = await client.get("/health")
        health_payload = health.json() if health.status_code == 200 else {}
        results.append(
            _record(
                "langlobal.health",
                health,
                semantic_ok=isinstance(health_payload, dict)
                and health_payload.get("status") == "ok",
            )
        )
        integration = await client.get("/integrations/aipe")
        integration_payload = (
            integration.json() if integration.status_code == 200 else {}
        )
        results.append(
            _record(
                "langlobal.aipe_status",
                integration,
                semantic_ok=integration_payload.get("connected") is True,
                connected=(
                    integration_payload.get("connected")
                    if isinstance(integration_payload, dict)
                    else None
                ),
            )
        )

        document_csv = _csv(
            [
                ("source", "target", "status", "category"),
                (RAG_CASES["project"][0], "", "untranslated", "ui"),
            ]
        )
        uploaded = await client.post(
            "/files/upload",
            files={"file": ("langlobal_day5.csv", document_csv, "text/csv")},
            data={
                "source_lang": "zh",
                "target_lang": "en",
                "aipe_project_id": PROJECT_ID,
            },
        )
        uploaded_payload = uploaded.json() if uploaded.status_code == 200 else {}
        document_id = (
            uploaded_payload.get("id") if isinstance(uploaded_payload, dict) else None
        )
        segments = (
            uploaded_payload.get("segments", [])
            if isinstance(uploaded_payload, dict)
            else []
        )
        segment_id = segments[0].get("id") if segments else None
        results.append(
            _record(
                "langlobal.file_upload",
                uploaded,
                semantic_ok=(
                    uploaded_payload.get("aipe_project_id") == PROJECT_ID
                    and uploaded_payload.get("total") == 1
                    and bool(document_id)
                    and bool(segment_id)
                ),
                bound_to_synthetic_project=(
                    uploaded_payload.get("aipe_project_id") == PROJECT_ID
                    if isinstance(uploaded_payload, dict)
                    else False
                ),
            )
        )
        if not document_id or not segment_id:
            return results

        results.extend(
            await asyncio.gather(
                *(_langlobal_rag(client, scope) for scope in COLLECTIONS)
            )
        )
        translated = await client.post(
            "/translate/run",
            json={
                "document_id": document_id,
                "segment_ids": [segment_id],
                "enable_rag": True,
                "enable_web_search": False,
                "use_tm_exact_match": False,
            },
        )
        translated_payload = translated.json() if translated.status_code == 200 else {}
        translated_completed = (
            translated_payload.get("completed")
            if isinstance(translated_payload, dict)
            else None
        )
        results.append(
            _record(
                "langlobal.translate",
                translated,
                semantic_ok=translated_completed == 1,
                completed=translated_completed,
            )
        )

        terminology_csv = _csv(
            [
                ("source", "target", "category", "notes"),
                *[
                    (
                        item["source"],
                        item["target"],
                        item["category"],
                        item["notes"],
                    )
                    for item in TERMINOLOGY
                ],
            ]
        )
        glossary = await client.post(
            "/glossary/upload",
            files={"file": ("terminology_upload.csv", terminology_csv, "text/csv")},
            data={"document_id": document_id, "sync_aipe": "true"},
        )
        glossary_payload = glossary.json() if glossary.status_code == 200 else {}
        results.append(
            _record(
                "langlobal.glossary_write",
                glossary,
                semantic_ok=glossary_payload.get("aipe_synced") is True,
                aipe_synced=(
                    glossary_payload.get("aipe_synced")
                    if isinstance(glossary_payload, dict)
                    else None
                ),
            )
        )

        web = await client.post(
            "/chat/search",
            json={
                "query": "localization engineering terminology management",
                "document_id": document_id,
                "provider": "auto",
                "top_k": 1,
            },
        )
        results.append(_record("langlobal.web_search", web))

        request_id = f"day5-chat-{uuid.uuid4().hex}"
        chat_request = {
            "document_id": document_id,
            "request_id": request_id,
            "message": "请用一句英文确认这是隔离的合成测试。",
            "context": {"segment_id": segment_id},
            "use_rag": False,
            "use_web_search": False,
            "prompt_preset": "general",
        }
        results.append(
            await _chat_sse(
                client,
                chat_request,
                name="langlobal.chat_sse",
                expected_mode="provider",
                expected_replayed=False,
            )
        )
        replay = await _chat_sse(
            client,
            chat_request,
            name="langlobal.chat_idempotent_replay",
            expected_mode="replay",
            expected_replayed=True,
        )
        replay["ok"] = bool(
            replay["ok"]
            and replay["replayed"]
            and replay["event_types"] == ["meta", "chunk", "done"]
        )
        results.append(replay)
        return results


async def cleanup_collections(qdrant_base_url: str) -> list[dict[str, Any]]:
    parsed = urlsplit(qdrant_base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("安全限制：只允许清理本机 Qdrant 的固定 Day5 collections")
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=qdrant_base_url.rstrip("/"),
        timeout=30.0,
        trust_env=False,
    ) as client:
        for scope, collection in COLLECTIONS.items():
            if not collection.startswith("langlobal_day5_"):
                raise RuntimeError(f"拒绝清理非 Day5 collection: {collection}")
            deleted = await client.delete(f"/collections/{collection}")
            verified = await client.get(f"/collections/{collection}")
            results.append(
                {
                    "name": f"cleanup.qdrant.{scope}",
                    "ok": deleted.status_code in {200, 404}
                    and verified.status_code == 404,
                    "delete_http_status": deleted.status_code,
                    "verify_http_status": verified.status_code,
                    "collection": collection,
                }
            )
    return results


async def async_main(args: argparse.Namespace) -> int:
    profile_path = ensure_synthetic_profile(args.projects_dir)
    matrix: list[dict[str, Any]] = [
        {
            "name": "fixture.synthetic_profile",
            "ok": True,
            "path": str(profile_path),
            "project_id": PROJECT_ID,
        }
    ]
    try:
        matrix.extend(await run_aipe(args.aipe_base_url))
        if args.langlobal_base_url:
            matrix.extend(await run_langlobal(args.langlobal_base_url))
    finally:
        if not args.keep_collections:
            matrix.extend(await cleanup_collections(args.qdrant_base_url))

    print(json.dumps({"matrix": matrix}, ensure_ascii=False, indent=2))
    return 0 if all(item.get("ok") for item in matrix) else 1


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"fatal": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
