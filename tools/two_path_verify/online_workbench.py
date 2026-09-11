#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_BASE = "https://www.langlobal-tech.com/localization-tool/api"


def upload(base, path, source_lang, target_lang, project_id, timeout):
    with open(path, "rb") as fh:
        files = {"file": (Path(path).name, fh)}
        data = {"source_lang": source_lang, "target_lang": target_lang}
        if project_id:
            data["aipe_project_id"] = project_id
        r = httpx.post(f"{base}/files/upload", files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run_translate(base, document_id, enable_rag, enable_web_search, use_tm_exact_match, timeout):
    payload = {
        "document_id": document_id,
        "enable_rag": enable_rag,
        "enable_web_search": enable_web_search,
        "use_tm_exact_match": use_tm_exact_match,
    }
    r = httpx.post(f"{base}/translate/run", json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise SystemExit(f"translate/run {r.status_code}: {r.text[:2000]}")
    return r.json()


def normalize(doc, base, project_id, flags):
    items = []
    for seg in doc.get("segments", []):
        md = seg.get("translation_metadata") or {}
        items.append(
            {
                "position": seg.get("position"),
                "segment_id": seg.get("id"),
                "source": seg.get("source"),
                "target": seg.get("target"),
                "status": seg.get("status"),
                "category": seg.get("category"),
                "category_confidence": seg.get("category_confidence"),
                "content_type": md.get("content_type"),
                "terminology_used": md.get("terminology_used") or [],
                "rag_references": md.get("rag_references") or [],
                "reason": md.get("translation_reason"),
                "task_id": md.get("task_id"),
                "tm_exact_match_used": md.get("tm_exact_match_used"),
                "tm_exact_match_status": md.get("tm_exact_match_status"),
                "web_search_triggered": md.get("web_search_triggered"),
                "error": md.get("error_msg"),
                "warnings": seg.get("warnings") or [],
                "tag_integrity": seg.get("tag_integrity"),
            }
        )
    return {
        "side": "online_workbench",
        "base_url": base,
        "document_id": doc.get("id"),
        "document_name": doc.get("name"),
        "aipe_project_id": doc.get("aipe_project_id"),
        "active_glossary_id": doc.get("active_glossary_id"),
        "tag_rules": doc.get("tag_rules"),
        "flags": flags,
        "requested_project_id": project_id,
        "items": items,
    }


def write_outputs(norm, raw, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "online_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "online_normalized.json").write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
    return outdir / "online_normalized.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--project-id", default="")
    p.add_argument("--source-lang", default="zh")
    p.add_argument("--target-lang", default="en")
    p.add_argument("--out", default="./out_online")
    p.add_argument("--no-rag", dest="enable_rag", action="store_false")
    p.add_argument("--web-search", dest="enable_web_search", action="store_true")
    p.add_argument("--no-tm", dest="use_tm_exact_match", action="store_false")
    p.add_argument("--upload-timeout", type=float, default=120.0)
    p.add_argument("--translate-timeout", type=float, default=900.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验输入和请求计划，不上传文件、不调用线上 API",
    )
    args = p.parse_args()

    flags = {
        "enable_rag": args.enable_rag,
        "enable_web_search": args.enable_web_search,
        "use_tm_exact_match": args.use_tm_exact_match,
    }

    input_path = Path(args.file)
    if not input_path.is_file():
        raise SystemExit(f"输入文件不存在: {input_path}")
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "input": str(input_path.resolve()),
            "base_url": args.base_url,
            "project_id": args.project_id or None,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "flags": flags,
            "would_write_online_database": True,
            "api_called": False,
        }, ensure_ascii=False, indent=2))
        return 0

    doc = upload(args.base_url, args.file, args.source_lang, args.target_lang, args.project_id, args.upload_timeout)
    print(f"uploaded document_id={doc['id']} segments={len(doc.get('segments', []))}")
    if not doc.get("aipe_project_id"):
        raise SystemExit("文档未绑定 aipe_project_id，translate/run 会 400")

    result = run_translate(
        args.base_url,
        doc["id"],
        flags["enable_rag"],
        flags["enable_web_search"],
        flags["use_tm_exact_match"],
        args.translate_timeout,
    )
    document = result.get("document") or result
    norm = normalize(document, args.base_url, args.project_id, flags)
    norm["task_id"] = result.get("task_id")
    path = write_outputs(norm, result, args.out)
    ok = sum(1 for i in norm["items"] if i["status"] == "translated")
    print(f"translated {ok}/{len(norm['items'])} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
