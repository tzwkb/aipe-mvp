#!/usr/bin/env python3
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import httpx

DEFAULT_BASE = "http://localhost:8000/api/v1"
SOURCE_KEYS = ("source", "source_text", "原文", "源文", "中文", "待译文本", "text")


def load_texts(path):
    if path.endswith(".json"):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("items") or data.get("rows") or data.get("segments") or []
        texts, content_types = [], []
        for row in data:
            if isinstance(row, str):
                texts.append(row)
                content_types.append(None)
            else:
                texts.append(row.get("source") or row.get("text"))
                content_types.append(row.get("content_type"))
        return texts, content_types
    if path.endswith((".csv", ".tsv")):
        raw = Path(path).read_bytes()
        for enc in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        rows = list(csv.DictReader(io.StringIO(text)))
        key = next((k for k in rows[0] if k and k.strip().lower() in SOURCE_KEYS), None) if rows else None
        if key is None:
            raise SystemExit(f"CSV 未找到原文列: {list(rows[0]) if rows else 'empty'}")
        return [r[key] for r in rows if r.get(key)], [None] * len(rows)
    raise SystemExit("texts 模式仅支持 .json/.csv/.tsv")


def run_texts(base, texts, content_types, args):
    payload = {
        "texts": texts,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "enable_rag": args.enable_rag,
        "rag_threshold": args.rag_threshold,
        "rag_top_k": args.rag_top_k,
        "content_types": content_types if any(content_types) else None,
        "project_id": args.project_id,
    }
    r = httpx.post(f"{base}/translate", json=payload, timeout=args.timeout)
    if r.status_code >= 400:
        raise SystemExit(f"translate {r.status_code}: {r.text[:2000]}")
    return r.json()


def run_file(base, path, args):
    data = {
        "enable_rag": str(args.enable_rag).lower(),
        "rag_threshold": str(args.rag_threshold),
        "rag_top_k": str(args.rag_top_k),
        "enable_cluster": str(args.enable_cluster).lower(),
    }
    if args.project_id:
        data["project_id"] = args.project_id
    if args.task_id:
        data["task_id"] = args.task_id
    with open(path, "rb") as fh:
        files = {"file": (Path(path).name, fh)}
        r = httpx.post(f"{base}/translate/file", files=files, data=data, timeout=args.timeout)
    if r.status_code >= 400:
        raise SystemExit(f"translate/file {r.status_code}: {r.text[:2000]}")
    return r.json()


def normalize(raw, side, base, args, mode):
    items = []
    for res in raw.get("results", []):
        items.append(
            {
                "source": res.get("source"),
                "target": res.get("translation"),
                "status": res.get("status"),
                "content_type": res.get("content_type"),
                "terminology_used": res.get("terminology_used") or [],
                "rag_references": res.get("rag_references") or [],
                "reason": res.get("translation_reason"),
                "tm_exact_match_used": res.get("tm_exact_match_used"),
                "tm_exact_match_status": res.get("tm_exact_match_status"),
                "web_search_triggered": res.get("web_search_triggered"),
                "error": res.get("error_msg"),
                "warnings": res.get("warnings") or [],
            }
        )
    return {
        "side": side,
        "base_url": base,
        "mode": mode,
        "task_id": raw.get("task_id"),
        "project_id": args.project_id,
        "flags": {
            "enable_rag": args.enable_rag,
            "rag_threshold": args.rag_threshold,
            "rag_top_k": args.rag_top_k,
            "enable_cluster": args.enable_cluster,
        },
        "items": items,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--label", default="local_aipe")
    p.add_argument("--project-id", default="wwm/zh-en")
    p.add_argument("--source-lang", default="zh")
    p.add_argument("--target-lang", default="en")
    p.add_argument("--out", default="./out_local")
    p.add_argument("--mode", choices=["auto", "texts", "file"], default="auto")
    p.add_argument("--task-id", default="")
    p.add_argument("--no-rag", dest="enable_rag", action="store_false")
    p.add_argument("--no-cluster", dest="enable_cluster", action="store_false")
    p.add_argument("--rag-threshold", type=float, default=0.85)
    p.add_argument("--rag-top-k", type=int, default=3)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验输入和请求计划，不调用 AIPE API",
    )
    args = p.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "texts" if args.input.endswith((".json", ".csv", ".tsv")) else "file"

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"输入文件不存在: {input_path}")
    if args.dry_run:
        item_count = None
        if mode == "texts":
            texts, _ = load_texts(args.input)
            item_count = len(texts)
            if not item_count:
                raise SystemExit("texts 模式输入为空")
        print(json.dumps({
            "dry_run": True,
            "input": str(input_path.resolve()),
            "base_url": args.base_url,
            "project_id": args.project_id or None,
            "mode": mode,
            "item_count": item_count,
            "flags": {
                "enable_rag": args.enable_rag,
                "rag_threshold": args.rag_threshold,
                "rag_top_k": args.rag_top_k,
                "enable_cluster": args.enable_cluster,
            },
            "api_called": False,
        }, ensure_ascii=False, indent=2))
        return 0

    if mode == "texts":
        texts, content_types = load_texts(args.input)
        raw = run_texts(args.base_url, texts, content_types, args)
    else:
        raw = run_file(args.base_url, args.input, args)

    norm = normalize(raw, args.label, args.base_url, args, mode)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "local_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    target = outdir / "local_normalized.json"
    target.write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for i in norm["items"] if i["status"] in ("success", "translated"))
    print(f"{mode}: {ok}/{len(norm['items'])} ok -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
