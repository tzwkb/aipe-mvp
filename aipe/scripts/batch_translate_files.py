#!/usr/bin/env python3
"""批量翻译目录下的 Excel 文件，并把结果导出为 CSV。

逐个文件调用 POST /api/v1/translate/file 翻译（task_id = 文件名去掉扩展名），
再用接口返回的完整结果按 /translate/task/{task_id}/csv 的列格式写出
CSV 到输出目录。

用法:
    python scripts/batch_translate_files.py
    python scripts/batch_translate_files.py --base-url http://localhost:8000 --overwrite
    python scripts/batch_translate_files.py --src-dir ./input --out-dir ./output --task-prefix run_20260706

服务端需先启动:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# 项目根目录（本脚本在 scripts/ 下）
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "to_be_translated"
OUT_DIR = ROOT / "having_translated"

# 与 /api/v1/translate/task/{task_id}/csv 完全一致的列
CSV_HEADER = [
    "source",
    "translation",
    "translation_reason",
    "status",
    "content_type",
    "terminology_used",
    "rag_references",
    "tm_exact_match_used",
    "tm_exact_match_source",
    "tm_exact_match_target",
    "tm_exact_match_status",
    "tm_exact_match_score",
    "web_references",
    "web_search_triggered",
    "image_analysis",
    "error_msg",
]


def _bool_form(value: bool) -> str:
    return "true" if value else "false"


def build_translate_params(args: argparse.Namespace, task_id: str) -> dict:
    """构造 /translate/file form 参数。

    默认不传 ``rag_collection``，让后端按 ``project_id`` 的 profile 选择 collection；
    只有人工显式传入 ``--rag-collection`` 时才覆盖。
    """
    data = {
        "task_id": task_id,
        "batch_size": args.batch_size,
        "project_id": args.project_id,
        "enable_rag": _bool_form(args.enable_rag),
        "rag_threshold": args.rag_threshold,
        "rag_top_k": args.rag_top_k,
        "enable_cluster": _bool_form(args.enable_cluster),
        "dialog_mode": _bool_form(args.dialog_mode),
        "enable_web_search": _bool_form(args.enable_web_search),
        "web_search_dense_threshold": args.web_search_dense_threshold,
        "enable_vision": _bool_form(args.enable_vision),
        "use_tm_exact_match": _bool_form(args.use_tm_exact_match),
    }
    if args.rag_collection:
        data["rag_collection"] = args.rag_collection
    return data


def discover_xlsx_files(src_dir: Path, *, recursive: bool = False) -> list[Path]:
    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    return sorted(
        p for p in src_dir.glob(pattern) if p.is_file() and not p.name.startswith("~$")
    )


def write_csv(resp: dict, out_path: Path) -> None:
    """按导出接口的格式把翻译结果写为 CSV（utf-8-sig，Excel 可直接识别）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in resp.get("results", []):
            term = r.get("terminology_used") or []
            rag = r.get("rag_references")
            web = r.get("web_references")
            triggered = r.get("web_search_triggered")
            writer.writerow(
                [
                    r.get("source", ""),
                    r.get("translation", ""),
                    r.get("translation_reason") or "",
                    r.get("status", ""),
                    r.get("content_type") or "",
                    json.dumps(term, ensure_ascii=False) if term else "",
                    json.dumps(rag, ensure_ascii=False) if rag else "",
                    str(bool(r.get("tm_exact_match_used", False))).lower(),
                    r.get("tm_exact_match_source") or "",
                    r.get("tm_exact_match_target") or "",
                    r.get("tm_exact_match_status") or "",
                    "" if r.get("tm_exact_match_score") is None else str(r.get("tm_exact_match_score")),
                    json.dumps(web, ensure_ascii=False) if web else "",
                    "" if triggered is None else str(triggered).lower(),
                    r.get("image_analysis") or "",
                    r.get("error_msg") or "",
                ]
            )


def translate_file(base_url: str, path: Path, task_id: str, timeout: float, params: dict) -> dict:
    """调用 /translate/file 翻译单个文件，返回 BatchTranslateResponse 字典。"""
    import requests

    url = f"{base_url.rstrip('/')}/api/v1/translate/file"
    data = dict(params)
    with path.open("rb") as fh:
        files = {
            "file": (
                path.name,
                fh,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        resp = requests.post(url, data=data, files=files, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--src-dir", type=Path, default=SRC_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--recursive", action="store_true", help="递归查找 src-dir 下的 .xlsx")
    parser.add_argument("--task-prefix", default="", help="给 task_id 增加前缀，避免与旧进度文件冲突")
    parser.add_argument("--project-id", default="wwm/zh-en")
    parser.add_argument(
        "--rag-collection",
        default=None,
        help="默认不传，让 project profile 决定；仅在需要人工覆盖时填写",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--rag-threshold", type=float, default=0.85)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--no-rag", dest="enable_rag", action="store_false")
    parser.set_defaults(enable_rag=True)
    parser.add_argument("--no-cluster", dest="enable_cluster", action="store_false")
    parser.set_defaults(enable_cluster=True)
    parser.add_argument("--dialog-mode", action="store_true")
    parser.add_argument("--no-web-search", dest="enable_web_search", action="store_false")
    parser.set_defaults(enable_web_search=True)
    parser.add_argument("--web-search-dense-threshold", type=float, default=0.85)
    parser.add_argument("--enable-vision", dest="enable_vision", action="store_true")
    parser.set_defaults(enable_vision=False)
    parser.add_argument(
        "--use-tm-exact-match",
        action="store_true",
        help="如 TM/RAG 中存在 source 完全一致条目，直接采用 target 并跳过 AI 翻译",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="单文件翻译超时（秒），默认 1 小时",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="即使 having_translated/ 已有同名 CSV 也重新翻译（默认跳过已完成文件）",
    )
    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.out_dir
    if not src_dir.is_dir():
        print(f"待翻译目录不存在: {src_dir}", file=sys.stderr)
        return 1

    xlsx_files = discover_xlsx_files(src_dir, recursive=args.recursive)
    if not xlsx_files:
        print(f"未在 {src_dir} 找到 .xlsx 文件")
        return 0

    print(f"共发现 {len(xlsx_files)} 个待翻译文件 -> 输出到 {out_dir}\n")

    ok, skipped, failed = 0, 0, 0
    for i, path in enumerate(xlsx_files, 1):
        task_id = f"{args.task_prefix}{path.stem}"  # 文件名（去扩展名）作为 task_id
        out_path = out_dir / f"{task_id}.csv"
        prefix = f"[{i}/{len(xlsx_files)}] {path.name}"

        if out_path.exists() and not args.overwrite:
            print(f"{prefix} -> 已存在，跳过（用 --overwrite 强制重译）")
            skipped += 1
            continue

        print(f"{prefix} -> 翻译中 (task_id={task_id}) ...", flush=True)
        try:
            params = build_translate_params(args, task_id)
            resp = translate_file(args.base_url, path, task_id, args.timeout, params)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断整批
            print(f"{prefix} -> 失败: {exc}", file=sys.stderr)
            failed += 1
            continue

        write_csv(resp, out_path)
        total = resp.get("total", 0)
        completed = resp.get("completed", 0)
        status = resp.get("status", "?")
        print(
            f"{prefix} -> 完成 {completed}/{total} ({status})，已写出 {out_path.name}"
        )
        ok += 1

    print(f"\n完成。成功 {ok}，跳过 {skipped}，失败 {failed}。")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
