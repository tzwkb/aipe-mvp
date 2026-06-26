#!/usr/bin/env python3
"""批量翻译 to_be_translated/ 下的 Excel 文件，并把结果导出为 CSV。

逐个文件调用 POST /api/v1/translate/file 翻译（task_id = 文件名去掉扩展名），
再用接口返回的完整结果按 /translate/task/{task_id}/csv 的列格式写出
CSV 到 having_translated/{task_id}.csv。

用法:
    python scripts/batch_translate_files.py
    python scripts/batch_translate_files.py --base-url http://localhost:8000 --overwrite

服务端需先启动:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

# 项目根目录（本脚本在 scripts/ 下）
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "to_be_translated"
OUT_DIR = ROOT / "having_translated"

# 翻译参数（按需求固定）
TRANSLATE_PARAMS = {
    "batch_size": 20,
    "enable_rag": "true",
    "rag_threshold": 0.85,
    "rag_top_k": 3,
    "enable_cluster": "true",
    "dialog_mode": "false",
    "rag_collection": "yanyun_0512",
    "enable_web_search": "true",
    "web_search_dense_threshold": 0.85,
    "enable_vision": "false",
}

# 与 /api/v1/translate/task/{task_id}/csv 完全一致的列
CSV_HEADER = [
    "source",
    "translation",
    "translation_reason",
    "status",
    "content_type",
    "terminology_used",
    "rag_references",
    "web_references",
    "web_search_triggered",
    "image_analysis",
    "error_msg",
]


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
                    json.dumps(web, ensure_ascii=False) if web else "",
                    "" if triggered is None else str(triggered).lower(),
                    r.get("image_analysis") or "",
                    r.get("error_msg") or "",
                ]
            )


def translate_file(base_url: str, path: Path, task_id: str, timeout: float) -> dict:
    """调用 /translate/file 翻译单个文件，返回 BatchTranslateResponse 字典。"""
    url = f"{base_url.rstrip('/')}/api/v1/translate/file"
    data = dict(TRANSLATE_PARAMS)
    data["task_id"] = task_id
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

    if not SRC_DIR.is_dir():
        print(f"待翻译目录不存在: {SRC_DIR}", file=sys.stderr)
        return 1

    xlsx_files = sorted(p for p in SRC_DIR.glob("*.xlsx") if not p.name.startswith("~$"))
    if not xlsx_files:
        print(f"未在 {SRC_DIR} 找到 .xlsx 文件")
        return 0

    print(f"共发现 {len(xlsx_files)} 个待翻译文件 -> 输出到 {OUT_DIR}\n")

    ok, skipped, failed = 0, 0, 0
    for i, path in enumerate(xlsx_files, 1):
        task_id = path.stem  # 文件名（去扩展名）作为 task_id
        out_path = OUT_DIR / f"{task_id}.csv"
        prefix = f"[{i}/{len(xlsx_files)}] {path.name}"

        if out_path.exists() and not args.overwrite:
            print(f"{prefix} -> 已存在，跳过（用 --overwrite 强制重译）")
            skipped += 1
            continue

        print(f"{prefix} -> 翻译中 (task_id={task_id}) ...", flush=True)
        try:
            resp = translate_file(args.base_url, path, task_id, args.timeout)
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
