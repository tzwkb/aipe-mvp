#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import re
import sys
import unicodedata
from pathlib import Path

TAG_PATTERNS = [
    re.compile(r"<[^<>]{1,40}>"),
    re.compile(r"\{[^{}]{1,40}\}"),
    re.compile(r"%\d*\$?[a-zA-Z]"),
    re.compile(r"\[[^\[\]]{1,40}\]"),
    re.compile(r"#(?:G|C|Y|E)(?=$|[^A-Za-z])"),
    re.compile(r"\\n"),
]


def norm_text(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def loose(s):
    s = norm_text(s) or ""
    s = re.sub(r"[\s]+", "", s)
    s = re.sub(r"[.,!?;:。，！？；：、…\"'()（）\-—]", "", s)
    return s.lower()


def tokens(s):
    out = []
    for pat in TAG_PATTERNS:
        out.extend(pat.findall(s or ""))
    nums = re.findall(r"\d+(?:[.,]\d+)?", s or "")
    return sorted(out), sorted(nums)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_items(norm):
    by_source = {}
    for i, item in enumerate(norm["items"]):
        by_source.setdefault(norm_text(item.get("source")), []).append((i, item))
    return by_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("online")
    p.add_argument("local")
    p.add_argument("--out", default="./report")
    p.add_argument("--label-online", default="online_workbench")
    p.add_argument("--label-local", default="local_aipe")
    args = p.parse_args()

    a = load(args.online)
    b = load(args.local)
    b_index = index_items(b)

    pairs, missing = [], []
    for item in a["items"]:
        key = norm_text(item.get("source"))
        cands = b_index.get(key)
        if not cands:
            missing.append(item.get("source"))
            continue
        pairs.append((item, cands.pop(0)[1]))

    exact = same_loose = 0
    diffs = []
    ct_agree = ct_total = 0
    term_agree = term_total = 0
    tag_issues = []
    hard_pass = 0
    for oi, li in pairs:
        ot, lt = oi.get("target"), li.get("target")
        if ot == lt:
            exact += 1
        elif loose(ot) == loose(lt):
            same_loose += 1
        else:
            diffs.append(
                {
                    "source": oi.get("source"),
                    "online": ot,
                    "local": lt,
                    "online_reason": oi.get("reason"),
                    "local_reason": li.get("reason"),
                }
            )
        oct_, lct = oi.get("content_type"), li.get("content_type")
        if oct_ or lct:
            ct_total += 1
            ct_agree += int(oct_ == lct)
        otm = sorted(t.get("target") or t.get("source") for t in (oi.get("terminology_used") or []))
        ltm = sorted(t.get("target") or t.get("source") for t in (li.get("terminology_used") or []))
        if otm or ltm:
            term_total += 1
            term_agree += int(otm == ltm)
        otags, onums = tokens(ot)
        ltags, lnums = tokens(lt)
        hard_ok = otags == ltags and onums == lnums
        if hard_ok:
            hard_pass += 1
        else:
            tag_issues.append(
                {
                    "source": oi.get("source"),
                    "online_missing_tokens": dict(Counter(otags + onums) - Counter(ltags + lnums)),
                    "local_missing_tokens": dict(Counter(ltags + lnums) - Counter(otags + onums)),
                }
            )

    n = len(pairs)
    report = {
        "online_source": args.online,
        "local_source": args.local,
        "online": {
            "side": a.get("side"),
            "base_url": a.get("base_url"),
            "project_id": a.get("aipe_project_id") or a.get("requested_project_id"),
            "task_id": a.get("task_id"),
            "flags": a.get("flags"),
            "items": len(a["items"]),
        },
        "local": {
            "side": b.get("side"),
            "base_url": b.get("base_url"),
            "project_id": b.get("project_id"),
            "task_id": b.get("task_id"),
            "flags": b.get("flags"),
            "mode": b.get("mode"),
            "items": len(b["items"]),
        },
        "metrics": {
            "online_items": len(a["items"]),
            "local_items": len(b["items"]),
            "paired": n,
            "online_only_sources": len(missing),
            "exact_equal": exact,
            "exact_equal_pct": round(exact / n * 100, 2) if n else None,
            "loose_equal": same_loose,
            "different": len(diffs),
            "content_type_agreement": f"{ct_agree}/{ct_total}" if ct_total else "n/a",
            "terminology_agreement": f"{term_agree}/{term_total}" if term_total else "n/a",
            "token_integrity_issues": len(tag_issues),
            "hard_constraint_pass": hard_pass,
            "hard_constraint_pass_pct": round(hard_pass / n * 100, 2) if n else None,
            "soft_text_equal_or_loose": exact + same_loose,
            "soft_text_equal_or_loose_pct": round((exact + same_loose) / n * 100, 2) if n else None,
        },
    }

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "compare_report.json").write_text(
        json.dumps({**report, "differences": diffs, "token_issues": tag_issues, "online_only_sources": missing},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 两条路径产出一致性报告",
        "",
        f"- 线上：{report['online']['base_url']} | project={report['online']['project_id']} | task={report['online']['task_id']} | items={len(a['items'])}",
        f"- 本地：{report['local']['base_url']} | project={report['local']['project_id']} | task={report['local']['task_id']} | mode={report['local']['mode']} | items={len(b['items'])}",
        "",
        "## 指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for k, v in report["metrics"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 差异明细", ""]
    if diffs:
        lines += ["| 原文 | 线上译文 | 本地译文 |", "|---|---|---|"]
        for d in diffs[:200]:
            lines.append(f"| {d['source']} | {d['online']} | {d['local']} |")
        if len(diffs) > 200:
            lines.append(f"| … 其余 {len(diffs) - 200} 条见 JSON | | |")
    else:
        lines.append("无差异。")
    if tag_issues:
        lines += ["", "## 标签/数字不一致", "", "| 原文 | 线上缺失 | 本地缺失 |", "|---|---|---|"]
        for t in tag_issues[:100]:
            lines.append(f"| {t['source']} | {t['online_missing_tokens']} | {t['local_missing_tokens']} |")
    (outdir / "compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"report -> {outdir}/compare_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
