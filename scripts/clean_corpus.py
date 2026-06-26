"""清洗 5 个语料库 → 合并 → 去重（含来源标注）→ 输出 corpus.jsonl

规则：
- 来源优先级：Designer > CQA > LQA > GT > Stable
- 同一 (source, target) 完全重复 → 只保留优先级最高的来源
- 同一 source 不同 target → 全部保留（B 方案：让 RAG 按 status 加权）
- 删除：source 或 target 为空 / source == target / 纯数字/纯标点 / source 长度 > 500
- status 字段映射统一为 AIPE 已识别的标签
"""

import pandas as pd
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"E:\Langlobal\AIPEMVP_0526")
OUT_DIR = BASE / "cleaned_corpus"
OUT_DIR.mkdir(exist_ok=True)

# 来源优先级（数字越小越优先）
SOURCE_PRIORITY = {
    "Designer": 1,
    "CQA": 2,
    "LQA": 3,
    "GT": 4,
    "Stable": 5,
}

# 各库的状态映射（统一成 AIPE 已识别标签）
STATUS_MAP = {
    "Designer": "Designer Reviewed",
    "CQA": "CQA_Done",
    "LQA": "Done_LQA edited",
    "GT": "Done",
    "Stable": "Done",
}

# 文件列表（按优先级顺序排）
FILES = [
    ("Designer", "Designer库0526.xlsx"),
    ("CQA",      "CQA库0526.xlsx"),
    ("LQA",      "LQA库0526.xlsx"),
    ("GT",       "GT库0526.xlsx"),
    ("Stable",   "Stable库0526.xlsx"),
]

# 列约定（按 inspect 结果）：col0=ID col1=中文 col2=英文 col3=原始status [col4=branch]
COL_ID, COL_ZH, COL_EN, COL_STATUS = 0, 1, 2, 3

JUNK_RE = re.compile(r"^[\s\d\W]+$")  # 纯数字/纯标点/纯空白

def is_junk(text: str) -> bool:
    if not text:
        return True
    if JUNK_RE.match(text):
        return True
    return False

def normalize(text) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip()


report = []
def log(msg: str):
    print(msg)
    report.append(msg)


log("="*70)
log("步骤 1：逐文件读取 + 行级过滤")
log("="*70)

# all_rows[(zh, en)] = {"source": "...", "status": "...", "id": "..."}
all_rows = {}
stats = {}

for source_name, fname in FILES:
    fpath = BASE / fname
    log(f"\n[{source_name}] 读取 {fname}")
    df = pd.read_excel(fpath, sheet_name="Sheet1", header=None)
    raw_n = len(df)
    log(f"  原始行数: {raw_n}")

    kept = 0
    dropped_empty = 0
    dropped_same = 0
    dropped_junk = 0
    dropped_too_long = 0
    dup_in_file = 0
    upgraded = 0  # 同一 (zh,en) 被更高优先级覆盖
    skipped_lower = 0  # 当前条目被更高优先级挡住

    for _, row in df.iterrows():
        if len(row) < 4:
            continue
        ident = normalize(row.iloc[COL_ID])
        zh = normalize(row.iloc[COL_ZH])
        en = normalize(row.iloc[COL_EN])

        # 行级过滤
        if not zh or not en:
            dropped_empty += 1
            continue
        if zh == en:
            dropped_same += 1
            continue
        if is_junk(zh) or is_junk(en):
            dropped_junk += 1
            continue
        if len(zh) > 500:
            dropped_too_long += 1
            continue

        key = (zh, en)
        new_pri = SOURCE_PRIORITY[source_name]

        if key in all_rows:
            old_pri = SOURCE_PRIORITY[all_rows[key]["source_db"]]
            if new_pri < old_pri:
                # 当前来源更高优先级，覆盖
                all_rows[key] = {
                    "source_db": source_name,
                    "status": STATUS_MAP[source_name],
                    "id": ident,
                }
                upgraded += 1
            else:
                # 当前来源优先级低或相等，跳过
                if new_pri == old_pri:
                    dup_in_file += 1
                else:
                    skipped_lower += 1
            continue

        all_rows[key] = {
            "source_db": source_name,
            "status": STATUS_MAP[source_name],
            "id": ident,
        }
        kept += 1

    stats[source_name] = {
        "raw": raw_n,
        "kept_first_time": kept,
        "dropped_empty": dropped_empty,
        "dropped_same": dropped_same,
        "dropped_junk": dropped_junk,
        "dropped_too_long": dropped_too_long,
        "dup_in_file": dup_in_file,
        "upgraded_over_existing": upgraded,
        "skipped_lower_priority": skipped_lower,
    }
    log(f"  本文件新增: {kept}")
    log(f"  覆盖已有(更高优先级): {upgraded}")
    log(f"  被挡住(已有更高优先级): {skipped_lower}")
    log(f"  同库重复: {dup_in_file}")
    log(f"  过滤掉 - 空值: {dropped_empty}, 中英相同: {dropped_same}, 垃圾: {dropped_junk}, 超长: {dropped_too_long}")

log("\n" + "="*70)
log("步骤 2：聚合统计")
log("="*70)
log(f"\n最终保留唯一 (zh,en) 对: {len(all_rows)}")

# 按 status 统计
from collections import Counter
by_status = Counter(v["status"] for v in all_rows.values())
log(f"\n按 status 分布:")
for s, n in sorted(by_status.items(), key=lambda x: -x[1]):
    log(f"  {s:25s}: {n:>8}")

by_source = Counter(v["source_db"] for v in all_rows.values())
log(f"\n按来源库分布:")
for s, n in sorted(by_source.items(), key=lambda x: SOURCE_PRIORITY[x[0]]):
    log(f"  {s:10s}: {n:>8}")

# 同 zh 不同 en（多译法）统计
from collections import defaultdict
zh_variants = defaultdict(set)
for (zh, en) in all_rows.keys():
    zh_variants[zh].add(en)
multi = {zh: ens for zh, ens in zh_variants.items() if len(ens) > 1}
log(f"\n同一中文有多种译法的条数: {len(multi)}")
log(f"  多译法占总条数比例: {sum(len(v) for v in multi.values())/len(all_rows)*100:.2f}%")

log("\n" + "="*70)
log("步骤 3：写出 cleaned_corpus.jsonl 和 cleaned_corpus.xlsx")
log("="*70)

out_jsonl = OUT_DIR / "cleaned_corpus.jsonl"
out_xlsx = OUT_DIR / "cleaned_corpus.xlsx"

rows_out = []
with open(out_jsonl, "w", encoding="utf-8") as f:
    for (zh, en), meta in all_rows.items():
        rec = {
            "source": zh,
            "target": en,
            "status": meta["status"],
            "source_db": meta["source_db"],
            "id": meta["id"],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rows_out.append(rec)

# 也输出 Excel 方便人工抽检
df_out = pd.DataFrame(rows_out)
df_out.to_excel(out_xlsx, index=False)

log(f"\n输出文件:")
log(f"  JSONL: {out_jsonl}  ({out_jsonl.stat().st_size/1024/1024:.1f} MB)")
log(f"  XLSX:  {out_xlsx}  ({out_xlsx.stat().st_size/1024/1024:.1f} MB)")

# 写报告
report_path = OUT_DIR / "clean_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
    f.write("\n\n详细统计:\n")
    for src, s in stats.items():
        f.write(f"\n[{src}]\n")
        for k, v in s.items():
            f.write(f"  {k}: {v}\n")
log(f"  报告:  {report_path}")
