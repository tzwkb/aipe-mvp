# TM Exact Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 TM exact commit 的六项审查问题并完成结构拆分。

**Architecture:** 把 prompt、输出解析和 TM exact 解析从 `TranslationPipeline` 抽离。RAG 层批量分页查询，pipeline 在完整语义单元上翻译并覆盖锁定结果。

**Tech Stack:** Python 3.12、FastAPI、Pydantic 2、Qdrant async client、pytest、Docker Compose。

## Global Constraints

- 保持现有 HTTP/CLI 参数与响应字段兼容。
- `use_tm_exact_match=False` 行为不变。
- 使用项目原生 `python3 scripts/run_checks.py` 验证。
- 不提交、不推送；用户未要求修改 Git 历史。

---

### Task 1: Target-aware prompt policy

**Files:**
- Create: `aipe/app/services/translation_prompts.py`
- Modify: `aipe/app/services/style_guide_service.py`
- Modify: `aipe/app/services/translation_pipeline.py`
- Test: `aipe/tests/test_pipeline.py`

**Interfaces:**
- `content_type_mode(content_type: ContentType) -> ContentTypeMode`
- `build_single_messages(..., target_lang: str) -> list[dict[str, str]]`
- 等价的 group/dialog builders。

- [x] 写失败测试：EN→DE prompt 不含英语专属规则；`QUEST_DESCRIPTION` 使用 contextual 规则。
- [x] 用 Docker pytest 运行两个测试，确认分别因当前全局英语规则和错误类型集合失败。
- [x] 在 style guide 层增加规范模式映射，并将 prompt 构建迁入新模块。
- [x] 运行 prompt 与现有 pipeline 测试至通过。

### Task 2: Progress context migration

**Files:**
- Modify: `aipe/app/utils/progress_tracker.py`
- Test: `aipe/tests/test_progress_tracker.py`

**Interfaces:**
- `_normalize_stored_context(context: dict[str, Any]) -> dict[str, Any]` 将缺失 TM flag 补为 `False`。

- [x] 写失败测试：旧 context + 新请求 `False` 可以续传；新请求 `True` 仍失败。
- [x] 运行目标测试并确认 `context.use_tm_exact_match` mismatch。
- [x] 在比较边界迁移旧 context，不改写现有进度文件。
- [x] 运行 progress 与 batch processor 测试至通过。

### Task 3: Complete batched exact lookup

**Files:**
- Modify: `aipe/app/services/rag_service.py`
- Create: `aipe/tests/test_rag_exact.py`
- Verify: `aipe/tests/test_rag.py`

**Interfaces:**
- `find_exact_source_matches_many(sources, collection=None, top_k=None) -> dict[str, list[RAGSearchResult]]`
- `find_exact_source_matches()` 委托批量接口。

- [x] 写 fake-client 失败测试：第二页 Designer candidate 必须胜过第一页 Done candidates。
- [x] 写失败测试：多个 source 共用一次分页查询，不按 source 发 N 次 scroll。
- [x] 实现 MatchAny 分页、全候选排序及 source keyword payload index。
- [x] 运行 unit 和 Qdrant integration 测试至通过。

### Task 4: Preserve group/dialog context

**Files:**
- Create: `aipe/app/services/translation_memory.py`
- Modify: `aipe/app/services/translation_pipeline.py`
- Modify: `aipe/tests/test_pipeline.py`

**Interfaces:**
- `TMExactMatchResolver.resolve_many(...) -> list[TranslationResult | None]`
- `merge_exact_results(exact, generated) -> list[TranslationResult]`
- prompt builder 接收可选 locked results。

- [x] 写失败测试：部分 dialog 命中时 prompt 仍包含全部 source、speaker 和锁定译文。
- [x] 修改现有 group 测试，要求完整 source 仍在 prompt。
- [x] 实现批量 resolver；全命中短路，部分命中继续完整核心流程并最终覆盖。
- [x] 运行 single/group/dialog TM 测试至通过。

### Task 5: Recursive CLI identity

**Files:**
- Modify: `aipe/scripts/batch_translate_files.py`
- Modify: `aipe/tests/test_ops_scripts.py`

**Interfaces:**
- `build_file_job(path, src_dir, out_dir, task_prefix, recursive) -> tuple[str, Path]`

- [x] 写失败测试：`a/foo.xlsx` 与 `b/foo.xlsx` 产生不同 task ID 和输出路径。
- [x] 写兼容测试：非递归 `foo.xlsx` 仍得到 `prefix + foo`。
- [x] 实现安全相对路径标识和稳定短哈希，并让 main 使用统一 helper。
- [x] 运行 ops tests 至通过。

### Task 6: Output parsing and test decomposition

**Files:**
- Create: `aipe/app/services/translation_output.py`
- Modify: `aipe/app/services/translation_pipeline.py`
- Create: `aipe/tests/pipeline_fakes.py`
- Modify: `aipe/tests/test_pipeline.py`

**Interfaces:**
- `parse_single_llm_output(raw) -> tuple[str, str | None]`
- `parse_numbered_output(raw, expected) -> list[tuple[str, str | None]] | None`
- `parse_numbered_output_positional(raw, expected) -> list[...] | None`

- [x] 在现有测试全绿后移动纯解析函数，保留现有行为测试。
- [x] 把完整 fake schema 移入共享测试夹具模块，减少主测试文件体积。
- [x] 运行拆分后的目标测试；确认 `translation_pipeline.py` 与每个测试文件都低于 1000 行。

### Task 7: Full verification

**Files:** 所有本计划修改文件。

- [x] 运行 `python3 scripts/run_checks.py`，要求全部测试通过。
- [x] 运行 `git diff --check`，要求无 whitespace error。
- [x] 运行行数、导入和 `git status` 检查。
- [x] 复核六项审查问题各有一个失败过、现已通过的回归测试。
