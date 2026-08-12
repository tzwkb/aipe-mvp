# TM Exact Review Fixes Design

## Goal

修复 `de3cb93` 审查确认的六项问题，同时保持现有 FastAPI、CLI 参数和默认关闭 TM 精确匹配时的行为兼容。

## Approved Scope

用户已在审查结论后明确要求“全部修复”。本设计仅覆盖以下项目：

1. 提示规则必须感知目标语言，内容类型归属只有一个规范来源。
2. 升级前的进度文件缺失 `use_tm_exact_match` 时，默认按 `False` 兼容。
3. 精确 TM 查询必须批量、分页，并在所有候选中选择最高审校等级。
4. group/dialog 部分命中时保留完整语义单元和说话人上下文。
5. 递归批处理对同名文件生成稳定且唯一的任务和输出名。
6. 拆分超大 pipeline 和测试文件，不改变公共调用方式。

## Considered Approaches

### A. 原地补丁

在现有分支中增加条件、循环和兼容判断。改动最小，但会继续放大 1512 行 pipeline、重复 group/dialog 分支，也无法解决结构审查问题。

### B. 定向抽取（采用）

抽取提示构建、LLM 输出解析和 TM 精确匹配三个聚焦模块；pipeline 只保留编排。RAG 服务提供一次批量精确查询，group/dialog 在完整输入上执行并覆盖锁定译文。该方案能删除重复逻辑，且不需要重写现有调度器。

### C. 状态机重写

把整条翻译流水线改造成统一阶段状态机。长期可能更整齐，但影响单句、分组、对话、Web Search 和错误降级全部路径，超出本次修复范围。

## Architecture

### Prompt policy

`style_guide_service.py` 负责 `ContentType` 到 `functional/contextual/neutral` 的规范映射，`QUEST_DESCRIPTION` 归入 contextual。新模块 `translation_prompts.py` 根据 `content_type` 和项目 `target_lang` 生成提示：通用规则对所有语言生效；em dash、英文省略号、straight punctuation 和 contraction 只对英语目标生效。legacy 无项目路径继续按现有 ZH→EN 默认处理。

### Exact TM lookup

`RAGService.find_exact_source_matches_many()` 接收去重后的 source 列表，用一个 `MatchAny` filter 分页 scroll，收齐所有候选后再按 status 排序。collection 的 `source` payload 建立 keyword index。单句方法委托给批量方法，保持旧接口。

`translation_memory.py` 把 RAG 命中转换成 `TranslationResult`，统一异常降级和结果覆盖。它不负责 LLM 编排。

### Context preservation

全命中直接返回 TM，不调用 LLM。部分命中不再递归删除已命中项；完整 sources/speakers 继续进入原 group/dialog 流程，并在 prompt 中附带锁定 TM 译文。LLM 返回或错误结果生成后，仅用 TM 结果覆盖命中位置。

### Progress compatibility

进度上下文比较前应用字段默认值。旧 context 缺失 `use_tm_exact_match` 等价于 `False`；显式 `True` 仍判定为不兼容，避免混合结果。

### Recursive CLI identity

非递归模式保留原有 stem 任务名。递归模式使用相对路径的可读安全形式和短哈希生成 task ID；输出文件使用同一 ID，确保同名文件不碰撞。前缀也进行路径安全化。

### Decomposition

- `translation_prompts.py`: prompt 常量、项目语言 system prompt、single/group/dialog message 构建与参考段格式化。
- `translation_output.py`: 单句和编号 LLM 输出解析。
- `translation_memory.py`: exact-match 解析、锁定段格式和结果覆盖。
- `translation_pipeline.py`: 仅保留资源解析、检索/Web Search、LLM 编排和错误策略。
- 共享 pipeline fakes 抽成独立夹具模块，主测试文件与所有新增测试文件保持低于 1000 行。

## Error Handling

- Qdrant exact lookup 失败：记录一次 warning，整批回退常规翻译。
- 批量 exact 分页必须完整结束；不会用固定倍数截断。
- 部分 dialog LLM 失败：命中位置仍返回 TM success，未命中位置保留既有 dialog error。
- 旧进度只有新增默认字段缺失时兼容；其他上下文差异继续拒绝。

## Testing

每项先加入失败回归测试，再实现：非英语 prompt、任务描述分类、旧进度兼容、分页优先级、单次批量查询、完整对话上下文、递归同名文件。重构后运行项目原生 Docker 检查、`git diff --check` 和文件行数门禁。
