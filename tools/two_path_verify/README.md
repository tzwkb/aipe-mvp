# 两条路径产出一致性验证 — 准备状态

## 0. 待定（阻塞项，跑之前必须先定）

| # | 问题 | 现状 | 影响 |
|---|---|---|---|
| 1 | LLM 不一致 | 线上 AIPE 用 `deepseek-chat`（api.deepseek.com）；本地 AIPE 用 `claude-sonnet-5`（api.relayrouter.ai） | 译文文本必然不同。要"逐字一致"就必须把本地切到 deepseek-chat |
| 2 | 代码不一致 | 线上容器 = `langlobal_localization` commit `3cd88e10a897`；本地 = 工作区（有多处未提交改动） | 592/89/263/284/114/339 行分叉，见第 3 节 |
| 3 | 项目资源不一致 | 线上 `project_3ae0457a0ddd`（13 术语 + 风格指南）、`project_d0a5c9dffb6f`（仅 profile）；本地 `wwm/zh-en`、`lom/zh-en`、`tuyou/zh-en` | 术语/风格/RAG 不同 → 译文不同 |
| 4 | RAG 语料不一致 | 线上项目 collection 均 0 points；本地 `yanyun_corpus` 有数据 | 开 RAG 时召回源不同 |
| 5 | TM 默认值不一致 | 工作台 `/api/translate/run` 默认 `use_tm_exact_match=true`；AIPE 原生默认 `false` | 本地直调 AIPE 需显式对齐 |

## 1. 线上侧（路径 2）

- 入口：`https://www.langlobal-tech.com/localization-tool/api`
- 鉴权：**未启用**（`/api/files`、`/api/projects` 无 token 返回 200；`/auth/login` 404）
- 链路：`POST /files/upload` → `POST /translate/run` → 响应 `document.segments[].target`，或 `POST /files/{document_id}/export`
- 现有项目：`project_3ae0457a0ddd`(Test)、`project_d0a5c9dffb6f`(eric)
- 线上 AIPE：容器 `langlobalAITransAIPE`，v1.2.0，镜像 tag `3cd88e10a897`，Qdrant 容器内，LLM `deepseek-chat`
- 部署态：`/home/langlobal_localization`，git clean，HEAD `b42ca338c139`
- 单次上限 5000 段；上传上限 32 MiB / 50000 段

## 2. 本地侧（路径 1）

- 入口：`http://localhost:8000/api/v1`，容器 `aipe-api-1`（Up 3 天），**代码与当前工作区逐文件哈希一致**
- v1.1.0，LLM `claude-sonnet-5`，Qdrant `:6333`
- Qdrant collections：`yanyun_corpus`、`isekai_de_corpus`、`isekai_fr_corpus`、`lom_reference_en_sparse_v1`
- 项目 profile：`data/projects/{wwm,lom,tuyou,chaoyang,isekai,isekai-kimetsu,...}/zh-en`
- 备用入口：`aipe/scripts/batch_translate_files.py`（目录批处理 → CSV）

## 3. 代码分叉（部署版 `3cd88e10a897` vs 本地工作区）

| 文件 | 差异行数 |
|---|---|
| `app/services/translation_pipeline.py` | 592 |
| `app/utils/file_parser.py` | 339 |
| `app/services/rag_service.py` | 284 |
| `app/services/llm_service.py` | 263 |
| `app/api/translate.py` | 114 |
| `app/services/translation_prompts.py` | 89 |
| `app/schemas/translate.py` | 32 |

部署版独有：`api/chat.py`、`api/knowledge_base.py`、`api/projects.py`、`observability.py`、
`schemas/{chat,knowledge_base,projects}.py`、`services/{chat_service,knowledge_base_service,language_prompts,languages,rag_library_service,translation_web_support}.py`、`utils/knowledge_parser.py`。

## 4. 脚本

| 脚本 | 作用 |
|---|---|
| `online_workbench.py` | 上传文件 → 触发工作台翻译 → 落 `online_raw.json` + `online_normalized.json` |
| `aipe_run.py` | 对指定 AIPE 跑同一批文本（`texts` 模式）或同一文件（`file` 模式）→ `local_*.json` |
| `compare.py` | 按原文对齐两份结果，出 `compare_report.md/json`（逐字一致率、宽松一致率、六分类一致率、术语一致率、标签/数字完整性） |

### 用法

```bash
cd tools/two_path_verify

python3 online_workbench.py <文件> --project-id project_3ae0457a0ddd --out out_online

python3 aipe_run.py out_online/online_normalized.json --mode texts \
  --project-id wwm/zh-en --base-url http://localhost:8000/api/v1 --out out_local

python3 compare.py out_online/online_normalized.json out_local/local_normalized.json --out report
```

准备阶段可先使用 `--dry-run`。它只检查输入文件、模式和请求参数，不上传文件、不调用 AIPE；线上脚本会标记该真实流程会写线上数据库：

```bash
python3 online_workbench.py <文件> --project-id project_3ae0457a0ddd --dry-run
python3 aipe_run.py <文件> --mode texts --project-id wwm/zh-en --dry-run
```

`compare.py` 将标签、变量和数字完整性作为硬约束指标，将逐字/宽松文本、分类和术语作为软指标；逐字一致不是唯一通过标准。

### 引擎层直连线上 AIPE（可选，用于剥离工作台层差异）

```bash
ssh -N -L 18008:127.0.0.1:8008 langlobal-server
python3 aipe_run.py out_online/online_normalized.json --mode texts \
  --base-url http://localhost:18008/api/v1 --project-id <线上项目> --out out_online_engine
```

## 5. 未验证

- 三个脚本仅通过语法检查，**未做过端到端实跑**（等文件 + 授权）
- 线上 `translate/run` 会写线上数据库（新增文档与译文），不是纯只读操作
