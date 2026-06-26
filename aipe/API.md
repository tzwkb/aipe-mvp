# 燕云十六声 AI 翻译服务 — API 文档

**版本**: 1.0.0  
**服务名称**: `yanyun-ai-translate`  
**交互式文档**: 启动后访问 `http://localhost:8000/docs`

---

## 目录

- [概述](#概述)
- [Base URL](#base-url)
- [通用约定](#通用约定)
- [错误处理](#错误处理)
- [Health — 健康检查](#health--健康检查)
- [Translate — 翻译](#translate--翻译)
  - [POST /translate — 单句/批量翻译](#post-translate--单句批量翻译)
  - [POST /translate/file — 文件上传翻译](#post-translatefile--文件上传翻译)
  - [GET /translate/task/{task_id} — 查询任务状态](#get-translatetasktask_id--查询任务状态)
  - [GET /translate/task/{task_id}/csv — 导出 CSV](#get-translatetasktask_idcsv--导出-csv)
- [Terminology — 术语表](#terminology--术语表)
  - [POST /terminology/upload — 上传术语表](#post-terminologyupload--上传术语表)
  - [GET /terminology — 查询术语表](#get-terminology--查询术语表)
- [RAG — 双语语料库](#rag--双语语料库)
  - [POST /rag/corpus/upload — 上传语料](#post-ragcorpusupload--上传语料)
  - [POST /rag/search — 手动检索测试](#post-ragsearch--手动检索测试)
- [Style Guide — 风格指南](#style-guide--风格指南)
  - [POST /style-guide/upload — 上传风格指南](#post-style-guideupload--上传风格指南)
  - [GET /style-guide — 查询风格指南](#get-style-guide--查询风格指南)
- [数据模型](#数据模型)
- [文件格式说明](#文件格式说明)
- [环境变量配置](#环境变量配置)

---

## 概述

本服务是基于 FastAPI 构建的中→英游戏本地化翻译服务，专为《燕云十六声》设计。翻译流水线集成了：

- **术语表（Terminology）**：确保角色名、地名、技能等专有名词译名一致
- **RAG（检索增强生成）**：从历史双语语料中检索相似句子作为 LLM 参考
- **风格指南（Style Guide）**：约束译文的语言风格（武侠 / 江湖 / 古风调性）
- **结构聚类（Cluster）**：将句式高度相似的句子合并为一次 LLM 调用，保证模板词一致性
- **对话模式（Dialog Mode）**：按对话 ID 聚合，整段对话一次 LLM 调用，保持上下文连贯

---

## Base URL

```
http://localhost:8000/api/v1
```

所有业务接口均以 `/api/v1` 为前缀。根路径 `GET /` 返回服务信息。

---

## 通用约定

| 项目 | 说明 |
|------|------|
| 协议 | HTTP/1.1 |
| 请求编码 | `application/json`（JSON 接口）或 `multipart/form-data`（文件上传接口）|
| 响应编码 | `application/json`，UTF-8 |
| 时间戳 | ISO 8601 字符串 |
| 布尔值 | JSON `true` / `false` |
| 空值 | JSON `null` |

---

## 错误处理

所有错误均返回统一 JSON 结构：

```json
{
  "error": "<error_code>",
  "detail": "<human-readable message>"
}
```

| HTTP 状态码 | error_code | 含义 |
|------------|-----------|------|
| 400 | `validation_failed` 或具体业务描述 | 请求参数错误、文件格式不支持、内容为空等 |
| 404 | - | 任务 ID 不存在 |
| 413 | - | 单次请求文本数超过 5000 句上限 |
| 422 | `validation_failed` | Pydantic 字段校验失败（类型错误、范围越界等）|
| 500 | `translation_failed` | LLM 调用失败、向量库异常等翻译流程内部错误 |

**422 错误示例**：

```json
{
  "error": "validation_failed",
  "detail": [
    {
      "loc": ["body", "rag_threshold"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## Health — 健康检查

### GET /health

检查服务是否正常运行。

**请求**：无参数

**响应** `200 OK`：

```json
{
  "status": "ok",
  "service": "yanyun-ai-translate",
  "version": "1.0.0"
}
```

**示例**：

```bash
curl http://localhost:8000/api/v1/health
```

---

## Translate — 翻译

### POST /translate — 单句/批量翻译

以 JSON 请求体提交待翻译文本列表，同步阻塞直到所有文本翻译完成后返回结果。单次请求上限 **5000 句**；超过请改用 `/translate/file`。

**请求体** `application/json`：

| 字段 | 类型 | 必填 | 默认 | 描述 |
|------|------|------|------|------|
| `texts` | `string[]` | 是 | — | 待翻译中文文本列表 |
| `source_lang` | `string` | 否 | `"zh"` | 源语言 |
| `target_lang` | `string` | 否 | `"en"` | 目标语言 |
| `batch_size` | `integer` | 否 | `50` | 每批次并发处理句数，范围 `[1, 500]` |
| `enable_rag` | `boolean` | 否 | `true` | 是否启用 RAG 检索增强 |
| `rag_threshold` | `float` | 否 | `0.85` | RAG 相似度阈值，范围 `[0.0, 1.0]` |
| `rag_top_k` | `integer` | 否 | `3` | RAG 返回最相似的前 K 条，范围 `[1, 10]` |
| `rag_collection` | `string \| null` | 否 | `null` | 指定 Qdrant collection 名，`null` 使用配置默认值 |
| `task_id` | `string \| null` | 否 | `null` | 自定义任务 ID，用于断点续传；不填则自动生成 16 位 hex |
| `content_types` | `(string \| null)[] \| null` | 否 | `null` | 与 `texts` 一一对应的内容类型预标注；有值则跳过 LLM 自动分类 |

**响应** `200 OK` → [BatchTranslateResponse](#batchtranslateresponse)

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["剑出鞘，江湖乱。", "此去经年，再难相见。"],
    "enable_rag": true,
    "rag_threshold": 0.8,
    "rag_top_k": 3
  }'
```

```json
{
  "task_id": "a3f2c1b0e9d84f12",
  "total": 2,
  "completed": 2,
  "progress_pct": 100.0,
  "status": "completed",
  "results": [
    {
      "source": "剑出鞘，江湖乱。",
      "translation": "The sword leaves its scabbard; the jianghu falls into turmoil.",
      "status": "success",
      "content_type": "narrative",
      "terminology_used": [],
      "rag_references": [
        {
          "source": "剑出，天下变。",
          "target": "The sword drawn, the world changes.",
          "score": 0.87,
          "status": "Designer Reviewed"
        }
      ],
      "error_msg": null
    },
    {
      "source": "此去经年，再难相见。",
      "translation": "After this parting of years, it will be hard to meet again.",
      "status": "success",
      "content_type": "dialog",
      "terminology_used": [],
      "rag_references": null,
      "error_msg": null
    }
  ]
}
```

---

### POST /translate/file — 文件上传翻译

上传文件进行批量翻译，支持 `.txt` / `.csv` / `.xlsx` / `.xls` / `.json` 格式。支持**对话模式**（Dialog Mode），可整段对话一次 LLM 调用。

**请求体** `multipart/form-data`：

| 字段 | 类型 | 必填 | 默认 | 描述 |
|------|------|------|------|------|
| `file` | `file` | 是 | — | 待翻译文件，支持 `.txt/.csv/.xlsx/.xls/.json` |
| `batch_size` | `integer` | 否 | `50` | 每批次处理句数，范围 `[1, 500]` |
| `enable_rag` | `boolean` | 否 | `true` | 是否启用 RAG |
| `rag_threshold` | `float` | 否 | `0.85` | RAG 相似度阈值，范围 `[0.0, 1.0]` |
| `rag_top_k` | `integer` | 否 | `3` | RAG Top-K，范围 `[1, 10]` |
| `enable_cluster` | `boolean` | 否 | `true` | 是否启用结构聚类（整组翻译路径）；关闭后全部走单句路径 |
| `dialog_mode` | `boolean` | 否 | `false` | 对话模式：按 `id` 聚合，按 `time` 排序，整段一次 LLM 调用。需文件含 `id`/`说话人`/`time` 列 |
| `rag_collection` | `string \| null` | 否 | `null` | 指定 Qdrant collection 名 |
| `task_id` | `string \| null` | 否 | `null` | 自定义任务 ID（断点续传） |

**响应** `200 OK` → [BatchTranslateResponse](#batchtranslateresponse)

**错误**：
- `400` — 文件格式不支持、文件内容为空、`dialog_mode=true` 但文件无 `id` 列

**示例**：

```bash
# 普通文件翻译
curl -X POST http://localhost:8000/api/v1/translate/file \
  -F "file=@texts.xlsx" \
  -F "enable_rag=true" \
  -F "enable_cluster=true"

# 对话模式（文件需含 id/说话人/time 列）
curl -X POST http://localhost:8000/api/v1/translate/file \
  -F "file=@dialogs.xlsx" \
  -F "dialog_mode=true" \
  -F "task_id=my_dialog_task_001"
```

---

### GET /translate/task/{task_id} — 查询任务状态

查询已提交任务的进度和结果（支持断点续传场景）。

**路径参数**：

| 参数 | 类型 | 描述 |
|------|------|------|
| `task_id` | `string` | 任务 ID |

**响应** `200 OK` → [BatchTranslateResponse](#batchtranslateresponse)

**错误**：
- `404` — 任务不存在

**示例**：

```bash
curl http://localhost:8000/api/v1/translate/task/a3f2c1b0e9d84f12
```

---

### GET /translate/task/{task_id}/csv — 导出 CSV

将指定任务的翻译结果导出为 CSV 文件（UTF-8 BOM，Excel 可直接打开）。

**路径参数**：

| 参数 | 类型 | 描述 |
|------|------|------|
| `task_id` | `string` | 任务 ID |

**响应** `200 OK`：

- Content-Type: `text/csv; charset=utf-8`
- Content-Disposition: `attachment; filename="{task_id}.csv"`

**CSV 列说明**：

| 列名 | 说明 |
|------|------|
| `source` | 原文 |
| `translation` | 译文 |
| `status` | `success` 或 `error` |
| `content_type` | AI 预分类的文本类型 |
| `terminology_used` | 命中的术语（JSON 字符串） |
| `rag_references` | RAG 参考例句（JSON 字符串） |
| `error_msg` | 错误信息（失败时） |

**错误**：
- `404` — 任务不存在

**示例**：

```bash
curl -o result.csv http://localhost:8000/api/v1/translate/task/a3f2c1b0e9d84f12/csv
```

---

## Terminology — 术语表

服务启动时会自动从 `data/terminology/` 目录加载术语表。可通过 API 动态上传追加/更新。

### POST /terminology/upload — 上传术语表

上传术语表文件，新术语追加入库，已存在的中文词条自动更新目标译文。

**请求体** `multipart/form-data`：

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `file` | `file` | 是 | 术语表文件，支持 `.xlsx` / `.xls` / `.csv` |

**文件列名要求**（自动识别以下别名，大小写不敏感）：

| 列 | 识别的列名 |
|----|-----------|
| 中文原文（必填）| `中文`、`原文`、`source`、`src`、`zh`、`zh_cn`、`chinese` |
| 英文译文（必填）| `英语`、`英文`、`译文`、`target`、`tgt`、`en`、`english` |
| 类型（可选）| `category`、`类型`、`分类` |
| 备注（可选）| `notes`、`note`、`备注`、`说明` |

> 若列名不匹配任何别名，退化为按列序取第 1 列为原文、第 2 列为译文。

**响应** `200 OK` → [TerminologyUploadResponse](#terminologyuploadresponse)

**错误**：
- `400` — 文件格式不支持、列数不足、内容为空

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/terminology/upload \
  -F "file=@术语表.xlsx"
```

```json
{
  "total": 1024,
  "added": 56,
  "updated": 12,
  "message": "术语表加载成功，previous_total=956, duplicates_skipped=3"
}
```

---

### GET /terminology — 查询术语表

分页查询当前内存中加载的所有术语。

**查询参数**：

| 参数 | 类型 | 默认 | 范围 | 描述 |
|------|------|------|------|------|
| `limit` | `integer` | `100` | `[1, 1000]` | 每页返回条数 |
| `offset` | `integer` | `0` | `≥ 0` | 偏移量 |

**响应** `200 OK` → [TerminologyListResponse](#terminologylistresponse)

**示例**：

```bash
# 查询前 50 条
curl "http://localhost:8000/api/v1/terminology?limit=50&offset=0"
```

```json
{
  "total": 1024,
  "items": [
    {
      "source": "燕云十六声",
      "target": "Yanyun: The Sixteen Sounds",
      "category": "游戏名",
      "notes": null
    }
  ]
}
```

---

## RAG — 双语语料库

RAG（Retrieval-Augmented Generation）模块使用 Qdrant 向量数据库存储双语语料，翻译时自动检索相似例句作为 LLM 参考。检索采用**混合检索**（Dense + BM25 + RRF 融合排序）策略，并按语料审校状态进行优先级加权排序。

### POST /rag/corpus/upload — 上传语料

上传双语语料并写入向量数据库。

**请求体** `multipart/form-data`：

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `file` | `file` | 是 | 双语语料文件，支持 `.xlsx` / `.xls` / `.csv` / `.json` |
| `collection` | `string \| null` | 否 | 目标 Qdrant collection 名，不填使用默认值；不存在时自动创建 |

**文件格式要求**：

| 列 | 识别的列名 |
|----|-----------|
| 中文原文（必填）| `中文`、`原文`、`source`、`src`、`zh` 等 |
| 英文译文（必填）| `英语`、`英文`、`译文`、`target`、`tgt`、`en` 等 |
| 上下文（可选）| `context`、`上下文`、`场景`、`scene`、`note`、`备注` |
| 审校状态（可选）| `status`、`状态`、`审校状态`、`review_status`、`级别` |

**审校状态优先级**（影响检索排序权重）：

| 状态值 | 权重 | 说明 |
|-------|------|------|
| `Designer Reviewed` | 1.00 | 最高质量，设计师审核 |
| `CQA_Done` | 0.97 | 文化 QA 完成 |
| `Done_LQA edited` | 0.94 | 语言 QA 编辑后完成 |
| `Done` | 0.90 | 已完成 |
| 其他 / 未填 | 0.85 | 默认 |

**JSON 格式示例**：

```json
[
  {
    "source": "侠之大者，为国为民。",
    "target": "The greatest of knights serve the nation and its people.",
    "context": "师父训诫场景",
    "status": "Designer Reviewed"
  }
]
```

**响应** `200 OK` → [CorpusUploadResponse](#corpusuploadresponse)

**错误**：
- `400` — 文件格式不支持、语料解析失败、内容为空

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/rag/corpus/upload \
  -F "file=@corpus.xlsx" \
  -F "collection=yanyun_v2"
```

```json
{
  "total": 5000,
  "indexed": 4998,
  "message": "语料入库成功，filename=corpus.xlsx"
}
```

---

### POST /rag/search — 手动检索测试

手动触发 RAG 检索，用于测试语料入库效果。

**请求体** `application/json`：

| 字段 | 类型 | 必填 | 默认 | 描述 |
|------|------|------|------|------|
| `query` | `string` | 是 | — | 检索原文（中文） |
| `threshold` | `float` | 否 | `0.85` | 相似度阈值，范围 `[0.0, 1.0]` |
| `top_k` | `integer` | 否 | `3` | 返回最相似前 K 条，范围 `[1, 10]` |
| `collection` | `string \| null` | 否 | `null` | 指定 Qdrant collection 名 |

**响应** `200 OK` → [RAGSearchResponse](#ragsearchresponse)

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "剑出鞘，天下乱",
    "threshold": 0.75,
    "top_k": 5
  }'
```

```json
{
  "query": "剑出鞘，天下乱",
  "total": 2,
  "results": [
    {
      "source": "剑出鞘，江湖乱。",
      "target": "The sword leaves its scabbard; the jianghu falls into turmoil.",
      "score": 0.92,
      "status": "Designer Reviewed"
    },
    {
      "source": "剑出，天下变。",
      "target": "The sword drawn, the world changes.",
      "score": 0.81,
      "status": "Done"
    }
  ]
}
```

---

## Style Guide — 风格指南

服务启动时会自动从 `data/style_guide/` 目录加载风格指南。

### POST /style-guide/upload — 上传风格指南

上传风格指南文本文件，替换当前内存中已加载的内容。

**请求体** `multipart/form-data`：

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `file` | `file` | 是 | 风格指南文件，支持 `.txt` / `.md` / `.markdown`，最大 **1 MB**，需 UTF-8 编码 |

**响应** `200 OK` → [StyleGuideUploadResponse](#styleguideuoloadresponse)

**错误**：
- `400` — 文件格式不支持、文件超过 1 MB、非 UTF-8 编码、内容为空

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/style-guide/upload \
  -F "file=@style_guide.md"
```

```json
{
  "filename": "style_guide.md",
  "char_count": 3842,
  "line_count": 127,
  "message": "风格指南加载成功 (loaded_at=2026-05-21T10:30:00)"
}
```

---

### GET /style-guide — 查询风格指南

查询当前已加载的风格指南信息。

**查询参数**：

| 参数 | 类型 | 默认 | 描述 |
|------|------|------|------|
| `full` | `boolean` | `false` | `true` 时在响应中附带完整规则正文；`false` 时只返回前 500 字预览 |

**响应** `200 OK` → [StyleGuideInfoResponse](#styleguideinforesposne)

**示例**：

```bash
# 仅查看预览
curl "http://localhost:8000/api/v1/style-guide"

# 获取完整规则
curl "http://localhost:8000/api/v1/style-guide?full=true"
```

```json
{
  "loaded": true,
  "filename": "style_guide.md",
  "char_count": 3842,
  "line_count": 127,
  "preview": "## 翻译风格要求\n\n1. 保持武侠 / 江湖 / 古风调性，避免现代口语...",
  "rules": null
}
```

---

## 数据模型

### BatchTranslateResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `task_id` | `string` | 任务唯一 ID（16 位 hex） |
| `total` | `integer` | 总句数 |
| `completed` | `integer` | 已完成句数 |
| `progress_pct` | `float` | 完成百分比（0.0 ~ 100.0） |
| `status` | `string` | `running` \| `completed` \| `resumed` |
| `results` | `TranslationResult[]` | 翻译结果列表 |

### TranslationResult

| 字段 | 类型 | 描述 |
|------|------|------|
| `source` | `string` | 原文 |
| `translation` | `string` | 译文 |
| `status` | `string` | `success` \| `error` |
| `content_type` | `string \| null` | AI 预分类的文本类型（如 `dialog`、`narrative`、`skill`） |
| `terminology_used` | `object[]` | 命中的术语列表，每项含 `source`、`target` 等字段 |
| `rag_references` | `object[] \| null` | RAG 检索到的参考例句（见 RAGSearchResult） |
| `error_msg` | `string \| null` | 翻译失败时的错误信息 |

### RAGSearchResult

| 字段 | 类型 | 描述 |
|------|------|------|
| `source` | `string` | 参考原文 |
| `target` | `string` | 参考译文 |
| `score` | `float` | 相似度分数 |
| `status` | `string \| null` | 该语料的审校状态 |

### RAGSearchResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `query` | `string` | 检索原文 |
| `total` | `integer` | 命中总数 |
| `results` | `RAGSearchResult[]` | 检索结果列表 |

### CorpusUploadResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `total` | `integer` | 本次上传总条数 |
| `indexed` | `integer` | 成功写入向量库的条数 |
| `message` | `string` | 状态说明 |

### TerminologyUploadResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `total` | `integer` | 当前内存中术语总数 |
| `added` | `integer` | 本次新增条数 |
| `updated` | `integer` | 本次更新条数 |
| `message` | `string` | 状态说明，含重复跳过数 |

### TerminologyListResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `total` | `integer` | 术语总数 |
| `items` | `TermEntry[]` | 当前页术语列表 |

### TermEntry

| 字段 | 类型 | 描述 |
|------|------|------|
| `source` | `string` | 中文原文 |
| `target` | `string` | 英文译文 |
| `category` | `string \| null` | 类型（角色名 / 地名 / 物品 / 技能 等） |
| `notes` | `string \| null` | 备注 |

### StyleGuideUploadResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `filename` | `string` | 原始文件名 |
| `char_count` | `integer` | 规则字符数（去除首尾空白后） |
| `line_count` | `integer` | 规则行数 |
| `message` | `string` | 状态说明 |

### StyleGuideInfoResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| `loaded` | `boolean` | 是否已加载 |
| `filename` | `string \| null` | 当前加载的文件名 |
| `char_count` | `integer` | 规则字符数 |
| `line_count` | `integer` | 规则行数 |
| `preview` | `string` | 规则前 500 字预览 |
| `rules` | `string \| null` | 完整规则正文（仅 `full=true` 时返回） |

---

## 文件格式说明

### 待翻译文本文件（`/translate/file`）

| 格式 | 说明 |
|------|------|
| `.txt` | 每行一句，UTF-8 编码，自动去除空行 |
| `.csv` / `.xlsx` / `.xls` | 需含原文列（`中文`/`原文`/`source` 等），可选 `content_type`/`文本类型` 列 |
| `.json` | 字符串数组 `["句子1", "句子2"]`，或对象数组 `[{"source": "...", "content_type": "..."}]` |

**对话模式额外列（`dialog_mode=true` 时需要）**：

| 列 | 识别名 | 说明 |
|----|--------|------|
| 对话 ID | `id`、`dialog_id`、`对话id`、`对话编号`、`dialogue_id` | 相同 ID 的行聚合为同一段对话 |
| 说话人 | `说话人`、`speaker`、`角色`、`name`、`character`、`actor` | 对话角色名 |
| 时刻 | `time`、`时间`、`时刻`、`timestamp`、`t` | 数值型，用于对话内排序 |

### 术语表文件（`/terminology/upload`）

支持 `.xlsx` / `.xls` / `.csv`，必须含中文原文和英文译文两列（列名识别规则见上方接口说明）。重复中文词条自动跳过（保留首次出现）。

### 双语语料文件（`/rag/corpus/upload`）

支持 `.xlsx` / `.xls` / `.csv` / `.json`。JSON 格式需为对象数组，每个对象至少含 `source`（原文）和 `target`（译文）字段。可选 `context`（场景说明）和 `status`（审校状态）字段。

### 风格指南文件（`/style-guide/upload`）

支持 `.txt` / `.md` / `.markdown`，UTF-8 编码，文件大小上限 **1 MB**。内容会完整传入 LLM System Prompt。

---

## 环境变量配置

通过项目根目录 `.env` 文件配置（参考 `.env.example`）：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 提供商标识 |
| `LLM_MODEL` | `gpt-4o` | LLM 模型名称 |
| `LLM_API_KEY` | — | LLM API Key（必填） |
| `LLM_BASE_URL` | `null` | 自定义 LLM 接口 Base URL（OpenAI 兼容接口）|
| `LLM_TEMPERATURE` | `0.3` | 生成温度 |
| `LLM_TIMEOUT` | `60.0` | 单次 LLM 调用超时（秒） |
| `LLM_MAX_RETRIES` | `5` | LLM 调用最大重试次数 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型名称 |
| `EMBEDDING_DIMENSIONS` | `null` | Matryoshka 截断维度，`null` 使用模型默认（当前为 1024）|
| `EMBEDDING_API_KEY` | — | Embedding API Key（必填） |
| `EMBEDDING_BASE_URL` | `null` | 自定义 Embedding 接口 Base URL |
| `QDRANT_HOST` | `localhost` | Qdrant 服务地址 |
| `QDRANT_PORT` | `6333` | Qdrant 服务端口 |
| `QDRANT_COLLECTION` | `yanyun_corpus` | 默认向量 collection 名称 |
| `RAG_THRESHOLD` | `0.5` | 全局 RAG 相似度阈值（Dense 路召回阶段） |
| `RAG_TOP_K` | `3` | 全局 RAG Top-K |
| `RAG_DENSE_PREFETCH` | `20` | Dense 路送入 RRF 的候选数 |
| `RAG_SPARSE_PREFETCH` | `20` | BM25 路送入 RRF 的候选数 |
| `BATCH_SIZE` | `50` | 全局默认批次大小 |
| `MAX_CONCURRENT` | `5` | 批量任务最大并发工作单元数 |
| `BATCH_SLEEP` | `0.5` | 批次间休眠时间（秒） |
| `CLUSTER_ENABLED` | `true` | 是否启用结构聚类整组翻译 |
| `CLUSTER_PAIR_THRESHOLD` | `0.4` | 句对前后缀相似度合并阈值（粗筛） |
| `CLUSTER_MIN_COVERAGE` | `0.5` | 公共前后缀覆盖率门槛（细筛） |
| `CLUSTER_MAX_GROUP_SIZE` | `10` | 单次整组 LLM 调用最大句数 |
| `DATA_DIR` | `./data` | 数据目录（术语表、风格指南自动加载路径） |
| `PROGRESS_DIR` | `./data/progress` | 批量任务断点续传进度文件目录 |
