# aipe-mvp

<!-- bilingual-readme:start -->

## 双语说明 / Bilingual Documentation

> 本节提供整篇 README 的中英双语维护说明；下方保留原始详细说明、命令、路径和配置示例。
> This section provides bilingual maintenance notes for the full README; the original detailed notes, commands, paths, and configuration examples are preserved below.

### 中文

**概览**：《燕云十六声》中译英游戏本地化 AI 翻译服务，基于 FastAPI、Qdrant、术语库、风格指南和 RAG 召回。

**主要能力**：
- 提供 API 化翻译服务。
- 用向量检索召回翻译记忆和参考语料。
- 结合术语、风格指南和 Web 搜索兜底。

**使用方式**：按下方部署说明配置服务、向量库和项目资源后启动 FastAPI。

**状态**：该仓库仍按当前 README 的说明维护或使用。

**注意事项**：项目事实以 README 下方的服务结构和配置说明为准。

### English

**Overview**: ZH-to-EN game-localization translation service for Where Winds Meet, built with FastAPI, Qdrant, terminology, style guides, and RAG retrieval.

**Key capabilities**:
- Provides an API-based translation service.
- Uses vector retrieval for translation memory and reference corpora.
- Combines terminology, style guides, and web-search fallback.

**Usage**: Configure the service, vector store, and project resources as described below, then start FastAPI.

**Status**: This repository is maintained or used according to the current README notes.

**Notes**: Repository facts follow the service structure and configuration details below.

<!-- bilingual-readme:end -->

《燕云十六声》中译英游戏本地化 AI 翻译服务。FastAPI + Qdrant 向量检索 + LLM，带术语库、风格指南、RAG 召回和 Web 搜索兜底。

## 能力

- 翻译流水线：术语注入 → RAG 召回相似译例 → 风格指南约束 → LLM 生成
- 术语管理：精确/模糊匹配查询、批量导入
- 向量检索：Qdrant dense(text-embedding-3-large 1024 维) + sparse(jieba 词频) 混合召回
- 风格指南：检索匹配、按场景注入
- Web 搜索兜底：RAG 弱召回或术语未命中时调博查搜索补充上下文
- 视觉分析：从配图提取对汉译英有帮助的信息
- 批量处理：异步并发 + 断点续传

## 结构

```
aipe/
├── app/
│   ├── api/          # FastAPI 路由：translate / rag / terminology / style_guide
│   ├── services/     # 翻译流水线、LLM、RAG、术语、风格、Web 搜索、视觉
│   ├── schemas/      # Pydantic 模型
│   ├── utils/        # 文件解析、聚类、进度、文本处理
│   ├── config.py     # pydantic-settings 配置（全部走环境变量）
│   └── main.py
├── scripts/          # 语料入库、批量翻译
├── tests/            # pytest
├── Dockerfile
└── docker-compose.yml
scripts/              # 语料清洗 / Qdrant 入库 / 文件检查
```

## 运行

需 Docker + Docker Compose。

```bash
cp aipe/.env.example aipe/.env   # 填入 LLM / Embedding API key、Qdrant 地址
cd aipe
docker compose up -d
```

启动后访问 `http://localhost:8000/docs` 看交互式 API 文档；接口说明见 `aipe/API.md`，更多见 `aipe/QUICKSTART.md`。

## 数据与密钥

仓库**只含代码**。以下不在仓库内，需自备：

- 语料、术语库、风格指南、向量快照（`*.snapshot` / `corpus/` / `aipe/data/`）—— 体积大且属客户机密
- `.env` 里的 LLM / Embedding / 博查 API key —— 照 `aipe/.env.example` 自行填写