# aipe-mvp

[English](README.md) | 中文


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
