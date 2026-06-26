# Quick Start

## 前置要求

- Docker + Docker Compose

---

## 1. 获取代码

将压缩包解压到目标目录：

```bash
unzip AIPE.zip -d AIPE
cd AIPE
```

---

## 2. 配置环境变量（如果缺少的话）

```bash
cp .env.example .env
```

编辑 `.env`，填入以下必填项：

```
LLM_API_KEY=...
LLM_BASE_URL=...
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
```

---

## 3. 恢复 Qdrant 向量数据

将快照文件（`yanyun_0512.snapshot`）放到项目根目录，然后：

```bash
# 启动 Qdrant
docker compose up -d qdrant

# 恢复快照（collection 名称以 .env 里的 QDRANT_COLLECTION 为准，默认 yanyun_corpus）
curl -X POST "http://localhost:6333/collections/yanyun_corpus/snapshots/recover" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@yanyun_0512.snapshot"
```

等待返回 `{"result":true}` 即表示恢复成功。

验证：
```bash
curl "http://localhost:6333/collections/yanyun_corpus" | python3 -m json.tool
```

---

## 4. 启动全部服务

```bash
docker compose up -d
```

验证：
```bash
curl http://localhost:8000/health
```

返回 `{"status":"ok"}` 即启动成功，API 文档见 http://localhost:8000/docs

---

## 迁移时额外需要复制的内容（如果没有找到的话）

| 内容 | 说明 |
|------|------|
| `.env` | API Key 等敏感配置，不在 git 里 |
| `data/` 目录 | 上传的语料文件、翻译进度记录 |
| `yanyun_0512.snapshot` | Qdrant 向量数据快照 |
