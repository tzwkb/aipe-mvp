# aipe-mvp

[中文](README.md) | English

## Overview

 ZH-to-EN game-localization translation service for Where Winds Meet, built with FastAPI, Qdrant, terminology, style guides, and RAG retrieval.

## Key Capabilities

- Provides an API-based translation service.
- Uses vector retrieval for translation memory and reference corpora.
- Combines terminology, style guides, and web-search fallback.

## Usage

 Configure the service, vector store, and project resources as described below, then start FastAPI.

## Status

 This repository is maintained or used according to the current README notes.

## Notes

 Repository facts follow the service structure and configuration details below.

## Command and Configuration Reference

The following code blocks are preserved from the primary README. Commands, paths, and configuration keys are not translated; adjust them for the actual environment.

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

```bash
cp aipe/.env.example aipe/.env   # 填入 LLM / Embedding API key、Qdrant 地址
cd aipe
docker compose up -d
```

## Detailed Technical Notes

The primary README keeps the original technical details, history notes, full commands, and file layout. This file maintains the English version of the core documentation; consult the primary README code blocks and paths when exact commands are needed.
