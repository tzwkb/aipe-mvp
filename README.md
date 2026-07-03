# aipe-mvp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-framework-009688.svg)](https://fastapi.tiangolo.com/)

English | [中文](README_ZH.md)

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

The following code blocks keep commands, paths, filenames, and configuration keys literal; explanatory comments are translated for the English README.

```
aipe/
├── app/
│   ├── api/          # FastAPI routes: translate / rag / terminology / style_guide
│   ├── services/     # translation pipeline, LLM, RAG, terminology, style guides, web search, vision
│   ├── schemas/      # Pydantic models
│   ├── utils/        # file parsing, clustering, progress tracking, text processing
│   ├── config.py     # pydantic-settings config (all via environment variables)
│   └── main.py
├── scripts/          # corpus ingestion and batch translation
├── tests/            # pytest
├── Dockerfile
└── docker-compose.yml
scripts/              # corpus cleaning / Qdrant ingestion / file checks
```

```bash
cp aipe/.env.example aipe/.env   # add LLM / embedding API key and Qdrant URL
cd aipe
docker compose up -d
```

## Detailed Technical Notes

This README keeps the English version of the core documentation. Code blocks, paths, commands, and file-layout examples are kept literal so they can be copied and checked against the repository.
