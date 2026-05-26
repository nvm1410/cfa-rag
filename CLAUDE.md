# CFA RAG Project

## Overview
Local RAG system for CFA PDFs.

## Stack
- Python
- FastAPI
- ChromaDB (vector store)
- BGE-M3 embeddings
- BM25 + reranker hybrid retrieval
- DeepSeek LLM via OpenAI-compatible API

## Key files
- app/ingest.py → PDF ingestion pipeline
- app/rag.py → retrieval + generation
- app/main.py → API server

## Rules
- Always use BGE-M3 for embeddings
- Always use hybrid search (BM25 + vector)
- Always rerank with BGE reranker before LLM
- Context must include page metadata

## LLM
- DeepSeek API (OpenAI compatible)
- model: deepseek-chat