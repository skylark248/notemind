# notemind

An AI assistant over *your* notes. Ask questions and get answers **grounded in your own notes, with citations** — not hallucinated, not generic. Built RAG-first, from the internals up.

> **Status:** 🚧 Phase 1 in progress — notes ingestion + retrieval-augmented Q&A with citations.

---

## What it does

- **Ingest** your notes (markdown / txt) → chunk → embed → store.
- **Ask** a question → retrieve the relevant chunks → get a **grounded answer with citations** back to the source note.
- **Honest fallback** — if the answer isn't in your notes, it says so instead of making something up.
- **Streaming** answers and a **"show retrieved context"** view so you can see *why* it answered the way it did.

## Why

Most note tools either just store text or bolt on a generic chatbot that ignores what you actually wrote. notemind answers *only* from your corpus and shows its sources — so you can trust it and verify it.

---

## Architecture (Phase 1)

```
upload ──► chunk ──► embed ──► pgvector store
                                    │
question ──► embed ──► retrieve (top-k) ──► rerank ──► prompt assembly ──► LLM ──► cited answer
```

| Layer | Choice |
|---|---|
| Backend | FastAPI (async Python) |
| Vector store | Postgres + pgvector (hybrid keyword + vector search) |
| Embeddings | API embeddings to start |
| Reranker | BGE-reranker-v2 |
| LLM | one major API (Claude / OpenAI / Gemini) |
| RAG | thin / raw — no framework, built to understand the internals |
| Frontend | minimal React (streaming + citation rendering) |
| Deploy | Docker → live URL |

**Design choice:** RAG is built raw — chunking, retrieval, prompt assembly, and reranking are hand-written rather than delegated to a framework. The goal is to understand the levers, not to wire up a black box.

---

## Roadmap

- **Phase 1** — notes ingestion + RAG Q&A with citations + clean backend, deployed to a live URL. *(current)*
- **Phase 2** — agentic features + MCP tools + auto-generated quizzes/puzzles from your notes.
- **Phase 3** — evaluation harness + production hardening.

A small evaluation set (retrieval hit-rate + answer grounding) is built alongside Phase 1 to measure retrieval quality rather than guess at it.

---

## Getting started

> Setup instructions land as Phase 1 takes shape. Coming: local dev (Docker Compose for Postgres/pgvector), environment config, and ingest/query usage.

---

## License

TBD
