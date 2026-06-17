# Week 1 Spike — "hello RAG" on one file

**Date:** 2026-06-17
**Phase:** notemind Phase 1, Week 1 (Foundations + end-to-end spike)
**Status:** Approved, ready for implementation plan

## Goal

One CLI command answers a question grounded in a single `.md`/`.txt` file, returns a
grounded answer with a citation, end-to-end. The spike teaches the full RAG loop built
raw — no framework, no web server, no database. Those arrive in later weeks per the
master plan (FastAPI/pgvector wk2, reranker wk4, React/streaming wk5, deploy wk6).

This is a learning vehicle: the value is understanding RAG internals well enough to
explain every line and its tradeoffs in an interview — not shipping the spike itself.

## Scope

**In scope (Week 1):**
- Read one `.md`/`.txt` file into text.
- Chunk the text (strategy chosen by the author — the learning).
- Embed chunks via OpenAI `text-embedding-3-small`.
- Hold `(chunk, vector)` pairs in memory (a Python list — no DB).
- Embed the question, retrieve top-k chunks by hand-rolled cosine similarity.
- Assemble a prompt that grounds the answer in retrieved chunks and asks for a citation.
- Generate the answer via OpenAI `gpt-4o-mini`.
- Print the grounded answer plus which chunk(s) it used.
- Basic "not found in your notes" path when top-k similarity is weak.

**Out of scope (later weeks — do not creep):**
- FastAPI / any web server (wk2).
- Postgres + pgvector (wk2 ingestion pipeline).
- Reranker, hybrid search, chunk/k tuning (wk4).
- Formal eval set ≥10 Q/A (wk3; we only seed `questions.md` now).
- Streaming, structured citation API contract, debug panel (wk5).
- Basic-auth cost gate, Docker, deploy (wk6).
- PDF ingestion (wk7–8 buffer).
- Multiple files, multi-user auth, polished frontend.

## Decisions (Week 1)

| Decision | Choice | Rationale |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | Cheap, standard RAG reference embedding, best learning docs. |
| Generation | OpenAI `gpt-4o-mini` | One provider, one key, pennies. |
| Vector store (spike only) | In-memory hand-rolled cosine (NumPy) | Matches "hand-write retrieval/similarity to learn internals." pgvector lands wk2. |

## Architecture

### Data flow
```
file → chunk_text() → embed(chunks) → [(chunk, vector)] in memory
question → embed(question) → retrieve() cosine top-k → build_prompt() → complete() → grounded answer + cited chunk
```

### Components & division of labor

Per the locked AI-usage rule (hand-write the RAG core, delegate true boilerplate, read
all generated code):

| File | Author | Responsibility |
|---|---|---|
| `src/notemind/config.py` | Claude (boilerplate) | Load `.env`, expose settings (API key, model names, k). |
| `src/notemind/llm.py` | Claude (boilerplate) | `embed(texts) -> list[vector]`, `complete(prompt) -> str` OpenAI wrappers. |
| `src/notemind/chunking.py` | **Author (stub)** | `chunk_text(text, ...) -> list[Chunk]` — chunking strategy. |
| `src/notemind/retrieval.py` | **Author (stub)** | cosine similarity + `retrieve(query_vec, store, k) -> top_k`. |
| `src/notemind/prompt.py` | **Author (stub)** | `build_prompt(question, chunks) -> str` — context + citation assembly. |
| `src/notemind/spike.py` | Claude skeleton + **author TODOs** | CLI (argparse), reads file, orchestrates the stubs end-to-end. |

Stubs ship as real signatures + docstrings + `raise NotImplementedError` + a one-line
comment naming the tradeoff to consider. Nothing runs until the author fills them.

### Supporting files
- `notes/sample.md` — a sample corpus to test against.
- `questions.md` — running list of test questions (seeds the wk3 eval set; "habit from day 1").
- `.env.example` — `OPENAI_API_KEY=`.
- `requirements.txt` — `openai`, `numpy`, `python-dotenv`.

### Data shape
A `Chunk` carries at least: an id, the source filename, the text, and (after embedding)
its vector. Exact representation (dataclass vs dict) is an implementation-plan detail.

## Error handling (spike-level, minimal)
- Missing `OPENAI_API_KEY` → clear error message, exit.
- File not found / wrong extension → clear error message, exit.
- Empty or weak retrieval (top similarity below a threshold) → "not found in your notes"
  rather than a hallucinated answer.

## Testing
Week 1 is a spike; no formal test suite. Verification is running the CLI against
`notes/sample.md` with questions from `questions.md` and confirming answers are grounded
and cite the right chunk. The formal eval set (≥10 Q/A, retrieval hit-rate + grounding)
is a Week 3 deliverable.

## Deliverable
```
python -m notemind.spike ask "..." --file notes/sample.md
```
prints a grounded answer and the chunk(s) it used, with a graceful "not found" path.
