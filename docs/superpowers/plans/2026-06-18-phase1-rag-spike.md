# Phase 1 Week 1 — "hello RAG" Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI command answers a question grounded in one `.md`/`.txt` file, returning a grounded answer plus the chunk it cited, end-to-end, with no framework/web-server/DB.

**Architecture:** Read one file → chunk → embed (OpenAI) → hold `(chunk, vector)` in a Python list → embed the question → hand-rolled cosine top-k retrieval → assemble a grounding prompt → generate (OpenAI) → print answer + citation. RAG core (chunking, similarity, prompt assembly, orchestration) is hand-written by the author to learn internals; true boilerplate (config, API wrappers, data model, CLI scaffold) is generated.

**Tech Stack:** Python 3.11+, `openai`, `numpy`, `python-dotenv`. Models: `text-embedding-3-small` (embeddings), `gpt-4o-mini` (generation).

---

## Conventions for this plan

- **Author split.** Tasks tagged **[BOILERPLATE]** get full code. Tasks tagged **[HAND-WRITE]** ship a stub + failing contract tests + guidance — the human author writes the implementation to make the tests pass. Do NOT fill a [HAND-WRITE] body for them.
- **Commits are the human's.** Every commit step stages only (`git add`) and shows a suggested message after a `▶ COMMIT (you):` marker. Never run `git commit` or `git push`.
- **plan.md stays local** (gitignored). Never stage or commit it.

## File structure

| File | Author | Responsibility |
|---|---|---|
| `requirements.txt` | BOILERPLATE | Pin deps. |
| `.env.example` | BOILERPLATE | `OPENAI_API_KEY=` template. |
| `src/notemind/__init__.py` | BOILERPLATE | Package marker. |
| `src/notemind/models.py` | BOILERPLATE | `Chunk` dataclass — shared contract between chunking & retrieval. |
| `src/notemind/config.py` | BOILERPLATE | Load `.env`, expose settings (key, model names, top-k, threshold). |
| `src/notemind/llm.py` | BOILERPLATE | `embed(texts)` + `complete(prompt)` OpenAI wrappers. |
| `src/notemind/chunking.py` | HAND-WRITE | `chunk_text(text, source)` — chunking strategy. |
| `src/notemind/retrieval.py` | HAND-WRITE | `cosine_similarity` + `retrieve(query_vec, store, k)`. |
| `src/notemind/prompt.py` | HAND-WRITE | `build_prompt(question, chunks)` — context + citation assembly. |
| `src/notemind/spike.py` | scaffold = BOILERPLATE, `answer_question` = HAND-WRITE | CLI + file read (generated); RAG orchestration + not-found gate (author). |
| `notes/sample.md` | BOILERPLATE | Sample corpus. |
| `questions.md` | BOILERPLATE | Seed test-question list (habit from day 1). |
| `tests/test_models.py` | BOILERPLATE | Chunk contract. |
| `tests/test_chunking.py` | BOILERPLATE (tests) | Failing contract tests for the author to satisfy. |
| `tests/test_retrieval.py` | BOILERPLATE (tests) | Failing contract tests for the author to satisfy. |
| `tests/test_prompt.py` | BOILERPLATE (tests) | Failing contract tests for the author to satisfy. |

---

## Task 1: Project skeleton [BOILERPLATE]

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/notemind/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
openai>=1.40
numpy>=1.26
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 2: Create `.env.example`**

```
OPENAI_API_KEY=
```

- [ ] **Step 3: Create empty package markers**

`src/notemind/__init__.py` and `tests/__init__.py` — both empty files.

- [ ] **Step 4: Create venv and install**

Run:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Expected: installs without error. `pip show openai` prints a version ≥ 1.40.

- [ ] **Step 5: Verify `.env` is gitignored, then stage**

Run: `git check-ignore .env` → expected output: `.env` (already covered by existing `.gitignore`). Create your real `.env` from the example: `cp .env.example .env` and paste your key.

▶ COMMIT (you): `git add requirements.txt .env.example src/notemind/__init__.py tests/__init__.py && git commit -m "chore: project skeleton + deps"`

---

## Task 2: Chunk data model [BOILERPLATE]

**Files:**
- Create: `src/notemind/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_models.py
from notemind.models import Chunk


def test_chunk_holds_fields_and_defaults_vector_none():
    c = Chunk(id=0, source="a.md", text="hello")
    assert c.id == 0
    assert c.source == "a.md"
    assert c.text == "hello"
    assert c.vector is None


def test_chunk_accepts_vector():
    c = Chunk(id=1, source="a.md", text="x", vector=[0.1, 0.2])
    assert c.vector == [0.1, 0.2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'notemind.models'` (run from repo root with `.venv` active; add `src` to path via step 3's layout or `pip install -e .` — simplest: run pytest with `PYTHONPATH=src`).

- [ ] **Step 3: Implement `models.py`**

```python
# src/notemind/models.py
from dataclasses import dataclass


@dataclass
class Chunk:
    """One unit of retrievable text from a source note.

    `vector` is None until the chunk is embedded.
    """

    id: int
    source: str
    text: str
    vector: list[float] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_models.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add a pytest config so `src` is always on the path**

Create `pytest.ini`:
```ini
[pytest]
pythonpath = src
```
Re-run `python -m pytest tests/test_models.py -v` (no PYTHONPATH needed now). Expected: PASS.

▶ COMMIT (you): `git add src/notemind/models.py tests/test_models.py pytest.ini && git commit -m "feat: Chunk data model"`

---

## Task 3: Config loader [BOILERPLATE]

**Files:**
- Create: `src/notemind/config.py`

- [ ] **Step 1: Implement `config.py`**

```python
# src/notemind/config.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 3
    # Below this top cosine score, treat the corpus as not containing the answer.
    min_similarity: float = 0.25


def get_settings() -> Settings:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Settings(openai_api_key=key)
```

- [ ] **Step 2: Verify it loads**

Run: `PYTHONPATH=src python -c "from notemind.config import get_settings; print(get_settings().embedding_model)"`
Expected: prints `text-embedding-3-small` (with a real `.env`). With no key: prints the clear `RuntimeError` message.

▶ COMMIT (you): `git add src/notemind/config.py && git commit -m "feat: settings loader"`

*Note: `min_similarity = 0.25` is a starting guess for the not-found gate. You will tune it by hand in Task 9 against `questions.md`.*

---

## Task 4: OpenAI wrappers [BOILERPLATE]

**Files:**
- Create: `src/notemind/llm.py`

- [ ] **Step 1: Implement `llm.py`**

```python
# src/notemind/llm.py
from openai import OpenAI

from notemind.config import get_settings

_settings = get_settings()
_client = OpenAI(api_key=_settings.openai_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns one vector per input, in order."""
    resp = _client.embeddings.create(model=_settings.embedding_model, input=texts)
    return [item.embedding for item in resp.data]


def complete(prompt: str) -> str:
    """Single-turn completion. Returns the assistant's text."""
    resp = _client.chat.completions.create(
        model=_settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""
```

- [ ] **Step 2: Smoke-test against the real API**

Run:
```bash
PYTHONPATH=src python -c "from notemind.llm import embed; v=embed(['hello'])[0]; print(len(v))"
```
Expected: prints `1536` (the `text-embedding-3-small` dimension). This confirms key + network + wrapper.

*No committed test — avoids spending API calls in any future CI run.*

▶ COMMIT (you): `git add src/notemind/llm.py && git commit -m "feat: OpenAI embed + complete wrappers"`

---

## Task 5: Sample corpus + question seed [BOILERPLATE]

**Files:**
- Create: `notes/sample.md`
- Create: `questions.md`

- [ ] **Step 1: Create `notes/sample.md`**

```markdown
# Espresso basics

A standard espresso shot uses about 18 grams of finely ground coffee and yields
roughly 36 grams of liquid in 25 to 30 seconds. The water temperature should sit
between 90 and 96 degrees Celsius.

# Grind size

If the shot runs too fast and tastes sour, the grind is too coarse — go finer.
If it runs too slow and tastes bitter, the grind is too fine — go coarser.

# Milk

Steam milk to about 60 to 65 degrees Celsius. Hotter than 70 scalds it and kills
the sweetness.
```

- [ ] **Step 2: Create `questions.md`**

```markdown
# Test questions (seeds the wk3 eval set)

Format: question | expected grounded answer | source section

1. How many grams of coffee for a shot? | ~18 g | Espresso basics
2. What temperature to steam milk to? | 60–65 °C | Milk
3. Shot is sour and fast — what do I change? | grind finer | Grind size
4. What is the capital of France? | NOT in notes (should refuse) | —
```

▶ COMMIT (you): `git add notes/sample.md questions.md && git commit -m "chore: sample corpus + question seed"`

---

## Task 6: Chunking [HAND-WRITE]

**Files:**
- Create: `src/notemind/chunking.py` (stub generated; body is yours)
- Test: `tests/test_chunking.py` (generated, contract only)

- [ ] **Step 1: Generate the stub**

```python
# src/notemind/chunking.py
from notemind.models import Chunk


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split raw note text into retrievable Chunks.

    Returns Chunks with sequential ids (0, 1, 2, ...), the given source, and
    non-empty text. Vector stays None (embedding happens later).

    YOUR CALL (the learning): fixed-size vs paragraph/heading-aware? Overlap or
    not? Too-big chunks dilute retrieval; too-small lose context. Decide and be
    ready to defend it.
    """
    raise NotImplementedError
```

- [ ] **Step 2: Generate the failing contract tests**

```python
# tests/test_chunking.py
from notemind.chunking import chunk_text
from notemind.models import Chunk

SAMPLE = "Para one has some words.\n\nPara two has other words.\n\nPara three ends it."


def test_returns_nonempty_list_of_chunks():
    chunks = chunk_text(SAMPLE, source="sample.md")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_text_is_nonempty_and_stripped():
    chunks = chunk_text(SAMPLE, source="sample.md")
    assert all(c.text.strip() for c in chunks)


def test_ids_are_sequential_from_zero():
    chunks = chunk_text(SAMPLE, source="sample.md")
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_source_is_set_and_vector_unset():
    chunks = chunk_text(SAMPLE, source="sample.md")
    assert all(c.source == "sample.md" for c in chunks)
    assert all(c.vector is None for c in chunks)


def test_all_input_words_survive_chunking():
    # Strategy-agnostic: every word in the input appears in some chunk.
    chunks = chunk_text(SAMPLE, source="sample.md")
    joined = " ".join(c.text for c in chunks)
    for word in ["one", "two", "three", "words", "ends"]:
        assert word in joined
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunking.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 4: YOUR TURN — implement `chunk_text`**

Hand-write the body to make all five tests pass. Tests fix the *contract* (types, ids, non-empty, words survive) but not your *strategy* — paragraph-split, fixed-window, sliding-overlap are all fair. Log your choice + why in a code comment or `questions.md`; it's interview fodder. Do not change the tests to fit a shortcut.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunking.py -v`
Expected: PASS (5 passed).

▶ COMMIT (you): `git add src/notemind/chunking.py tests/test_chunking.py && git commit -m "feat: chunking strategy"`

---

## Task 7: Retrieval [HAND-WRITE]

**Files:**
- Create: `src/notemind/retrieval.py` (stub generated; body is yours)
- Test: `tests/test_retrieval.py` (generated, contract only)

- [ ] **Step 1: Generate the stub**

```python
# src/notemind/retrieval.py
from notemind.models import Chunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. Range -1.0 .. 1.0.

    YOUR CALL (the learning): the dot-product / norm formula. NumPy is fine, but
    understand each term — this is the heart of vector search.
    """
    raise NotImplementedError


def retrieve(
    query_vec: list[float], store: list[Chunk], k: int
) -> list[tuple[Chunk, float]]:
    """Return the k chunks most similar to query_vec, as (chunk, score) pairs
    sorted by score descending. Every chunk in `store` must have a vector set.
    If k exceeds len(store), return all of them.
    """
    raise NotImplementedError
```

- [ ] **Step 2: Generate the failing contract tests**

```python
# tests/test_retrieval.py
import math

from notemind.models import Chunk
from notemind.retrieval import cosine_similarity, retrieve


def test_cosine_identical_vectors_is_one():
    assert math.isclose(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, rel_tol=1e-6)


def test_cosine_orthogonal_vectors_is_zero():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_opposite_vectors_is_minus_one():
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, rel_tol=1e-6)


def _store():
    return [
        Chunk(id=0, source="s", text="near", vector=[1.0, 0.0]),
        Chunk(id=1, source="s", text="mid", vector=[0.7, 0.7]),
        Chunk(id=2, source="s", text="far", vector=[0.0, 1.0]),
    ]


def test_retrieve_returns_k_pairs_sorted_desc():
    results = retrieve([1.0, 0.0], _store(), k=2)
    assert len(results) == 2
    assert results[0][0].id == 0  # most similar to [1,0]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_caps_at_store_size():
    results = retrieve([1.0, 0.0], _store(), k=99)
    assert len(results) == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 4: YOUR TURN — implement `cosine_similarity` and `retrieve`**

Hand-write both bodies to pass all five tests. Understand why normalization matters and what happens with a zero vector (guard it). Brute-force linear scan is correct and expected at this scale — note in a comment that pgvector replaces this in wk2.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: PASS (5 passed).

▶ COMMIT (you): `git add src/notemind/retrieval.py tests/test_retrieval.py && git commit -m "feat: cosine retrieval"`

---

## Task 8: Prompt assembly [HAND-WRITE]

**Files:**
- Create: `src/notemind/prompt.py` (stub generated; body is yours)
- Test: `tests/test_prompt.py` (generated, contract only)

- [ ] **Step 1: Generate the stub**

```python
# src/notemind/prompt.py
from notemind.models import Chunk


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Assemble the grounding prompt sent to the LLM.

    The prompt must (1) include the question, (2) include each retrieved chunk's
    text labeled by its id, and (3) instruct the model to answer ONLY from the
    provided context, cite the chunk id(s) used, and say it cannot find the
    answer in the notes if the context does not contain it.

    YOUR CALL (the learning): instruction wording, how you label/delimit chunks,
    where the question goes. This is prompt + context engineering — the grounding
    quality lives here.
    """
    raise NotImplementedError
```

- [ ] **Step 2: Generate the failing contract tests**

```python
# tests/test_prompt.py
from notemind.models import Chunk
from notemind.prompt import build_prompt

CHUNKS = [
    Chunk(id=0, source="s.md", text="Use 18 grams of coffee."),
    Chunk(id=1, source="s.md", text="Steam milk to 60 degrees."),
]


def test_prompt_includes_question():
    out = build_prompt("how much coffee?", CHUNKS)
    assert "how much coffee?" in out


def test_prompt_includes_every_chunk_text():
    out = build_prompt("q", CHUNKS)
    assert "Use 18 grams of coffee." in out
    assert "Steam milk to 60 degrees." in out


def test_prompt_references_chunk_ids():
    out = build_prompt("q", CHUNKS)
    assert "0" in out and "1" in out


def test_prompt_instructs_grounding():
    out = build_prompt("q", CHUNKS).lower()
    # Some instruction to stay within the provided context must be present.
    assert "context" in out or "notes" in out
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 4: YOUR TURN — implement `build_prompt`**

Hand-write the body to pass all four tests. The tests check the context/question/ids are present and that a grounding instruction exists — they do NOT dictate your wording. Iterate the wording later against `questions.md` (especially the "capital of France" refusal case).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: PASS (4 passed).

▶ COMMIT (you): `git add src/notemind/prompt.py tests/test_prompt.py && git commit -m "feat: grounding prompt assembly"`

---

## Task 9: CLI + orchestration [scaffold BOILERPLATE, `answer_question` HAND-WRITE]

**Files:**
- Create: `src/notemind/spike.py`

- [ ] **Step 1: Generate the scaffold (CLI + file read + the orchestration stub)**

```python
# src/notemind/spike.py
import argparse
import sys
from pathlib import Path

from notemind.chunking import chunk_text
from notemind.config import get_settings
from notemind.llm import complete, embed
from notemind.prompt import build_prompt
from notemind.retrieval import retrieve
from notemind.models import Chunk

ALLOWED_SUFFIXES = {".md", ".txt"}


def load_file(path: Path) -> str:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        sys.exit(f"Only .md/.txt supported, got: {path.suffix}")
    return path.read_text(encoding="utf-8")


def answer_question(question: str, text: str, source: str) -> str:
    """Run the RAG loop and return the answer string.

    YOUR TURN — wire the pieces and own the control flow:
      1. chunk_text(text, source) -> chunks
      2. embed the chunk texts; assign each vector back onto its Chunk
      3. embed the question
      4. retrieve(query_vec, chunks, settings.top_k) -> ranked (chunk, score)
      5. NOT-FOUND GATE: if the top score < settings.min_similarity, return a
         clear "I couldn't find that in your notes." WITHOUT calling the LLM
      6. build_prompt(question, top_chunks) -> prompt
      7. complete(prompt) -> answer
      8. return the answer plus which chunk id(s)/source were used

    settings = get_settings() gives you top_k and min_similarity.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(prog="notemind.spike")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="Ask a question grounded in a file")
    ask.add_argument("question")
    ask.add_argument("--file", required=True, type=Path)
    args = parser.parse_args()

    if args.cmd == "ask":
        text = load_file(args.file)
        print(answer_question(args.question, text, source=args.file.name))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: YOUR TURN — implement `answer_question`**

Fill the body per the docstring's 8 steps. This is where the loop becomes real: you decide how to attach vectors to chunks, how to slice top-k for the prompt, the not-found gate, and the citation line. No new tests here — verification is the end-to-end run in Task 10.

- [ ] **Step 3: Verify the CLI wiring resolves (before the API call)**

Run: `PYTHONPATH=src python -m notemind.spike --help` and `python -m notemind.spike ask --help`
Expected: argparse help prints, no import errors.

▶ COMMIT (you): `git add src/notemind/spike.py && git commit -m "feat: CLI + RAG orchestration"`

---

## Task 10: End-to-end verification

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests PASS (models + chunking + retrieval + prompt).

- [ ] **Step 2: Run a grounded question**

Run: `PYTHONPATH=src python -m notemind.spike ask "How many grams of coffee for a shot?" --file notes/sample.md`
Expected: answer states ~18 g and cites the chunk/section from "Espresso basics".

- [ ] **Step 3: Run the refusal case**

Run: `PYTHONPATH=src python -m notemind.spike ask "What is the capital of France?" --file notes/sample.md`
Expected: a "couldn't find that in your notes" style answer — NOT "Paris". If it answers Paris, tune `min_similarity` in `config.py` and/or your prompt's grounding instruction, then re-run.

- [ ] **Step 4: Log results in `questions.md`**

Record actual answers vs expected for all four seed questions. This is your first retrieval/grounding baseline (formalized into the eval set in wk3).

- [ ] **Step 5: Deliverable check**

You have: a working `ask` CLI, grounded answers with citations, a refusal path, and a baseline log. Week 1 done.

▶ COMMIT (you): `git add questions.md && git commit -m "test: wk1 baseline results"`

---

## Self-review notes (author)

- **Spec coverage:** read one md/txt ✓ (Task 9 load_file) · chunk/embed/store ✓ (6, 4, 9) · cosine top-k retrieval ✓ (7) · grounded answer + citation ✓ (8, 9) · not-found path ✓ (9 gate, 10 step 3) · question seed ✓ (5) · in-memory store ✓ (9) · OpenAI small + gpt-4o-mini ✓ (3, 4). Out-of-scope items (FastAPI, pgvector, reranker, streaming, auth, deploy, PDF) correctly absent.
- **Learning boundary honored:** chunking, cosine/retrieval, prompt assembly, and orchestration are HAND-WRITE stubs; only config/wrappers/model/CLI-scaffold/data are generated.
- **Type consistency:** `Chunk(id, source, text, vector)` used identically across models/chunking/retrieval/prompt/spike. `retrieve` returns `list[tuple[Chunk, float]]`; `answer_question` consumes that shape.
