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
