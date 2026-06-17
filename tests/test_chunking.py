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
