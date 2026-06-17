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
