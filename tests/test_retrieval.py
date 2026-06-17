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
