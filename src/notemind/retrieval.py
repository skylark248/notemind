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
