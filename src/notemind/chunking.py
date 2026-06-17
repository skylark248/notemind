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
