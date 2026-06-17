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
