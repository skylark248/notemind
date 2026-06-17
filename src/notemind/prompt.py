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
