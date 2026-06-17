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
