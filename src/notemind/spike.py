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
