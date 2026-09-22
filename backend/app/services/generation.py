"""
Generation service.

Builds the grounded RAG prompt from retrieved chunks and calls a local
Ollama LLM to produce the final answer.
"""
import logging

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful customer support assistant for an e-commerce store.
Answer the user's question using ONLY the information in the "Context" section below.
Rules:
- If the answer is not contained in the context, say clearly that you don't have
  that information and suggest the user contact human support. Do NOT make anything up.
- Keep answers concise and directly useful to a customer.
- When relevant, mention which policy/topic the answer comes from (e.g. "According to our
  returns policy...").
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] (source: {c['document']})\n{c['text']}")
    context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    return f"""Context:
{context}

Question: {question}

Answer the question based only on the context above, and mention which source
number(s) (e.g. [1], [2]) you used."""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call the local Ollama LLM with the retrieved context and return the answer."""
    prompt = build_prompt(question, chunks)

    if not chunks:
        return (
            "I couldn't find anything relevant to your question in our documentation. "
            "Please contact our support team for further help."
        )

    try:
        client = ollama.Client(host=settings.OLLAMA_HOST)
        response = client.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed")
        raise RuntimeError(f"Failed to generate an answer from the LLM: {exc}") from exc
