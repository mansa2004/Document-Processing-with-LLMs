"""
Part B - Contract Summarization
Generates a concise 100–150 word summary of a legal contract covering:

- Purpose of the agreement
- Key obligations of each party
- Notable risks or penalties

Long contracts are processed using a map-reduce strategy:

1. Split the contract into overlapping chunks.
2. Generate a brief summary for each chunk.
3. Merge the chunk summaries into a final contract summary.
"""

from __future__ import annotations

import logging

import config
from src.data_loader import chunk_text
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


# Prompt templates
_SUMMARY_SYSTEM_PROMPT = (
    config.PROMPTS_DIR / "summary_prompt.txt"
).read_text(encoding="utf-8")

_CHUNK_SUMMARY_SYSTEM_PROMPT = """
You are an expert legal contract analyst.

Summarize the following contract excerpt in 3–5 factual sentences.

Focus only on:
- Parties involved
- Purpose of the agreement
- Key obligations
- Important termination, liability, confidentiality, or penalty terms

Respond in plain text only.
Do not use bullet points.
Do not include headings.
"""


# Helper
def _generate_summary(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    """
    Generate a summary using the configured LLM.
    """

    return client.complete(
        system=system_prompt,
        user=user_prompt,
        max_tokens=max_tokens,
        temperature=0.2,
    ).strip()


def _summarize_single_pass(
    client: LLMClient,
    text: str,
) -> str:
    """
    Summarize a contract that fits within a single context window.
    """

    return _generate_summary(
        client=client,
        system_prompt=_SUMMARY_SYSTEM_PROMPT,
        user_prompt=f"CONTRACT TEXT:\n{text}",
        max_tokens=400,
    )

# Public API

def summarize_contract(
    client: LLMClient,
    contract_text: str,
) -> str:
    """
    Generate a 100–150 word summary of a contract.

    If the contract exceeds the configured chunk size,
    a map-reduce summarization strategy is used.
    """

    if not contract_text.strip():
        logger.warning("Received empty contract text for summarization.")
        return ""

    chunks = chunk_text(
        contract_text,
        config.MAX_CHARS_PER_CHUNK,
        config.CHUNK_OVERLAP_CHARS,
    )

    logger.info(
        "Contract split into %d chunk(s).",
        len(chunks),
    )

    # Single-pass summarization
    if len(chunks) == 1:

        logger.info("Generating single-pass contract summary.")

        summary = _summarize_single_pass(
            client,
            chunks[0],
        )

  
    # Map-Reduce summarization
    else:

        logger.info(
            "Using map-reduce summarization for long contract."
        )

        chunk_summaries: list[str] = []

        for index, chunk in enumerate(chunks, start=1):

            logger.info(
                "Summarizing chunk %d/%d",
                index,
                len(chunks),
            )

            chunk_summary = _generate_summary(
                client=client,
                system_prompt=_CHUNK_SUMMARY_SYSTEM_PROMPT,
                user_prompt=f"CONTRACT EXCERPT:\n{chunk}",
                max_tokens=250,
            )

            chunk_summaries.append(chunk_summary)

        logger.info("Generating final summary from chunk summaries.")

        combined = "\n\n".join(chunk_summaries)

        final_prompt = (
            "The following are summaries of consecutive excerpts "
            "from the SAME contract.\n\n"
            "Combine them into ONE coherent summary of "
            "100–150 words.\n\n"
            "Your summary must include:\n"
            "- Purpose of the agreement\n"
            "- Key obligations of each party\n"
            "- Notable risks or penalties\n\n"
            "Avoid repetition.\n"
            "Respond with plain text only.\n\n"
            f"EXCERPT SUMMARIES:\n{combined}"
        )

        summary = _generate_summary(
            client=client,
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            user_prompt=final_prompt,
            max_tokens=400,
        )

    # Validate summary length

    word_count = len(summary.split())

    logger.info(
        "Generated summary (%d words).",
        word_count,
    )

    if word_count < 100 or word_count > 150:

        logger.warning(
            "Summary length (%d words) is outside the required "
            "100–150 word range. Regenerating.",
            word_count,
        )

        rewrite_prompt = (
            "Rewrite the following summary so that it is between "
            "100 and 150 words.\n\n"
            "Keep all important information.\n"
            "Use plain text only.\n\n"
            f"{summary}"
        )

        summary = _generate_summary(
            client=client,
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            user_prompt=rewrite_prompt,
            max_tokens=400,
        )

        logger.info(
            "Regenerated summary (%d words).",
            len(summary.split()),
        )

    return summary