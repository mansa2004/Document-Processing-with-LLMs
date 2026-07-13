"""
Part A - Clause Extraction
==========================

Uses an LLM to identify termination, confidentiality, and liability clauses
from contract text. Handles long contracts by chunking and merging results
across chunks (a clause type is "found" as soon as any chunk returns a
non-empty extraction. If multiple chunks contain the same clause type,
the longest extracted clause is retained as it is typically the most complete.)
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

import config
from prompts.few_shot_examples import FEW_SHOT_EXAMPLES
from src.data_loader import chunk_text
from src.llm_client import LLMClient
from src.schemas.clause_schema import ClauseExtraction

logger = logging.getLogger(__name__)

CLAUSE_TYPES = [
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
]

_SYSTEM_PROMPT = (
    config.PROMPTS_DIR / "clause_extraction_prompt.txt"
).read_text(encoding="utf-8")



# Few-shot prompt construction


def _build_few_shot_messages() -> str:
    """
    Convert few-shot examples into text that is prepended to the prompt.
    """

    blocks = []

    for example in FEW_SHOT_EXAMPLES:
        blocks.append(
            "Example contract excerpt:\n"
            f"{example['contract_excerpt']}\n\n"
            "Expected JSON output:\n"
            f"{json.dumps(example['expected_json'], indent=2)}"
        )

    return "\n\n---\n\n".join(blocks)


_FEW_SHOT_BLOCK = _build_few_shot_messages()



# JSON Parsing

def _parse_json_response(raw: str) -> dict[str, str]:
    """
    Parse the JSON returned by the LLM.

    Handles common formatting issues such as Markdown code fences
    and attempts a best-effort recovery before returning empty values.
    """

    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")

        if cleaned.lower().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1]

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])

            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON returned by LLM.")
        logger.debug("Raw LLM response:\n%s", raw)

        return {key: "" for key in CLAUSE_TYPES}



# Chunk-level extraction
def extract_clauses_from_chunk(
    client: LLMClient,
    chunk: str,
    use_few_shot: bool = True,
) -> dict[str, str]:
    """
    Extract clauses from a single contract chunk.
    """

    user_prompt = ""

    if use_few_shot:
        user_prompt += f"{_FEW_SHOT_BLOCK}\n\n---\n\n"

    user_prompt += f"CONTRACT TEXT:\n{chunk}"

    logger.info("Running clause extraction on contract chunk...")

    raw = client.complete(
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=800,
        temperature=0.0,
    )

    result = _parse_json_response(raw)

    try:
        validated = ClauseExtraction.model_validate(result)

    except ValidationError as e:
        logger.warning(
            "Invalid clause extraction response: %s",
            e,
        )

        validated = ClauseExtraction()

    logger.info("Clause extraction completed.")

    return validated.model_dump()


# Merge chunk results
def _merge_clause_results(
    results: list[dict[str, str]],
) -> dict[str, str]:
    """
    Merge clause extractions from multiple chunks.

    If multiple chunks return the same clause type,
    retain the longest extracted clause.
    """

    merged = {key: "" for key in CLAUSE_TYPES}

    for result in results:

        for clause_type in CLAUSE_TYPES:

            candidate = result.get(clause_type, "")

            if candidate and len(candidate) > len(merged[clause_type]):
                merged[clause_type] = candidate

    return merged


# Full contract extraction
def extract_clauses(
    client: LLMClient,
    contract_text: str,
    use_few_shot: bool = True,
) -> dict[str, str]:
    """
    Extract termination, confidentiality, and liability clauses
    from an entire contract.

    Long contracts are automatically split into overlapping chunks
    before being processed by the LLM.
    """

    chunks = chunk_text(
        contract_text,
        config.MAX_CHARS_PER_CHUNK,
        config.CHUNK_OVERLAP_CHARS,
    )

    logger.info(
        "Processing %d chunk(s) for clause extraction.",
        len(chunks),
    )

    per_chunk_results: list[dict[str, str]] = []

    for index, chunk in enumerate(chunks, start=1):

        logger.info(
            "Processing chunk %d/%d",
            index,
            len(chunks),
        )

        per_chunk_results.append(
            extract_clauses_from_chunk(
                client=client,
                chunk=chunk,
                use_few_shot=use_few_shot,
            )
        )

    logger.info("Merging extracted clauses from all chunks.")

    return _merge_clause_results(per_chunk_results)