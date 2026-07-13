"""
Bonus - Semantic Search over Clauses
=====================================

Embeds extracted clauses (and/or full contract chunks) and supports
similarity search over them, e.g. "find contracts with unusually short
termination notice periods" or "find the strictest liability caps".

This is intentionally decoupled from the main pipeline: run
`build_clause_index()` on the pipeline's output JSON/CSV after extraction,
then use `search()` interactively or from a small CLI.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ClauseIndex:
    contract_ids: list[str]
    clause_types: list[str]
    texts: list[str]
    vectors: np.ndarray = field(repr=False)

    def search(self, client: LLMClient, query: str, top_k: int = 5) -> list[dict]:
        query_vec = np.array(client.embed([query]))
        sims = cosine_similarity(query_vec, self.vectors)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {
                "contract_id": self.contract_ids[i],
                "clause_type": self.clause_types[i],
                "text": self.texts[i],
                "score": float(sims[i]),
            }
            for i in top_idx
        ]

    def save(self, path: Path) -> None:
        np.savez(
            path,
            contract_ids=np.array(self.contract_ids, dtype=object),
            clause_types=np.array(self.clause_types, dtype=object),
            texts=np.array(self.texts, dtype=object),
            vectors=self.vectors,
        )

    @classmethod
    def load(cls, path: Path) -> "ClauseIndex":
        data = np.load(path, allow_pickle=True)
        return cls(
            contract_ids=list(data["contract_ids"]),
            clause_types=list(data["clause_types"]),
            texts=list(data["texts"]),
            vectors=data["vectors"],
        )


def build_clause_index(client: LLMClient, results_json_path: Path) -> ClauseIndex:
    """
    Build an embedding index from the pipeline's output JSON
    (list of {contract_id, termination_clause, confidentiality_clause,
    liability_clause, summary}).
    """
    records = json.loads(Path(results_json_path).read_text())

    contract_ids, clause_types, texts = [], [], []
    for rec in records:
        for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
            text = rec.get(clause_type, "")
            if text:
                contract_ids.append(rec["contract_id"])
                clause_types.append(clause_type)
                texts.append(text)

    if not texts:
        raise ValueError("No non-empty clauses found in results file; nothing to index.")

    logger.info("Embedding %d clauses for semantic search index...", len(texts))
    vectors = np.array(client.embed(texts))
    return ClauseIndex(contract_ids=contract_ids, clause_types=clause_types, texts=texts, vectors=vectors)
