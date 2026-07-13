"""
Pipeline orchestration: load contracts -> extract clauses -> summarize ->
write CSV/JSON output.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from src.clause_extraction import extract_clauses
from src.data_loader import load_contracts
from src.llm_client import LLMClient
from src.summarization import summarize_contract

logger = logging.getLogger(__name__)


def process_contract(client: LLMClient, contract, use_few_shot: bool = True) -> dict:
    """Run clause extraction + summarization for a single loaded contract."""
    clauses = extract_clauses(client, contract.text, use_few_shot=use_few_shot)
    summary = summarize_contract(client, contract.text)
    return {
        "contract_id": contract.contract_id,
        "source_path": contract.source_path,
        "num_chars": contract.num_chars,
        "summary": summary,
        "termination_clause": clauses["termination_clause"],
        "confidentiality_clause": clauses["confidentiality_clause"],
        "liability_clause": clauses["liability_clause"],
    }


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    limit: int | None = 50,
    provider: str | None = None,
    use_few_shot: bool = True,
) -> pd.DataFrame:
    """
    Full end-to-end run: load up to `limit` contracts from `input_dir`,
    extract clauses + summaries, and write results to `output_dir` as both
    CSV and JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient(provider=provider)

    contracts = list(load_contracts(input_dir, limit=limit))
    logger.info("Loaded %d contracts from %s", len(contracts), input_dir)

    results = []
    for contract in tqdm(contracts, desc="Processing contracts"):
        try:
            results.append(process_contract(client, contract, use_few_shot=use_few_shot))
        except Exception as e:
            logger.error("Failed to process %s: %s", contract.contract_id, e)
            results.append({
                "contract_id": contract.contract_id,
                "source_path": contract.source_path,
                "num_chars": contract.num_chars,
                "summary": "",
                "termination_clause": "",
                "confidentiality_clause": "",
                "liability_clause": "",
                "error": str(e),
            })

    df = pd.DataFrame(results)

    csv_path = output_dir / "clause_extraction_results.csv"
    json_path = output_dir / "clause_extraction_results.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2))

    logger.info("Wrote results to %s and %s", csv_path, json_path)
    return df
