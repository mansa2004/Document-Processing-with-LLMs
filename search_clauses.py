#!/usr/bin/env python3
"""
Semantic Contract Search
========================

Search a contract using semantic similarity.

Example:
python search_clauses.py ^
    --contract sample_data/contracts/DEMO_SoftwareLicenseAgreement.txt ^
    --query "termination clause"

"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import extract_text_from_file
from src.llm_client import LLMClient
from src.semantic_search import search_contract


def main():

    parser = argparse.ArgumentParser(
        description="Semantic search over contracts."
    )

    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Path to the contract.",
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of results to return.",
    )

    args = parser.parse_args()

    client = LLMClient()

    contract_text = extract_text_from_file(args.contract)

    results = search_contract(
        client=client,
        contract_text=contract_text,
        query=args.query,
        top_k=args.top_k,
    )

    print("\n" + "=" * 80)
    print(f'Search Query: "{args.query}"')
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 80)
        print(f"Similarity : {result['score']:.3f}\n")
        print(result["text"][:1000])

        if len(result["text"]) > 1000:
            print("\n...")

        print("-" * 80)


if __name__ == "__main__":
    main()