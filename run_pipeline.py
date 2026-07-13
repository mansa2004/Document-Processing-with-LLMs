#!/usr/bin/env python3
"""
CLI entry point for the CUAD Contract Processing Pipeline.

This script performs the complete document processing workflow:

1. Load contracts from the CUAD dataset
2. Extract text from PDF files
3. Normalize contract text
4. Extract termination, confidentiality, and liability clauses using Gemini
5. Generate a contract summary
6. Save results as CSV and JSON

Examples
--------
Process the first 50 contracts:

    python run_pipeline.py --input_dir data/CUAD_v1/full_contract_pdf --limit 50

Process only 10 contracts:

    python run_pipeline.py --input_dir data/CUAD_v1/full_contract_pdf --limit 10

Run on the bundled demo contract:

    python run_pipeline.py --input_dir sample_data --limit 1

Enable verbose logging:

    python run_pipeline.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from src.pipeline import run_pipeline


def main() -> None:
    """Parse command-line arguments and execute the pipeline."""

    parser = argparse.ArgumentParser(
        description="LLM-powered contract clause extraction and summarization pipeline."
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=config.DEFAULT_INPUT_DIR,
        help="Directory containing CUAD contract PDF files (searched recursively).",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and JSON results will be saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of contracts to process (default: 50). Use 0 to process all contracts.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini"],
        help="Override the LLM provider from .env (default: Gemini).",
    )

    parser.add_argument(
        "--no_few_shot",
        action="store_true",
        help="Disable few-shot prompting and use zero-shot clause extraction.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    limit = None if args.limit == 0 else args.limit

    df = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=limit,
        provider=args.provider,
        use_few_shot=not args.no_few_shot,
    )

    print("\n" + "=" * 60)
    print(f"Successfully processed {len(df)} contract(s).")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

    print(
        df[
            [
                "contract_id",
                "summary",
            ]
        ].to_string(
            index=False,
            max_colwidth=60,
        )
    )


if __name__ == "__main__":
    sys.exit(main())