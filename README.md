# CUAD Contract Clause Extraction & Summarization Pipeline

An LLM-powered pipeline that analyzes legal contracts from [CUAD](https://www.atticusprojectai.org/cuad),
extracts termination / confidentiality / liability clauses, and generates a
concise 100–150 word summary of each contract.

## Features

- Extracts Termination, Confidentiality, and Liability clauses
- Generates concise 100–150 word contract summaries
- Supports both PDF and TXT contracts
- Handles long contracts using paragraph-aware chunking
- Uses Groq-hosted Llama 3.3 70B for fast inference
- Performs semantic search using local Sentence Transformer embeddings
- Exports results in CSV and JSON formats

## Setup

```bash
git clone <this-repo-url>
cd cuad-clause-extraction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

The pipeline uses the Groq API with `llama-3.3-70b-versatile` by default.
The model can be changed using the `GROQ_MODEL` environment variable.

> **Note:** Processing large batches may exceed the Groq free-tier token
> quota. If the daily quota is reached, the pipeline records the error for
> that contract and continues processing the remaining contracts.

## Get the data

Download CUAD v1 from the [Atticus Project](https://www.atticusprojectai.org/cuad)
and point `--input_dir` at the extracted `full_contract_txt/` or
`full_contract_pdf/` folder (`.txt` is preferred when both exist).

No download handy? A demo contract ships at
`sample_data/contracts/DEMO_SoftwareLicenseAgreement.txt`:

```bash
python run_pipeline.py --input_dir sample_data/contracts --limit 1
```

## Run

```bash
python run_pipeline.py \
    --input_dir /path/to/CUAD_v1/full_contract_txt \
    --output_dir output \
    --limit 50
```

The pipeline generates:

- `output/clause_extraction_results.csv`
- `output/clause_extraction_results.json`

Each record contains:

- `contract_id`
- `summary`
- `termination_clause`
- `confidentiality_clause`
- `liability_clause`

| Flag | Purpose |
|---|---|
| `--no_few_shot` | Run clause extraction zero-shot, for A/B comparison |
| `--limit 0` | Process every contract found |
| `--verbose` | Debug logging |

### Sample Output

Real output from a run against CUAD contracts:

```json
{
  "contract_id": "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT",
  "num_chars": 26260,
  "summary": "The agreement is between the Provider and the Recipient, which includes TELCOSTAR PTE, LTD and Ability Computer & Software Industries Ltd. The purpose of the agreement is for the Provider to provide certain services and resources to the Recipient. The Provider must provide services in good faith and in accordance with applicable law, while the Recipient is obligated to pay fees. The agreement can be terminated by either party with prior written notice, and upon termination, the Provider must return all Recipient-owned property and confidential information...",
  "termination_clause": "Either party may terminate without cause with 90 days' written notice; either party may terminate for material breach if uncured after 30 days' notice; termination also possible upon insolvency or bankruptcy of the other party.",
  "confidentiality_clause": "Parties must maintain each other's Confidential Information in confidence, using at least reasonable care, and only disclose for the Permitted Purpose or as required by law; obligation survives termination.",
  "liability_clause": "Provider shall indemnify Recipient against losses arising from Provider's negligence, willful misconduct, or breach of this Agreement."
}
```

When a clause type genuinely isn't present in a contract, the pipeline
returns `""` rather than hallucinating a plausible-sounding clause — verified
on a real run against a joint venture agreement with no confidentiality or
liability terms.

### Tests

```bash
pytest tests/
```
Covers text normalization, chunking, and JSON-parsing robustness — no API key required.

### Bonus: semantic search

```bash
python search_clauses.py --contract sample_data/contracts/DEMO_SoftwareLicenseAgreement.txt --query "short termination notice period"
```

Embeds the contract locally via `sentence-transformers` (no extra API key)
and returns the top-k passages closest to the query, even without lexical
overlap with the source wording. Performs semantic search over the
selected contract.

## Approach

### Data Loading
Loads `.txt` directly, falls back to `pdfplumber`/`PyPDF2` for PDFs;
normalizes whitespace and unicode; chunks long contracts on paragraph
boundaries with overlap so clauses spanning a chunk edge aren't lost.

### Clause Extraction
A structured JSON prompt with few-shot examples (toggle off via
`--no_few_shot`), run per-chunk and merged by keeping the longest
non-empty hit per clause type. Responses are parsed defensively to survive
stray markdown fences or malformed JSON without failing the batch.

### Summarization
Single-pass for short contracts; map-reduce (per-chunk summary → final
combine) for long ones, keeping every summary within the 100–150 word
budget regardless of contract length.

### Semantic Search (Bonus)
Local embeddings + cosine similarity, no hosted embedding API required.

The pipeline processes each contract through the following stages:

```
CUAD Contracts
      │
      ▼
Load & Preprocess
      │
      ▼
Chunk Long Documents
      │
      ▼
Groq LLM
 ├── Clause Extraction
 └── Contract Summarization
      │
      ▼
CSV / JSON Results
      │
      ▼
Semantic Search (Bonus)
```

## Design decisions

- **Chunking over truncation**: contracts routinely exceed a comfortable
  single-call context window; paragraph-aware chunking with overlap scales
  to arbitrary contract length instead of losing late clauses.
- **Groq backend**: `LLMClient` wraps Groq behind one `.complete()`
  interface — fast inference keeps a 50-contract batch quick even with
  per-chunk calls, and the interface stays swappable for another provider
  later.
- **Temperature 0** for extraction (deterministic, low hallucination) vs. a
  small temperature for summarization (more natural prose).
- **Local embeddings** for the search bonus, avoiding a second API key.
- **Per-contract error isolation**: if a single contract fails (e.g. an API
  rate limit or malformed response), the pipeline records the error against
  that contract and continues the batch rather than aborting the whole run.

## Tech Stack

- Python
- Groq API (Llama 3.3 70B)
- Sentence Transformers
- pdfplumber
- PyPDF2
- pandas
- scikit-learn

## Repository layout

```
src/
├── data_loader.py
├── clause_extraction.py
├── summarization.py
├── semantic_search.py
├── llm_client.py
└── pipeline.py

run_pipeline.py
search_clauses.py
config.py
prompts/
sample_data/
tests/
output/           # git-ignored
```

## Known Limitations

- Chunk-boundary clause duplication may occur for extremely long contracts, although overlap-aware chunking reduces this effect.
- The project does not include quantitative evaluation against CUAD annotations.
- Groq's free-tier token quota may not be sufficient to process all 50
  contracts in a single run. If the quota is exceeded, the pipeline records
  the API error for that contract and continues processing the remaining
  contracts. Running the pipeline with a higher quota or across multiple
  days resolves this limitation.

## Future Improvements

- Evaluate extraction accuracy using CUAD ground-truth annotations.
- Extend extraction to additional legal clause categories.
- Support semantic search across the complete processed contract collection.
- Add rate-limit-aware retry (sleep until reset) and a resume/checkpoint mode so large batches can complete across multiple days without reprocessing already-completed contracts.