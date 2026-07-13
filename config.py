"""
Central configuration for the CUAD Document Processing Pipeline.

Loads configuration values from a local `.env` file (if present).

All API keys are stored only in `.env`, which should never be committed
to version control.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = ROOT_DIR / "sample_data" / "contracts"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
PROMPTS_DIR = ROOT_DIR / "prompts"

# ---------------------------------------------------------------------
# LLM Configuration (Groq)
# ---------------------------------------------------------------------

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "groq",
).lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

# ---------------------------------------------------------------------
# Embeddings (Bonus Semantic Search)
# ---------------------------------------------------------------------

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "sentence-transformers",
).lower()

# Local embedding model (used later for semantic search)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

# ---------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------

# Contracts often exceed an LLM's context window.
# They are therefore split into overlapping chunks.

MAX_CHARS_PER_CHUNK = 12000

CHUNK_OVERLAP_CHARS = 500

# ---------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------

REQUEST_MAX_RETRIES = int(
    os.getenv("REQUEST_MAX_RETRIES", 3)
)

REQUEST_TIMEOUT_S = int(
    os.getenv("REQUEST_TIMEOUT_S", 120)
)