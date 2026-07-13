"""
Lightweight unit tests that don't require an LLM API key - they exercise
text normalization and chunking logic, plus JSON-parsing robustness for
clause extraction. Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import normalize_text, chunk_text, load_contract
from src.clause_extraction import _parse_json_response, CLAUSE_TYPES


def test_normalize_text_collapses_whitespace():
    raw = "Hello   world\n\n\n\nFoo"
    out = normalize_text(raw)
    assert "   " not in out
    assert "\n\n\n" not in out


def test_normalize_text_rejoins_hyphenated_linebreaks():
    raw = "This is a confiden-\ntiality clause."
    out = normalize_text(raw)
    assert "confidentiality" in out


def test_normalize_text_strips_page_numbers():
    raw = "Some clause text.\nPage 3 of 10\nMore text."
    out = normalize_text(raw)
    assert "Page 3 of 10" not in out


def test_chunk_text_short_text_single_chunk():
    text = "short contract text"
    chunks = chunk_text(text, max_chars=1000, overlap=50)
    assert chunks == [text]


def test_chunk_text_long_text_multiple_chunks():
    paragraph = "Clause text. " * 200  # long enough to exceed max_chars
    text = "\n\n".join([paragraph] * 5)
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 + 100 for c in chunks)  # allow small overlap slack


def test_parse_json_response_handles_clean_json():
    raw = '{"termination_clause": "a", "confidentiality_clause": "b", "liability_clause": "c"}'
    result = _parse_json_response(raw)
    assert result == {"termination_clause": "a", "confidentiality_clause": "b", "liability_clause": "c"}


def test_parse_json_response_handles_markdown_fences():
    raw = '```json\n{"termination_clause": "a", "confidentiality_clause": "", "liability_clause": ""}\n```'
    result = _parse_json_response(raw)
    assert result["termination_clause"] == "a"


def test_parse_json_response_recovers_from_garbage_prefix():
    raw = 'Sure, here is the JSON:\n{"termination_clause": "x", "confidentiality_clause": "", "liability_clause": ""}'
    result = _parse_json_response(raw)
    assert result["termination_clause"] == "x"


def test_parse_json_response_falls_back_gracefully_on_invalid_json():
    result = _parse_json_response("not json at all")
    assert result == {k: "" for k in CLAUSE_TYPES}


def test_load_contract_txt_roundtrip(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("TERMINATION.\n\nPage 1 of 1\n\nThis is the body.")
    contract = load_contract(p)
    assert contract.contract_id == "sample"
    assert "Page 1 of 1" not in contract.text
    assert "TERMINATION" in contract.text
