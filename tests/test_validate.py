"""
Unit tests for src/validate.py (the parsing half, Phase 4.1).

Parsing is deterministic (CLAUDE.md §2.10): a given raw string either cleans+parses
into a Receipt or raises. No LLM, no network — pure logic, classic unit tests.

Run with:  uv run pytest
"""

import pytest
from pydantic import ValidationError

from src.schema import Receipt
from src.validate import extract_and_validate, parse_receipt


def make_extractor(replies):
    """Return a zero-arg extractor that yields `replies` in order, one per call.

    This is the fake we inject in place of the real LLM call — it lets us drive the
    retry loop deterministically with no API, no tokens.
    """
    it = iter(replies)
    return lambda: next(it)


# A minimal well-formed JSON reply (as a string, the way the model returns it).
PLAIN_JSON = '{"total": 8.5, "line_items": [{"name": "Latte", "price": 5.0}]}'


def test_plain_json_parses():
    r = parse_receipt(PLAIN_JSON)
    assert isinstance(r, Receipt)
    assert r.total == 8.5
    assert r.line_items[0].name == "Latte"


def test_json_fenced_block_parses():
    # The exact shape we saw Claude return in Phase 3: ```json ... ```
    raw = f"```json\n{PLAIN_JSON}\n```"
    r = parse_receipt(raw)
    assert r.total == 8.5


def test_bare_fence_without_language_parses():
    raw = f"```\n{PLAIN_JSON}\n```"
    r = parse_receipt(raw)
    assert r.total == 8.5


def test_surrounding_whitespace_is_tolerated():
    raw = f"\n\n  ```json\n{PLAIN_JSON}\n```  \n"
    r = parse_receipt(raw)
    assert r.total == 8.5


def test_truncated_json_raises():
    # A reply cut off mid-object is not well-formed.
    with pytest.raises(ValidationError):
        parse_receipt('{"total": 8.5, "line_items": [')


def test_prose_only_raises():
    # The model ignored instructions and just chatted.
    with pytest.raises(ValidationError):
        parse_receipt("Sorry, I can't read this receipt.")


def test_wrong_type_raises():
    # Well-formed JSON, but `total` is non-numeric text — fails schema validation,
    # which is exactly the well-formedness guard we want.
    with pytest.raises(ValidationError):
        parse_receipt('{"total": "lots", "line_items": []}')


# ---- the retry loop (Phase 4.2) -------------------------------------------

def test_succeeds_on_first_try():
    calls = []
    extractor = lambda: (calls.append(1), PLAIN_JSON)[1]
    r = extract_and_validate(extractor)
    assert isinstance(r, Receipt)
    assert len(calls) == 1  # no retries needed


def test_retries_then_succeeds():
    # First reply is garbage, second is valid — the loop should recover.
    extractor = make_extractor(["not json at all", PLAIN_JSON])
    r = extract_and_validate(extractor, document_id="train-7")
    assert isinstance(r, Receipt)


def test_gives_up_and_returns_none_after_exhausting_retries(caplog):
    # Always-bad output: 1 initial try + 2 retries = 3 attempts, then None (not a crash).
    bad = ["nope", "still nope", "nope again", "this one should never be reached"]
    extractor = make_extractor(bad)
    result = extract_and_validate(extractor, document_id="train-9", max_retries=2)
    assert result is None
    # It logged the give-up at ERROR level rather than raising.
    assert any("Giving up on train-9" in rec.message for rec in caplog.records)


def test_stops_calling_extractor_once_valid():
    # The fake would raise StopIteration if called a 3rd time; a clean parse on the
    # 2nd call must stop the loop before that happens.
    extractor = make_extractor(["```json\nbroken", PLAIN_JSON])
    r = extract_and_validate(extractor)
    assert isinstance(r, Receipt)
