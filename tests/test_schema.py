"""
Unit tests for src/schema.py.

The schema is deterministic (CLAUDE.md §2.10): a given input either parses into a
valid Receipt or raises a clear error. That makes it perfect for classic unit
tests — unlike the LLM step, which we measure with the eval harness instead.

Run with:  uv run pytest
"""

import pytest
from pydantic import ValidationError

from src.schema import LineItem, Receipt


# A reusable, fully-valid receipt as a plain dict. Mirrors the LOCKED schema:
# line_items + total are required; quantity, subtotal, tax are optional.
VALID = {
    "total": 8.50,
    "subtotal": 8.50,
    "tax": 0.0,
    "line_items": [
        {"name": "Latte", "price": 5.00, "quantity": 1},
        {"name": "Muffin", "price": 3.50},  # quantity omitted -> allowed
    ],
}


# ---- the happy path -------------------------------------------------------

def test_valid_receipt_parses():
    r = Receipt.model_validate(VALID)
    assert r.total == 8.50
    assert len(r.line_items) == 2
    # nested objects become real typed LineItem instances, not bare dicts
    assert isinstance(r.line_items[0], LineItem)
    assert r.line_items[0].name == "Latte"
    assert r.line_items[0].quantity == 1


def test_optional_fields_default_to_none():
    # A minimal receipt: only the required fields. Optionals default to None.
    data = {"total": 5.0, "line_items": [{"name": "Latte", "price": 5.0}]}
    r = Receipt.model_validate(data)
    assert r.subtotal is None
    assert r.tax is None
    assert r.line_items[0].quantity is None


def test_empty_line_items_is_allowed():
    # A receipt with no line items is still well-formed (an empty list is fine).
    data = {**VALID, "line_items": []}
    r = Receipt.model_validate(data)
    assert r.line_items == []


def test_numeric_string_is_coerced_to_float():
    # Pydantic coerces a clean numeric string like "8.50" into a float. This is
    # convenient, but note it is still only VALIDATION (well-formedness), not a
    # claim that 8.50 is the CORRECT total — correctness is the eval's job.
    data = {**VALID, "total": "8.50"}
    r = Receipt.model_validate(data)
    assert r.total == 8.50
    assert isinstance(r.total, float)


# ---- the failure paths ----------------------------------------------------

def test_missing_required_field_raises():
    data = {k: v for k, v in VALID.items() if k != "total"}
    with pytest.raises(ValidationError) as exc:
        Receipt.model_validate(data)
    # the error names the offending field
    assert "total" in str(exc.value)


def test_non_numeric_total_raises():
    data = {**VALID, "total": "eight fifty"}
    with pytest.raises(ValidationError):
        Receipt.model_validate(data)


def test_nested_line_item_missing_price_raises():
    data = {**VALID, "line_items": [{"name": "Latte"}]}
    with pytest.raises(ValidationError) as exc:
        Receipt.model_validate(data)
    # the error path points into the nested structure
    err = exc.value.errors()[0]
    assert err["loc"] == ("line_items", 0, "price")


def test_line_items_must_be_a_list():
    data = {**VALID, "line_items": "Latte, Muffin"}
    with pytest.raises(ValidationError):
        Receipt.model_validate(data)


def test_malformed_json_string_raises():
    # model_validate_json takes a raw string (what an LLM hands back). A chatty
    # preamble before the JSON makes it unparseable.
    raw = 'Here is your receipt: {"total": 8.50}'
    with pytest.raises(ValidationError):
        Receipt.model_validate_json(raw)
