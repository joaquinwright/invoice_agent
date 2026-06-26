"""
Unit tests for src/store.py.

Storage is deterministic: a written value comes back exactly, a failure reads back
as None, a re-write overwrites. No LLM, no network, and no real file — each test uses
an in-memory SQLite DB (":memory:") that disappears when the connection closes, so
tests stay fast and leave no *.db files behind.

Run with:  uv run pytest
"""

import pytest

from src.schema import Receipt
from src.store import connect, get_receipt, save_failure, save_receipt


@pytest.fixture
def conn():
    """A fresh, empty in-memory database for each test."""
    connection = connect(":memory:")
    yield connection
    connection.close()


def make_receipt(total=8.5):
    return Receipt.model_validate(
        {
            "total": total,
            "subtotal": 8.5,
            "tax": 0.0,
            "line_items": [
                {"name": "Latte", "price": 5.0, "quantity": 1},
                {"name": "Muffin", "price": 3.5},  # quantity omitted -> None
            ],
        }
    )


def test_round_trip_preserves_the_receipt(conn):
    original = make_receipt()
    save_receipt(conn, "train-0", original)
    loaded = get_receipt(conn, "train-0")

    assert loaded == original  # Pydantic models compare by value
    # nested list survived the JSON serialize/deserialize trip intact
    assert loaded.line_items[0].name == "Latte"
    assert loaded.line_items[0].quantity == 1
    assert loaded.line_items[1].quantity is None


def test_missing_document_returns_none(conn):
    assert get_receipt(conn, "does-not-exist") is None


def test_failure_is_recorded_but_reads_back_as_none(conn):
    # A failure is on the books (status='failed') but has no receipt to return.
    save_failure(conn, "train-9")
    assert get_receipt(conn, "train-9") is None
    # ...and it really is stored, not just absent:
    row = conn.execute(
        "SELECT status FROM extractions WHERE document_id = ?", ("train-9",)
    ).fetchone()
    assert row["status"] == "failed"


def test_rewrite_overwrites_same_document_id(conn):
    # INSERT OR REPLACE: re-running on the same id updates the row, not crashes.
    save_receipt(conn, "train-0", make_receipt(total=8.5))
    save_receipt(conn, "train-0", make_receipt(total=99.0))

    loaded = get_receipt(conn, "train-0")
    assert loaded.total == 99.0
    # still exactly one row for that id
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM extractions WHERE document_id = ?", ("train-0",)
    ).fetchone()["n"]
    assert count == 1


def test_failure_can_overwrite_a_previous_success(conn):
    # A re-run where the doc now fails should flip its status and clear the data.
    save_receipt(conn, "train-0", make_receipt())
    save_failure(conn, "train-0")

    assert get_receipt(conn, "train-0") is None
    row = conn.execute(
        "SELECT status, total FROM extractions WHERE document_id = ?", ("train-0",)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["total"] is None


def test_sql_injection_in_a_value_is_harmless(conn):
    # An item named like a SQL attack must be stored as plain text, never run. If
    # parameterization were broken, this could drop the table.
    nasty = Receipt.model_validate(
        {
            "total": 1.0,
            "line_items": [{"name": "'); DROP TABLE extractions;--", "price": 1.0}],
        }
    )
    save_receipt(conn, "train-0", nasty)
    loaded = get_receipt(conn, "train-0")
    assert loaded.line_items[0].name == "'); DROP TABLE extractions;--"
