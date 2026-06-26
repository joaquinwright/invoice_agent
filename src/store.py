"""
Persist extraction results to SQLite — the filing cabinet.

Once a receipt is extracted and validated, the result goes here so it can be read
back later (e.g. by the eval) without re-calling the LLM. SQLite is a SQL database
in a single local file — no server, no credentials — and Python's sqlite3 is built
in, so there's nothing to install.

One table, keyed by document_id:

    extractions
      document_id  TEXT PRIMARY KEY  -- "train-7"; one row per document
      status       TEXT             -- 'ok' or 'failed'
      total        REAL             -- NULL when failed
      subtotal     REAL
      tax          REAL
      line_items   TEXT             -- the line_items list as a JSON string; NULL when failed
      created_at   TEXT             -- ISO timestamp

A receipt's line_items is a list, but a SQL cell holds one value, so we store it
as a JSON string in one column. The relational alternative is a second table joined
by foreign key; we never query individual items here, so the JSON column is simpler.

The status column lets one table record both successes and failures, so a failed
document is on the books rather than silently missing.
"""

import json
import sqlite3
from datetime import datetime, timezone

from src.schema import Receipt

# Default on-disk location. *.db is gitignored. Tests pass ":memory:" instead.
DEFAULT_DB_PATH = "data/receipts.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS extractions (
    document_id TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    total       REAL,
    subtotal    REAL,
    tax         REAL,
    line_items  TEXT,
    created_at  TEXT NOT NULL
)
"""


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (or create) the database file and ensure the table exists.

    CREATE TABLE IF NOT EXISTS makes this safe to call every run. Setting
    row_factory = sqlite3.Row lets us read columns by name (row["total"]).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def _now() -> str:
    """Current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def save_receipt(conn: sqlite3.Connection, document_id: str, receipt: Receipt) -> None:
    """Write one extracted receipt as a row with status='ok'.

    INSERT OR REPLACE means re-running on a document already in the table overwrites
    its row instead of erroring on a duplicate key, so a run is safely repeatable.

    The ? placeholders are SQL parameters: values are substituted safely, so a value
    can never be executed as SQL. Never build SQL by concatenating values.
    """
    line_items_json = json.dumps([item.model_dump() for item in receipt.line_items])
    conn.execute(
        """
        INSERT OR REPLACE INTO extractions
            (document_id, status, total, subtotal, tax, line_items, created_at)
        VALUES (?, 'ok', ?, ?, ?, ?, ?)
        """,
        (document_id, receipt.total, receipt.subtotal, receipt.tax, line_items_json, _now()),
    )
    conn.commit()


def save_failure(conn: sqlite3.Connection, document_id: str) -> None:
    """Record that a document was attempted but produced no valid receipt.

    The receipt columns stay NULL; only the status and id are kept, so the document
    is logged as a known failure rather than missing.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO extractions
            (document_id, status, total, subtotal, tax, line_items, created_at)
        VALUES (?, 'failed', NULL, NULL, NULL, NULL, ?)
        """,
        (document_id, _now()),
    )
    conn.commit()


def get_receipt(conn: sqlite3.Connection, document_id: str) -> Receipt | None:
    """Read one stored receipt back, rebuilt and re-validated as a Receipt.

    Returns None if the id isn't in the table or is recorded as a failure. A
    successful row is reconstructed through Pydantic, so a malformed receipt can
    never come back out — reading re-validates, just like writing did.
    """
    row = conn.execute(
        "SELECT status, total, subtotal, tax, line_items FROM extractions WHERE document_id = ?",
        (document_id,),
    ).fetchone()

    if row is None or row["status"] != "ok":
        return None

    return Receipt.model_validate(
        {
            "total": row["total"],
            "subtotal": row["subtotal"],
            "tax": row["tax"],
            "line_items": json.loads(row["line_items"]),
        }
    )


if __name__ == "__main__":
    # Demo: write a receipt and a failure to an in-memory DB, then read the receipt
    # back — proves the round-trip without touching the real file.
    conn = connect(":memory:")

    demo = Receipt.model_validate(
        {
            "total": 8.5,
            "subtotal": 8.5,
            "tax": 0.0,
            "line_items": [{"name": "Latte", "price": 5.0, "quantity": 1}],
        }
    )
    save_receipt(conn, "demo-0", demo)
    save_failure(conn, "demo-1")

    print("read back demo-0:", get_receipt(conn, "demo-0"))
    print("read back demo-1 (failure):", get_receipt(conn, "demo-1"))
    print("read back missing:", get_receipt(conn, "demo-2"))
