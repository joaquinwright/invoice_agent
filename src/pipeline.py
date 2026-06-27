"""
The deterministic spine: loader -> extract -> validate -> store.

This connects the four components end-to-end. Plain Python controls the flow; the
LLM is one small step inside it:

    load_cord
      -> extract_receipt        send each image to the LLM
        -> extract_and_validate validate the reply (+ retries)
          -> save_receipt       a valid receipt is filed in SQLite
          -> save_failure       a bad one is logged as a failure instead

HARD_CAP is a cost safety rail. This loop calls the LLM once per document, so every
document costs tokens. A run refuses to process more than HARD_CAP documents, so a
too-large `limit` can't bill you for the whole dataset. Keep limits small during
development.

A bad document never crashes the run — extract_and_validate returns None on failure,
which routes to save_failure. We don't catch broader errors like a network outage
here; if the API is down, failing loudly is fine.
"""

import logging

from src.extract import extract_receipt
from src.loader import load_cord
from src.store import DEFAULT_DB_PATH, connect, save_failure, save_receipt
from src.validate import extract_and_validate

logger = logging.getLogger(__name__)

# A run never processes more than this many documents, whatever `limit` says.
HARD_CAP = 25


def process_documents(documents, *, db_path: str = DEFAULT_DB_PATH, max_retries: int = 2) -> dict[str, int]:
    """Extract, validate, and store every document in `documents`.

    This is the source-agnostic core of the pipeline: it takes any iterable of
    Documents (CORD, or a hand-labeled messy set) and runs the same loop. Returns a
    summary dict: {"ok": n, "failed": m, "total": n + m}. Each outcome is persisted
    to SQLite so the eval can read it back without re-calling the LLM.
    """
    conn = connect(db_path)
    counts = {"ok": 0, "failed": 0}

    try:
        for doc in documents:
            # Bind doc.image as a default arg so the extractor uses this document's
            # image. (It's called immediately, so it's safe either way, but binding
            # makes the intent explicit.)
            receipt = extract_and_validate(
                lambda image=doc.image: extract_receipt(image),
                document_id=doc.document_id,
                max_retries=max_retries,
            )

            if receipt is None:
                save_failure(conn, doc.document_id)
                counts["failed"] += 1
                logger.info("FAILED  %s (logged to needs-review)", doc.document_id)
            else:
                save_receipt(conn, doc.document_id, receipt)
                counts["ok"] += 1
                logger.info("OK      %s (total=%s)", doc.document_id, receipt.total)
    finally:
        conn.close()

    counts["total"] = counts["ok"] + counts["failed"]
    return counts


def run_pipeline(
    *,
    limit: int = 5,
    split: str = "train",
    db_path: str = DEFAULT_DB_PATH,
    max_retries: int = 2,
) -> dict[str, int]:
    """Run the pipeline over `limit` CORD documents and store every result.

    Raises ValueError if `limit` exceeds HARD_CAP — the safety rail fires before any
    document is loaded or any token is spent.
    """
    if limit > HARD_CAP:
        raise ValueError(
            f"limit={limit} exceeds HARD_CAP={HARD_CAP}. Refusing to run — "
            "every document costs tokens. Raise HARD_CAP deliberately if you mean it."
        )

    return process_documents(
        load_cord(split=split, limit=limit), db_path=db_path, max_retries=max_retries
    )


if __name__ == "__main__":
    # Whoever runs the pipeline configures logging (the modules only emit). INFO
    # shows one line per document plus any retry warnings.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    summary = run_pipeline(limit=5)
    print(f"\nDone. {summary['ok']} ok, {summary['failed']} failed, {summary['total']} total.")
    print(f"Results stored in {DEFAULT_DB_PATH}.")
