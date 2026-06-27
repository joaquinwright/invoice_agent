"""
The evaluation harness: run the pipeline over a set and report how good it is.

This is the centerpiece. It runs extraction over a set of documents, scores each
result against ground truth (eval/scoring.py), and prints two things:
  * a per-field accuracy table — what breaks (maybe total is perfect but
    line_items is mediocre), and
  * a failure dump — the specific documents it got wrong, so we can see why.

A single overall number would hide all of that, so we never collapse to one score.

It runs on two sets:
  * the DEV set — a slice of CORD's clean `train` split, which we look at freely.
  * the MESSY set — our own hand-labeled real receipts (eval/draft_labels.py +
    src.loader.load_messy). Comparing the two numbers is the headline result: the
    gap shows how much the clean-dataset score flatters real-world performance.

CORD's `test` split is the HELD-OUT set — we deliberately do NOT run on it until
the very end, so the final number isn't contaminated by tuning to examples we
already stared at.

Predictions come from SQLite: extraction stores results, then we read each one back
and join it to ground truth by document_id. Pass run_extraction=False to skip the
(paid) extraction and just re-score rows already in the database — handy while
tuning the scoring rules.
"""

import logging
import sys

from eval.scoring import score_document
from src.loader import load_cord, load_messy
from src.pipeline import HARD_CAP, process_documents
from src.store import DEFAULT_DB_PATH, connect, get_receipt

# CORD's held-out split, named so the promise not to touch it is explicit in code.
HELD_OUT_SPLIT = "test"

# The messy set gets its own database file so its rows never mix with CORD's.
MESSY_DB_PATH = "data/messy.db"


def aggregate(results: list[dict]) -> dict:
    """Combine per-document score dicts into summary accuracies (pure, no I/O).

    Scalar fields report a correct-count out of n; line_items reports the mean of
    its per-document precision / recall / F1.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    out = {"n": n}
    for field in ("total", "subtotal", "tax"):
        out[field] = sum(1 for r in results if r[field])
    out["line_items_precision"] = sum(r["line_items"]["precision"] for r in results) / n
    out["line_items_recall"] = sum(r["line_items"]["recall"] for r in results) / n
    out["line_items_f1"] = sum(r["line_items"]["f1"] for r in results) / n
    return out


def _score_stored(documents, db_path: str) -> dict:
    """Read each document's stored prediction and score it against ground truth."""
    conn = connect(db_path)
    results: list[dict] = []
    failures: list[dict] = []
    n_failed_extraction = 0

    try:
        for doc in documents:
            prediction = get_receipt(conn, doc.document_id)
            if prediction is None:
                n_failed_extraction += 1
                failures.append({"document_id": doc.document_id, "extraction_failed": True})
                continue

            result = score_document(prediction, doc.ground_truth)
            results.append(result)

            wrong_scalars = [f for f in ("total", "subtotal", "tax") if not result[f]]
            if wrong_scalars or result["line_items"]["f1"] < 1.0:
                failures.append(
                    {
                        "document_id": doc.document_id,
                        "wrong_scalars": wrong_scalars,
                        "prediction": prediction,
                        "ground_truth": doc.ground_truth,
                        "line_items": result["line_items"],
                    }
                )
    finally:
        conn.close()

    return {
        "summary": aggregate(results),
        "failures": failures,
        "n_failed_extraction": n_failed_extraction,
    }


def evaluate(*, documents_factory, db_path: str, run_extraction: bool = True) -> dict:
    """Extract (optionally) and score a set produced by `documents_factory`.

    documents_factory is a zero-arg callable returning a FRESH iterator of Documents
    each time — we iterate the set twice (once to extract, once to score), so it
    can't be a one-shot generator.
    """
    if run_extraction:
        process_documents(documents_factory(), db_path=db_path)
    return _score_stored(documents_factory(), db_path)


def evaluate_cord(*, split: str = "train", limit: int = 8, db_path: str = DEFAULT_DB_PATH,
                  run_extraction: bool = True) -> dict:
    """Evaluate the clean CORD dev set."""
    if split == HELD_OUT_SPLIT:
        raise ValueError(
            f"'{HELD_OUT_SPLIT}' is the held-out set — don't evaluate on it until the end."
        )
    if limit > HARD_CAP:
        raise ValueError(f"limit={limit} exceeds HARD_CAP={HARD_CAP}.")
    return evaluate(
        documents_factory=lambda: load_cord(split=split, limit=limit),
        db_path=db_path,
        run_extraction=run_extraction,
    )


def evaluate_messy(*, db_path: str = MESSY_DB_PATH, run_extraction: bool = True) -> dict:
    """Evaluate our hand-labeled messy set (only verified labels are scored)."""
    return evaluate(documents_factory=load_messy, db_path=db_path, run_extraction=run_extraction)


def print_report(report: dict, *, label: str) -> None:
    """Pretty-print the accuracy table and the failure dump."""
    summary = report["summary"]
    n = summary["n"]

    print(f"\n=== Per-field accuracy ({label}, n={n}) ===")
    if n == 0:
        print("No documents scored. (Did you add verified labels?)")
        return

    for field in ("total", "subtotal", "tax"):
        correct = summary[field]
        print(f"  {field:<10} {correct}/{n}  ({100 * correct / n:.0f}%)")
    print(
        f"  {'line_items':<10} mean F1 {summary['line_items_f1']:.2f}  "
        f"(P {summary['line_items_precision']:.2f} / R {summary['line_items_recall']:.2f})"
    )
    if report["n_failed_extraction"]:
        print(f"  (extraction failed on {report['n_failed_extraction']} document(s))")

    print(f"\n=== Failures ({len(report['failures'])}) ===")
    for fail in report["failures"]:
        doc_id = fail["document_id"]
        if fail.get("extraction_failed"):
            print(f"  {doc_id}: EXTRACTION FAILED (no valid receipt)")
            continue

        print(f"  {doc_id}:")
        pred, gt = fail["prediction"], fail["ground_truth"]
        for field in fail["wrong_scalars"]:
            print(f"    {field:<9} ✗  predicted={getattr(pred, field)!r}  truth={gt.get(field)!r}")
        li = fail["line_items"]
        if li["f1"] < 1.0:
            print(
                f"    line_items P {li['precision']:.2f} / R {li['recall']:.2f} / F1 {li['f1']:.2f}  "
                f"(matched {li['tp']} of {li['n_pred']} predicted, {li['n_truth']} truth)"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Usage: `python -m eval.run_eval` (dev set) or `python -m eval.run_eval messy`.
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"

    if which == "messy":
        print_report(evaluate_messy(), label="messy set")
    else:
        print_report(evaluate_cord(split="train", limit=8), label="dev set: train")
