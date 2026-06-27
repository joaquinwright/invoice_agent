"""
The evaluation harness: run the pipeline over a dev set and report how good it is.

This is the centerpiece. It runs extraction over a set of documents, scores each
result against ground truth (eval/scoring.py), and prints two things:
  * a per-field accuracy table — what breaks (maybe total is perfect but
    line_items is mediocre), and
  * a failure dump — the specific documents it got wrong, so we can see why.

A single overall number would hide all of that, so we never collapse to one score.

Dev set vs. held-out set: we evaluate on a slice of CORD's `train` split (the DEV
set), which we look at freely. CORD's separate `test` split is the HELD-OUT set —
we deliberately do NOT run on it until the very end, so the final number isn't
contaminated by us having tuned to examples we already stared at.

Predictions come from SQLite: run_pipeline extracts and stores, then we read each
result back and join it to ground truth by document_id. Pass run_extraction=False
to skip the (paid) extraction and just re-score rows already in the database —
handy while tuning the scoring rules.
"""

import logging

from eval.scoring import score_document
from src.loader import load_cord
from src.pipeline import run_pipeline
from src.store import DEFAULT_DB_PATH, connect, get_receipt

# The held-out split we promise not to touch until the end. Naming it here makes
# the promise explicit; the dev runs below never use it.
HELD_OUT_SPLIT = "test"


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


def evaluate(*, split: str = "train", limit: int = 8, db_path: str = DEFAULT_DB_PATH,
             run_extraction: bool = True) -> dict:
    """Run (or reuse) extraction over `limit` docs, score them, and return results.

    Returns {"summary": <aggregate>, "failures": [...], "n_failed_extraction": int}.
    Printing is done separately by print_report so this stays testable-ish.
    """
    if split == HELD_OUT_SPLIT:
        raise ValueError(
            f"'{HELD_OUT_SPLIT}' is the held-out set — don't evaluate on it until the end."
        )

    if run_extraction:
        run_pipeline(split=split, limit=limit, db_path=db_path)

    conn = connect(db_path)
    results: list[dict] = []
    failures: list[dict] = []
    n_failed_extraction = 0

    try:
        for doc in load_cord(split=split, limit=limit):
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


def print_report(report: dict, *, split: str, limit: int) -> None:
    """Pretty-print the accuracy table and the failure dump."""
    summary = report["summary"]
    n = summary["n"]

    print(f"\n=== Per-field accuracy (dev set: {split}, n={n}) ===")
    if n == 0:
        print("No documents scored.")
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
    SPLIT, LIMIT = "train", 8
    report = evaluate(split=SPLIT, limit=LIMIT)
    print_report(report, split=SPLIT, limit=LIMIT)
