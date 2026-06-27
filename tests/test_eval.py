"""
Unit tests for eval/run_eval.py's pure aggregation.

evaluate() and print_report() are I/O (LLM, SQLite, printing) and are verified by
running them. aggregate() is pure — given per-document score dicts it returns fixed
summary numbers — so it gets a classic unit test here.

Run with:  uv run pytest
"""

from eval.run_eval import aggregate


def doc_result(total, subtotal, tax, f1, precision=None, recall=None):
    """Build one score_document-shaped result dict."""
    return {
        "total": total,
        "subtotal": subtotal,
        "tax": tax,
        "line_items": {
            "f1": f1,
            "precision": precision if precision is not None else f1,
            "recall": recall if recall is not None else f1,
        },
    }


def test_empty_results():
    assert aggregate([]) == {"n": 0}


def test_counts_and_means():
    results = [
        doc_result(True, True, True, 1.0),
        doc_result(True, False, True, 0.5),
        doc_result(False, True, True, 0.0),
        doc_result(True, True, False, 1.0),
    ]
    summary = aggregate(results)
    assert summary["n"] == 4
    assert summary["total"] == 3       # 3 of 4 correct
    assert summary["subtotal"] == 3
    assert summary["tax"] == 3
    assert summary["line_items_f1"] == (1.0 + 0.5 + 0.0 + 1.0) / 4  # 0.625


def test_line_item_means_average_precision_and_recall():
    results = [
        doc_result(True, True, True, 0.8, precision=1.0, recall=0.6),
        doc_result(True, True, True, 0.4, precision=0.5, recall=0.3),
    ]
    summary = aggregate(results)
    assert summary["line_items_precision"] == 0.75
    assert summary["line_items_recall"] == (0.6 + 0.3) / 2
