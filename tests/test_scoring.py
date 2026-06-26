"""
Unit tests for eval/scoring.py.

Scoring is deterministic: given a prediction and a ground truth, the score is fixed.
That makes it unit-testable in the classic sense (unlike the LLM step). These tests
pin down the fiddly bits — comma-stripping, None handling, and multiset matching of
line items — because a subtle scoring bug silently corrupts every accuracy number.

Run with:  uv run pytest
"""

from eval.scoring import (
    normalize_number,
    normalize_quantity,
    numbers_match,
    score_document,
    score_line_items,
)
from src.schema import LineItem, Receipt


# ---- normalization --------------------------------------------------------

def test_normalize_number_strips_thousands_commas():
    assert normalize_number("1,591,600") == 1591600.0
    assert normalize_number("75,000") == 75000.0
    assert normalize_number("40,000.") == 40000.0


def test_normalize_number_passes_through_numbers():
    assert normalize_number(75000) == 75000.0
    assert normalize_number(8.5) == 8.5


def test_normalize_number_handles_missing_and_junk():
    assert normalize_number(None) is None
    assert normalize_number("") is None
    assert normalize_number("n/a") is None


def test_normalize_quantity_extracts_integer():
    assert normalize_quantity("1 x") == 1
    assert normalize_quantity("4") == 4
    assert normalize_quantity(3) == 3
    assert normalize_quantity(None) is None


# ---- numeric matching (with None as a real value) -------------------------

def test_numbers_match_after_normalizing():
    # our clean float vs CORD's comma string for the same amount
    assert numbers_match(1591600.0, "1,591,600") is True


def test_numbers_match_within_tolerance():
    assert numbers_match(8.50, 8.504) is True   # within abs_tol=0.01
    assert numbers_match(8.50, 8.60) is False


def test_both_none_match_but_one_none_does_not():
    # train-2 has no tax: predicting None is correct; predicting a number is wrong.
    assert numbers_match(None, None) is True
    assert numbers_match(0.0, None) is False
    assert numbers_match(None, "52,815") is False


# ---- line-item scoring ----------------------------------------------------

def items(*triples):
    """Build a list of LineItem from (name, price, quantity) triples."""
    return [LineItem(name=n, price=p, quantity=q) for n, p, q in triples]


def truth(*triples):
    """Build CORD-style ground-truth items (raw strings) from triples."""
    return [{"name": n, "price": p, "quantity": q} for n, p, q in triples]


def test_identical_line_items_score_perfect():
    pred = items(("Latte", 5.0, 1), ("Muffin", 3.5, 2))
    gt = truth(("Latte", "5,000", 1), ("Muffin", "3,500", 2))
    # NB prices differ here on purpose (5.0 vs 5,000), so this is NOT a match:
    result = score_line_items(pred, gt)
    assert result["f1"] == 0.0


def test_truly_identical_line_items_score_perfect():
    pred = items(("Latte", 5000.0, 1), ("Muffin", 3500.0, 2))
    gt = truth(("latte", "5,000", "1"), ("MUFFIN", "3,500", "2 x"))  # case/format differ, values same
    result = score_line_items(pred, gt)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "n_pred": 2, "n_truth": 2}


def test_two_empty_lists_score_perfect():
    result = score_line_items([], [])
    assert result["f1"] == 1.0


def test_missed_and_spurious_items_lower_precision_and_recall():
    # The real train-1 PEPPER case: model renamed and mispriced one item.
    pred = items(("PEPPER AUS WELL DONE", 145000.0, 1))
    gt = truth(("PEPPER AUS", "165,000", "1"))
    result = score_line_items(pred, gt)
    assert result["tp"] == 0          # no item matched
    assert result["precision"] == 0.0  # our one item was wrong
    assert result["recall"] == 0.0     # we found none of the real items


def test_partial_overlap_counts_only_matches():
    pred = items(("Latte", 5000.0, 1), ("Tea", 3000.0, 1))    # Tea is spurious
    gt = truth(("Latte", "5,000", "1"), ("Cake", "8,000", "1"))  # Cake is missed
    result = score_line_items(pred, gt)
    assert result["tp"] == 1
    assert result["precision"] == 0.5  # 1 of 2 predicted was real
    assert result["recall"] == 0.5     # found 1 of 2 real items


# ---- whole-document scoring ----------------------------------------------

def test_score_document_combines_fields():
    receipt = Receipt(
        line_items=items(("Latte", 5000.0, 1)),
        total=5000.0,
        subtotal=5000.0,
        tax=None,
    )
    gt = {
        "total": "5,000",
        "subtotal": "5,000",
        "tax": None,
        "line_items": truth(("Latte", "5,000", "1")),
    }
    result = score_document(receipt, gt)
    assert result["total"] is True
    assert result["subtotal"] is True
    assert result["tax"] is True          # both None
    assert result["line_items"]["f1"] == 1.0
