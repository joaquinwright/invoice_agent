"""
Per-field scoring: compare an extracted Receipt against CORD's ground truth.

This is the only place where a prediction and its ground-truth answer meet. The
extraction step never sees the answer (it only gets the image), so these scores
are honest — if the answer ever leaked into the model's input, every number here
would be meaningless.

The work this module does is mostly normalization, because the two sides arrive in
different forms:
  * our prediction is a clean Receipt (price 75000.0, quantity 1)
  * CORD's ground truth is raw strings (price "75,000", quantity "1 x", and
    numbers use a comma as the thousands separator: "1,591,600")

So before comparing we strip commas, pull integers out of quantity strings, and
treat None as a real value (a receipt with no tax should match a prediction of
None, and mismatch a number).

Scoring rules per field:
  * total / subtotal / tax — numeric match within a small tolerance.
  * line_items — matched as multisets keyed by (name, price, quantity); reported
    as precision / recall / F1. Names are matched exactly after normalizing case
    and whitespace (no fuzzy matching), so a misspelling counts as a miss.
"""

import re
from collections import Counter
from typing import Any

from src.schema import LineItem, Receipt


def normalize_number(value: Any) -> float | None:
    """Turn a price/total into a float, or None if it's absent or unparseable.

    Accepts numbers as-is and strings like "1,591,600" or "40,000." (comma =
    thousands separator, dot = decimal point). Strips currency symbols and any
    other non-numeric characters.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    stripped = re.sub(r"[^0-9.\-]", "", str(value))  # drop commas, currency, spaces
    if stripped in ("", "-", "."):
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def normalize_quantity(value: Any) -> int | None:
    """Pull the integer out of a quantity, or None. "1 x" -> 1, "4" -> 4."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None


def normalize_name(value: Any) -> str:
    """Lowercase and collapse whitespace so names compare consistently."""
    if value is None:
        return ""
    return " ".join(str(value).split()).lower()


def numbers_match(a: Any, b: Any, *, abs_tol: float = 0.01) -> bool:
    """Compare two numbers after normalizing both sides.

    Two missing values (both None) match; one missing and one present do not.
    Present values match if they're within abs_tol of each other.
    """
    na, nb = normalize_number(a), normalize_number(b)
    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False
    return abs(na - nb) <= abs_tol


def _item_key(name: Any, price: Any, quantity: Any) -> tuple:
    """A canonical, comparable key for one line item (name, price, quantity)."""
    return (normalize_name(name), normalize_number(price), normalize_quantity(quantity))


def score_line_items(predicted: list[LineItem], truth: list[dict[str, Any]]) -> dict[str, float]:
    """Score predicted line items against truth as multisets, returning P/R/F1.

    Two items match only if name, price, AND quantity all agree. tp is the size of
    the multiset overlap; precision = tp / predicted count, recall = tp / truth
    count. Two empty lists score a perfect 1.0 (correctly finding nothing).
    """
    pred_keys = Counter(_item_key(i.name, i.price, i.quantity) for i in predicted)
    truth_keys = Counter(_item_key(i.get("name"), i.get("price"), i.get("quantity")) for i in truth)

    tp = sum((pred_keys & truth_keys).values())  # multiset intersection
    n_pred = sum(pred_keys.values())
    n_truth = sum(truth_keys.values())

    if n_pred == 0 and n_truth == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "n_pred": 0, "n_truth": 0}

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "n_pred": n_pred,
        "n_truth": n_truth,
    }


def score_document(receipt: Receipt, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Score one extracted receipt against its ground truth, field by field.

    Returns booleans for the scalar fields and a P/R/F1 dict for line_items.
    """
    return {
        "total": numbers_match(receipt.total, ground_truth.get("total")),
        "subtotal": numbers_match(receipt.subtotal, ground_truth.get("subtotal")),
        "tax": numbers_match(receipt.tax, ground_truth.get("tax")),
        "line_items": score_line_items(receipt.line_items, ground_truth.get("line_items", [])),
    }
