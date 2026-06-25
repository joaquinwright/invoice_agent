"""
The shape of an extracted receipt, declared once with Pydantic.

This replaces the hand-written json.loads + isinstance checking we did in
phase1_manual_parsing.py. Here "what a valid receipt looks like" lives in plain,
readable class definitions, and Pydantic does the parsing and validation.

FIELDS LOCKED IN PHASE 2 against the real CORD dataset (CLAUDE.md §2.1, §2.4).
What changed from the provisional Phase-1 guess, and why:

  * Dropped `vendor` and `date`. CORD's ground truth does NOT label them, so we
    could never score them in the eval (CLAUDE.md §2.8: we can only evaluate
    fields the ground truth contains). Keeping them would be a lie in the
    accuracy table.
  * Prices are `float` here even though CORD stores messy strings like
    "75,000" (comma = thousands separator) or "40,000.". Turning that mess into
    clean typed numbers is exactly the job our system adds. The raw strings stay
    in the ground truth; the eval's scoring step (Phase 6) normalizes them to
    numbers to compare against our output.

Field choices are grounded in how often each appears across 150 CORD examples:
name 376 / price 375 (required on a line item), cnt 346 (common -> optional),
total_price 146/150 (required), subtotal 110/150 and tax 65/150 (optional).
"""

from pydantic import BaseModel


class LineItem(BaseModel):
    """One row on a receipt: a thing bought, how many, and its price."""

    name: str
    price: float
    quantity: int | None = None  # CORD's `cnt` (e.g. "1 x"); not always present


class Receipt(BaseModel):
    """A whole receipt: the items bought, optional subtotal/tax, and the total."""

    line_items: list[LineItem]
    total: float
    subtotal: float | None = None
    tax: float | None = None
