"""
The shape of an extracted receipt, declared once with Pydantic.

Pydantic parses and validates data against these classes, so "what a valid
receipt looks like" lives here instead of in hand-written checks.

Fields are limited to what the CORD dataset actually labels, since we can only
score fields that have a ground-truth answer. Prices are floats even though CORD
stores messy strings like "75,000"; turning that into clean numbers is part of
the job. Line-item name/price are required and quantity is optional; total is
required, subtotal and tax are optional.
"""

from pydantic import BaseModel


class LineItem(BaseModel):
    """One row on a receipt: a thing bought, how many, and its price."""

    name: str
    price: float
    quantity: int | None = None  # CORD's "cnt"; not always present


class Receipt(BaseModel):
    """A whole receipt: the items bought, optional subtotal/tax, and the total."""

    line_items: list[LineItem]
    total: float
    subtotal: float | None = None
    tax: float | None = None
