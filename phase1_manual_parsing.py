"""
Phase 1.1 — THROWAWAY. Feel the pain of validating LLM output by hand.

The job: take a *string* that is supposed to be a receipt in JSON and confirm it
really has the shape we need (vendor:str, date:str, total:number, and a list of
line_items each with name:str and price:number) BEFORE we trust it.

We do it with json.loads + hand-written checks here ON PURPOSE, so that when we
switch to Pydantic in Phase 1.2 you can feel exactly how much tedious, fragile
code it deletes. Run with:  uv run python phase1_manual_parsing.py
"""

import json


def validate_receipt_by_hand(raw: str) -> dict:
    """Return the parsed receipt, or raise ValueError with a clear message."""

    # Step 1: is it even JSON? An LLM can return a missing brace or a stray
    # sentence like "Here is your receipt:" glued to the front.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}")

    # Step 2: is the top level even an object (dict)? It could be a list, or a
    # bare string, and then every check below would explode with a TypeError.
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")

    # Step 3: required fields present? Check each one by hand...
    for field in ("vendor", "date", "total", "line_items"):
        if field not in data:
            raise ValueError(f"missing required field: {field!r}")

    # Step 4: ...and now check each field's TYPE by hand, one at a time.
    if not isinstance(data["vendor"], str):
        raise ValueError("vendor must be a string")
    if not isinstance(data["date"], str):
        raise ValueError("date must be a string")
    # Careful: in Python, bool is a subclass of int, so isinstance(True, int) is
    # True. A real hand-rolled checker has to remember gotchas like this.
    if not isinstance(data["total"], (int, float)) or isinstance(data["total"], bool):
        raise ValueError("total must be a number")

    # Step 5: line_items is a list, and EACH item needs its own nested checks.
    if not isinstance(data["line_items"], list):
        raise ValueError("line_items must be a list")
    for i, item in enumerate(data["line_items"]):
        if not isinstance(item, dict):
            raise ValueError(f"line_items[{i}] must be an object")
        if "name" not in item or "price" not in item:
            raise ValueError(f"line_items[{i}] needs both 'name' and 'price'")
        if not isinstance(item["name"], str):
            raise ValueError(f"line_items[{i}].name must be a string")
        if not isinstance(item["price"], (int, float)) or isinstance(item["price"], bool):
            raise ValueError(f"line_items[{i}].price must be a number")

    return data


# A handful of inputs that mimic what an LLM might actually hand back.
EXAMPLES = {
    "1. good receipt": '''
        {"vendor": "Joe's Coffee", "date": "2026-06-25", "total": 8.50,
         "line_items": [{"name": "Latte", "price": 5.00},
                        {"name": "Muffin", "price": 3.50}]}
    ''',
    "2. chatty preamble (not pure JSON)":
        'Here is the receipt you asked for:\n{"vendor": "Joe\'s", "date": "2026-06-25", "total": 8.5, "line_items": []}',
    "3. missing the 'total' field":
        '{"vendor": "Joe\'s", "date": "2026-06-25", "line_items": []}',
    "4. total written as words":
        '{"vendor": "Joe\'s", "date": "2026-06-25", "total": "eight fifty", "line_items": []}',
    "5. a line item missing its price":
        '{"vendor": "Joe\'s", "date": "2026-06-25", "total": 8.5, "line_items": [{"name": "Latte"}]}',
}


if __name__ == "__main__":
    for label, raw in EXAMPLES.items():
        try:
            validate_receipt_by_hand(raw)
            print(f"{label}\n   -> OK, accepted\n")
        except ValueError as e:
            print(f"{label}\n   -> REJECTED: {e}\n")
