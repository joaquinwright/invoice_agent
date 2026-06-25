"""
Load CORD receipts into a uniform shape the rest of the pipeline can consume.

WHY THIS FILE EXISTS (CLAUDE.md §2.6, §2.9): CORD's ground truth is nested,
CORD-specific, and stringly-typed — e.g. line-item names live at
`gt_parse.menu[i].nm`, the total at `gt_parse.total.total_price`, and prices are
strings like "75,000" (comma = thousands separator). That mess must be
quarantined HERE. Every other file (extract, validate, store, eval) should see
only the clean uniform `Document` below and never touch a raw CORD field. If we
ever swap datasets, this is the only file that changes.

The uniform shape is `Document(document_id, image, ground_truth)`:
  * document_id  — a stable string id, e.g. "train-0", for logging/eval joins.
  * image        — the receipt photo (a PIL image); this is the model's INPUT,
                   because CORD-v2 is image-based (there is no plain-text field).
  * ground_truth — the correct answer, flattened to OUR schema's field names
                   (line_items / total / subtotal / tax) but with values kept as
                   the RAW CORD strings. We deliberately do NOT clean them here:
                   normalizing "75,000" -> 75000.0 is a scoring decision that
                   belongs to the eval (Phase 6), not the loader.

NOTE: ground_truth here is a plain dict of raw strings, NOT a validated Receipt.
Ground truth is reference data, not model output — it is the eval's job to
normalize and compare it, not Pydantic's to validate it.
"""

import json
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import load_dataset
from PIL import Image

# Where the Hugging Face download is cached (see Phase 2.1). Committed data dir,
# but the cache itself is gitignored.
_CACHE_DIR = "data/hf_cache"
_DATASET = "naver-clova-ix/cord-v2"


@dataclass
class Document:
    """One receipt in the uniform shape the pipeline consumes."""

    document_id: str
    image: Image.Image
    ground_truth: dict[str, Any]


def _flatten_ground_truth(gt_parse: dict[str, Any]) -> dict[str, Any]:
    """Map CORD's nested `gt_parse` onto OUR schema's field names.

    Values are kept as raw CORD strings on purpose (see module docstring).
    Defensive about CORD's quirks: `menu` is usually a list but can be a single
    dict when a receipt has exactly one item, and optional sections may be
    missing entirely.
    """
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):  # CORD collapses a single-item menu into a dict
        menu = [menu]

    line_items = [
        {
            "name": item.get("nm"),
            "price": item.get("price"),
            "quantity": item.get("cnt"),  # e.g. "1 x"; may be None
        }
        for item in menu
        if isinstance(item, dict)
    ]

    total = gt_parse.get("total", {})
    sub = gt_parse.get("sub_total", {})

    return {
        "line_items": line_items,
        "total": total.get("total_price") if isinstance(total, dict) else None,
        "subtotal": sub.get("subtotal_price") if isinstance(sub, dict) else None,
        "tax": sub.get("tax_price") if isinstance(sub, dict) else None,
    }


def load_cord(split: str = "train", limit: int | None = None) -> Iterator[Document]:
    """Yield CORD receipts as uniform Documents.

    Args:
        split: "train", "validation", or "test".
        limit: stop after this many documents (None = all). During development
            we always pass a small limit so we never iterate the whole dataset
            (and, later, never spend tokens on it) by accident — CLAUDE.md §2.3.
    """
    dataset = load_dataset(_DATASET, split=split, cache_dir=_CACHE_DIR)

    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        parsed = json.loads(example["ground_truth"])
        yield Document(
            document_id=f"{split}-{i}",
            image=example["image"],
            ground_truth=_flatten_ground_truth(parsed["gt_parse"]),
        )


if __name__ == "__main__":
    # Demo: print one receipt's image info beside its flattened ground truth.
    # CORD-v2 is image-based, so the "input" we show is the image's dimensions,
    # not text (see module docstring).
    for doc in load_cord(split="train", limit=1):
        print(f"document_id : {doc.document_id}")
        print(f"image       : {doc.image.size[0]}x{doc.image.size[1]} px, mode={doc.image.mode}")
        print("ground_truth (flattened, raw CORD strings):")
        print(json.dumps(doc.ground_truth, indent=2, ensure_ascii=False))
