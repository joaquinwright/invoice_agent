"""
Load CORD receipts into a uniform shape the rest of the pipeline can use.

CORD's ground truth is nested and stringly-typed (line-item names live at
gt_parse.menu[i].nm, prices are strings like "75,000"). All of that mess is
handled here, so every other file sees only the clean Document below. Swap
datasets and this is the only file that changes.

Document is (document_id, image, ground_truth):
  * document_id  — stable id like "train-0", for logging and joins.
  * image        — the receipt photo (CORD is image-based; this is the input).
  * ground_truth — the correct answer mapped onto our field names, but with
                   values left as raw CORD strings. We don't clean them here;
                   normalizing "75,000" -> 75000.0 is the scoring step's job.

ground_truth is a plain dict of raw strings, not a validated Receipt — it's
reference data for the eval to normalize and compare, not model output.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)

# Where the Hugging Face download is cached. The cache itself is gitignored.
_CACHE_DIR = "data/hf_cache"
_DATASET = "naver-clova-ix/cord-v2"

# Our hand-labeled "messy" set: real photos we supply, labeled by a human.
_MESSY_DIR = "data/messy"


@dataclass
class Document:
    """One receipt in the uniform shape the pipeline consumes."""

    document_id: str
    image: Image.Image
    ground_truth: dict[str, Any]


def _flatten_ground_truth(gt_parse: dict[str, Any]) -> dict[str, Any]:
    """Map CORD's nested gt_parse onto our schema's field names.

    Values stay as raw CORD strings on purpose. Handles CORD's quirks: `menu` is
    usually a list but can be a single dict for one-item receipts, and optional
    sections may be missing.
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
        limit: stop after this many documents (None = all). Always pass a small
            limit during development so we never iterate the whole dataset by
            accident.
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


def load_messy(
    labels_path: str = f"{_MESSY_DIR}/labels.json",
    images_dir: str = f"{_MESSY_DIR}/images",
) -> Iterator[Document]:
    """Yield our hand-labeled messy receipts as Documents (same shape as load_cord).

    labels.json maps document_id -> {image, verified, total, subtotal, tax,
    line_items}. Only entries with "verified": true are yielded. The labels start
    life as a model draft (see eval/draft_labels.py); a human must check every
    field against the photo and flip verified to true. Skipping unverified entries
    enforces that rule in code — otherwise we'd be grading the model against another
    model's unchecked guesses, which would make the score meaningless.
    """
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    for document_id, entry in labels.items():
        if not entry.get("verified", False):
            logger.warning("Skipping %s: label not marked verified", document_id)
            continue
        image = Image.open(os.path.join(images_dir, entry["image"])).convert("RGB")
        yield Document(
            document_id=document_id,
            image=image,
            ground_truth={
                "line_items": entry.get("line_items", []),
                "total": entry.get("total"),
                "subtotal": entry.get("subtotal"),
                "tax": entry.get("tax"),
            },
        )


if __name__ == "__main__":
    # Demo: print one receipt's image info beside its flattened ground truth.
    # CORD is image-based, so the "input" shown is the image size, not text.
    for doc in load_cord(split="train", limit=1):
        print(f"document_id : {doc.document_id}")
        print(f"image       : {doc.image.size[0]}x{doc.image.size[1]} px, mode={doc.image.mode}")
        print("ground_truth (flattened, raw CORD strings):")
        print(json.dumps(doc.ground_truth, indent=2, ensure_ascii=False))
