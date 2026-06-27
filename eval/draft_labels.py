"""
Model-assisted labeling for the messy receipt set.

Scans data/messy/images/, runs extraction on each photo, and writes draft entries
to data/messy/labels.json with "verified": false. This is a convenience, not the
labeling itself.

IMPORTANT: these drafts are the MODEL's guesses, not ground truth. Open labels.json,
check every field against the actual receipt, fix what's wrong, then set
"verified": true. The eval skips any entry that isn't verified, because grading the
model against its own unchecked guesses would measure nothing. You are the authority
on what's correct — the draft just saves typing.

Re-running is safe: entries already in labels.json are left untouched (so your
corrections survive), and only new, unlabeled images are drafted. document_id is the
image filename without its extension, so ids stay stable across runs.

Run with:  uv run python -m eval.draft_labels
"""

import json
import logging
import os

from src.extract import extract_receipt
from src.loader import _MESSY_DIR
from src.validate import parse_receipt
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGES_DIR = f"{_MESSY_DIR}/images"
_LABELS_PATH = f"{_MESSY_DIR}/labels.json"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic")


def _draft_one(path: str) -> dict:
    """Run extraction on one image and return a draft fields dict (empty on failure)."""
    image = Image.open(path).convert("RGB")
    try:
        receipt = parse_receipt(extract_receipt(image))
        return {
            "total": receipt.total,
            "subtotal": receipt.subtotal,
            "tax": receipt.tax,
            "line_items": [item.model_dump() for item in receipt.line_items],
        }
    except Exception as error:  # noqa: BLE001 — a bad draft is fine; the human fills it in
        logger.warning("Could not draft %s (%s); leaving blank for manual entry.", path, error)
        return {"total": None, "subtotal": None, "tax": None, "line_items": []}


def draft_labels(images_dir: str = _IMAGES_DIR, labels_path: str = _LABELS_PATH) -> None:
    """Draft labels for every image not already present in labels.json."""
    labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)

    existing_images = {entry.get("image") for entry in labels.values()}
    images = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(_IMAGE_EXTS))

    drafted = 0
    for filename in images:
        if filename in existing_images:
            continue
        document_id = os.path.splitext(filename)[0]
        logger.info("Drafting %s ...", filename)
        labels[document_id] = {"image": filename, "verified": False, **_draft_one(os.path.join(images_dir, filename))}
        drafted += 1

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Drafted {drafted} new label(s). Total entries: {len(labels)}.")
    print(f"Now open {labels_path}, CHECK every field against the photo, fix errors,")
    print('and set "verified": true on each entry you trust. Unverified entries are skipped.')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    draft_labels()
