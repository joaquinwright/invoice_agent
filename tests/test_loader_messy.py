"""
Unit tests for load_messy in src/loader.py.

The important, deterministic behavior to lock down: only VERIFIED labels are
yielded. An unverified label is a model draft, and grading against it would be
meaningless — so the loader must skip it. No LLM, no network; we build a tiny
fixture (two blank images + a labels.json) on disk in a temp dir.

Run with:  uv run pytest
"""

import json

from PIL import Image

from src.loader import load_messy


def _build_fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10)).save(images / "a.png")
    Image.new("RGB", (10, 10)).save(images / "b.png")

    labels = {
        "a": {
            "image": "a.png",
            "verified": True,
            "total": 100,
            "subtotal": 100,
            "tax": None,
            "line_items": [{"name": "Coffee", "price": 100, "quantity": 1}],
        },
        "b": {  # a draft the human hasn't checked yet
            "image": "b.png",
            "verified": False,
            "total": 50,
            "subtotal": 50,
            "tax": None,
            "line_items": [],
        },
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return str(labels_path), str(images)


def test_only_verified_labels_are_yielded(tmp_path):
    labels_path, images_dir = _build_fixture(tmp_path)
    docs = list(load_messy(labels_path, images_dir))
    # "b" is unverified and must be skipped.
    assert [d.document_id for d in docs] == ["a"]


def test_verified_document_has_expected_shape(tmp_path):
    labels_path, images_dir = _build_fixture(tmp_path)
    doc = next(load_messy(labels_path, images_dir))
    assert doc.ground_truth["total"] == 100
    assert doc.ground_truth["tax"] is None
    assert doc.ground_truth["line_items"][0]["name"] == "Coffee"
    assert doc.image.size == (10, 10)
