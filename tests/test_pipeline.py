"""
Unit tests for src/pipeline.py's deterministic parts.

The pipeline contains the LLM step, so we don't test "did extraction get the right
answer" here — that's the eval's job. What is deterministic and worth testing: the
HARD_CAP cost rail and the routing logic (a valid receipt -> save_receipt; a None ->
save_failure; counts add up). We test those by replacing the LLM-touching
collaborators with fakes, so the test loads no dataset and spends no tokens.

monkeypatch is pytest's built-in tool for temporarily replacing a name for one test,
then restoring it.

Run with:  uv run pytest
"""

import pytest

from src import pipeline
from src.schema import Receipt


def make_fake_docs(ids):
    """Fake loader output: objects with the two attributes the pipeline reads."""
    class FakeDoc:
        def __init__(self, document_id):
            self.document_id = document_id
            self.image = None  # never used, because we also fake the extractor

    return [FakeDoc(i) for i in ids]


def a_receipt():
    return Receipt.model_validate({"total": 8.5, "line_items": [{"name": "X", "price": 8.5}]})


def test_hard_cap_fires_before_any_work(monkeypatch):
    # If the cap check is wrong, this would try to load the dataset. We make the
    # loader explode if reached, then assert it raises before that.
    monkeypatch.setattr(pipeline, "load_cord", lambda **_: (_ for _ in ()).throw(AssertionError("loaded!")))
    with pytest.raises(ValueError, match="HARD_CAP"):
        pipeline.run_pipeline(limit=pipeline.HARD_CAP + 1)


def test_routes_ok_and_failures_and_counts(monkeypatch, tmp_path):
    docs = make_fake_docs(["d0", "d1", "d2"])
    monkeypatch.setattr(pipeline, "load_cord", lambda **_: iter(docs))

    # Fake validate step: d1 fails (returns None), the others succeed.
    def fake_validate(extractor, *, document_id, max_retries):
        return None if document_id == "d1" else a_receipt()

    monkeypatch.setattr(pipeline, "extract_and_validate", fake_validate)

    # Record which store function each document hit.
    saved, failed = [], []
    monkeypatch.setattr(pipeline, "save_receipt", lambda conn, doc_id, r: saved.append(doc_id))
    monkeypatch.setattr(pipeline, "save_failure", lambda conn, doc_id: failed.append(doc_id))

    summary = pipeline.run_pipeline(limit=3, db_path=str(tmp_path / "t.db"))

    assert summary == {"ok": 2, "failed": 1, "total": 3}
    assert saved == ["d0", "d2"]
    assert failed == ["d1"]
