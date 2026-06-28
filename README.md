# Receipt Extraction & Evaluation

A system that reads receipt/invoice images, uses an LLM (Claude) to extract structured fields, validates the output, stores it, and **measures its own accuracy against hand-labeled ground truth**.

This involves wrapping an unreliable model in reliable engineering, and MEASURING how well it works

**Core skills demonstrated:** structured output, Pydantic validation, SQLite storage, deterministic pipeline design, an evaluation harness, and failure analysis.

## How it works

The LLM is **one small, replaceable component** inside a deterministic pipeline. Plain Python controls the flow; only one step is "smart":

```
                                  ┌─────────── the only "smart" step ───────────┐
 CORD / my photos                 │                                             │
   loader.py  ──►  receipt image ──►  extract.py  ──►  validate.py  ──►  store.py  ──►  SQLite
 (uniform Document)               │   (Claude →        (Pydantic parse +     (one table
                                  │    raw JSON text)   bounded retries)      of results)
                                  └─────────────────────────────────────────────┘
                                                      │
                                  eval/ reads results back and scores them
                                  against ground truth → accuracy table + failure dump
```

Each file does one job (`src/schema.py`, `loader.py`, `extract.py`, `validate.py`, `store.py`, `pipeline.py`), and `extract.py` is the *only* file that talks to the LLM. This is important because the "unreliable" part is isolated to ONE file.

### validation vs. evaluation

- **Validation** (runtime, `validate.py`): *"Is this output well-formed?"* Did the model return valid JSON with the right fields and types? It catches garbage but says nothing about whether the answer is *true*. A total of `$9999.99` can be perfectly valid and completely wrong.
- **Evaluation** (offline, `eval/`): *"Across many labeled cases, how often is the system actually correct, and where does it fail?"* This produces the score.

## Project structure

```
src/
  schema.py     Pydantic models — the shape of an extracted receipt
  loader.py     load CORD receipts (and my hand-labeled "messy" set) into a uniform Document
  extract.py    the LLM component: receipt image → raw JSON text (the only file that calls Claude)
  validate.py   parse raw output into the schema; bounded retry on failure
  store.py      SQLite read/write (one table keyed by document_id)
  pipeline.py   the deterministic spine: loader → extract → validate → store
eval/
  scoring.py        per-field comparison rules (normalize, numeric tolerance, line-item P/R/F1)
  run_eval.py       run the pipeline over a set, print accuracy table + failure dump
  draft_labels.py   model-assisted labeling helper for the messy set (drafts; a human verifies)
tests/            unit tests for the deterministic pieces (schema, validate, store, scoring, ...)
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python version and dependencies.

```bash
# Install dependencies into a local .venv (uv reads pyproject.toml + uv.lock)
uv sync

# Provide your Anthropic API key (gitignored, never committed)
echo "ANTHROPIC_API_KEY=sk-..." > .env

# Run anything inside the project environment with `uv run`
uv run python hello_claude.py   # smoke test: prints a reply from Claude
uv run pytest                   # run the test suite
```

- **Add a dependency:** `uv add <package>` (use `uv add --dev <package>` for dev-only tools like pytest).
- **Python version** is pinned in `.python-version` (3.12); `uv` fetches it automatically.

## Running the evaluation

```bash
uv run python -m eval.run_eval          # clean dev set (a slice of CORD's train split)
uv run python -m eval.run_eval messy    # my own hand-labeled "messy" real receipts
```

Each run extracts fields from receipt images, scores them against ground truth, and
prints a per-field accuracy table plus a dump of the specific failures.

CORD's `test` split is reserved as a **held-out set** and is never evaluated during
development, so the dev numbers can't be contaminated by tuning to examples I've seen.

## Results: clean dataset vs. real-world receipts

The point of the project is measuring how much a clean benchmark flatters real-world performance, 
and understanding why. We scoreon two sets: a slice of the clean, curated CORD dataset, 
and a small set of real receipt photos I labeled by hand.

| Field            | Clean (CORD, n=8) | Messy (real photos, n=10) |
|------------------|-------------------|----------------------------|
| `total`          | 88%               | 90%                        |
| `subtotal`       | 100%              | 90%                        |
| `tax`            | 62%               | 80%                        |
| `line_items` F1  | 0.60              | 0.56                       |

The scalar totals transfer well to real photos; the drop is concentrated in line
items. Failure analysis shows the drop is driven less by the model failing on mess
and more by three identifiable, fixable causes:

1. **Image preprocessing.** Large phone photos exceed the API's 10 MB limit, so we
   downscale them. On a long, dense 23-item receipt this made the text unreadable
   and the model hallucinated entirely different items. Long receipts need smarter
   handling (e.g. higher-resolution tiling or splitting) rather than a flat resize.
2. **Schema ambiguity.** Our schema never defined whether `price` is the *unit* price
   or the *line total* (unit × quantity). On a quantity-2 item the model returned the
   line total while the human label used the unit price. The spec,
   not the model, was underspecified.
3. **Strict scoring.** A line item counts as correct only if name, price, *and*
   quantity all match exactly (names normalized only for case/whitespace). Many
   "failures" are near-misses: a one-character name difference, or a predicted
   quantity of 1 where the receipt left quantity blank. Fuzzy name matching and a
   more lenient quantity rule are measurable improvements to try next.

A clean-benchmark score can look healthy while real-world performance
hides preprocessing limits, spec gaps, and measurement choices. Naming those
causes is MUCH more useful than the headline number.

## Next steps

Each of the causes above is a concrete, *measurable* improvement. Fix one, re-run the
eval on the dev set, then confirm once on the held-out set:

- Pin down `price` (unit vs. line total) in `schema.py` and the extraction prompt.
- Add fuzzy name matching to line-item scoring to credit near-miss names.
- Relax the quantity rule (treat a missing quantity as matching 1).
- Handle long receipts better than a flat downscale.

## Testing

```bash
uv run pytest
```

Deterministic pieces (schema, validation, storage, scoring) are unit-tested. However, the LLM
step is deliberately *not* unit-tested for correctness because there is no single hard-coded 
right answer to assert. It is instead tested with the Evaluation Harness.
```
