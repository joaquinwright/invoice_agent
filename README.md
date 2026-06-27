# Receipt Extraction & Evaluation

A system that reads receipt/invoice documents, uses an LLM (Claude) to extract structured fields, and measures extraction accuracy against ground-truth labels.

**Core skills demonstrated:** structured output, Pydantic validation, SQLite storage, deterministic pipeline design, evaluation harness, failure analysis.

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

## Results: clean dataset vs. real-world receipts

The point of the project is not a single accuracy number — it's measuring how much a
clean benchmark flatters real-world performance, and understanding why. We score
on two sets: a slice of the clean, curated CORD dataset, and a small set of real
receipt photos I labeled by hand.

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
   line total while the human label used the unit price — both defensible. The spec,
   not the model, was underspecified.
3. **Strict scoring.** A line item counts as correct only if name, price, *and*
   quantity all match exactly (names normalized only for case/whitespace). Many
   "failures" are near-misses: a one-character name difference, or a predicted
   quantity of 1 where the receipt left quantity blank. Fuzzy name matching and a
   more lenient quantity rule are measurable improvements to try next.

The takeaway: a clean-benchmark score can look healthy while real-world performance
hides preprocessing limits, spec gaps, and measurement choices. Naming those
causes is MUCH more useful than the headline number.
