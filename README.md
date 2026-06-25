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
