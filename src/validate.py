"""
Turn Claude's raw text reply into a validated Receipt — or fail loudly but safely.

This is the VALIDATION half of the project (CLAUDE.md §1, §2.7): a runtime check that
the model's output is *well-formed* (valid JSON, right fields, right types). It does
NOT check that the values are *correct* — a wrong-but-well-formed price still passes
here. Correctness is the eval's job (Phase 6).

Phase 4.1: `parse_receipt` cleans and parses one raw reply into a Receipt, raising a
clear error when the text isn't well-formed.
Phase 4.2: `extract_and_validate` re-prompts up to N times on failure and logs instead
of crashing, so one bad document never takes down a whole run.
"""

import logging
from typing import Callable

from pydantic import ValidationError

from src.schema import Receipt

# Module logger. Whoever runs the pipeline decides where these messages go (console,
# file, ...) by configuring logging; this file just emits them.
logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove a leading ```/```json fence and trailing ``` from a model reply.

    Claude often wraps JSON in a markdown code block (we saw this in Phase 3). Those
    backticks are not valid JSON, so `json.loads` / Pydantic would choke on them. We
    peel them off here. If there's no fence, the text is returned unchanged.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json), keeping the rest.
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        # Drop the closing fence if present.
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


def parse_receipt(raw: str) -> Receipt:
    """Parse one raw model reply into a validated Receipt.

    Raises pydantic.ValidationError if the cleaned text isn't well-formed JSON that
    matches the Receipt schema. Callers (the retry loop) catch that to decide whether
    to re-prompt.
    """
    cleaned = _strip_code_fences(raw)
    return Receipt.model_validate_json(cleaned)


def extract_and_validate(
    extractor: Callable[[], str],
    *,
    document_id: str = "?",
    max_retries: int = 2,
) -> Receipt | None:
    """Get a raw reply, validate it, and re-prompt on failure up to `max_retries` times.

    `extractor` is a zero-argument function that returns one raw model reply. We accept
    it as an argument (dependency injection) rather than calling the LLM directly so the
    retry logic is testable with a fake extractor — no API calls, no tokens. In the real
    pipeline the caller passes e.g. `lambda: extract_receipt(doc.image)`.

    Returns a validated Receipt, or None if every attempt failed (the failure is logged,
    not raised — a single bad document must never crash the run, CLAUDE.md §2.7).
    """
    attempts = max_retries + 1  # one initial try plus `max_retries` re-prompts
    for attempt in range(1, attempts + 1):
        raw = extractor()
        try:
            return parse_receipt(raw)
        except ValidationError as error:
            logger.warning(
                "Validation failed for %s (attempt %d/%d): %s",
                document_id,
                attempt,
                attempts,
                error.errors(include_url=False),
            )

    logger.error("Giving up on %s after %d attempts; logging as a failure.", document_id, attempts)
    return None
