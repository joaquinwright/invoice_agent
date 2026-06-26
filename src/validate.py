"""
Turn Claude's raw text reply into a validated Receipt — or fail safely.

This is the validation step: a runtime check that the output is well-formed (valid
JSON, right fields, right types). It does NOT check that the values are correct —
that's the eval's job.

parse_receipt cleans and parses one raw reply into a Receipt, raising on bad input.
extract_and_validate re-prompts up to N times on failure and logs instead of
crashing, so one bad document never takes down a run.
"""

import logging
from typing import Callable

from pydantic import ValidationError

from src.schema import Receipt

# Module logger. Whoever runs the pipeline decides where these messages go.
logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove a leading ```/```json fence and trailing ``` from a model reply.

    Claude often wraps JSON in a markdown code block; those backticks aren't valid
    JSON, so we peel them off. Text without a fence is returned unchanged.
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

    Raises pydantic.ValidationError if the cleaned text isn't well-formed JSON
    matching the Receipt schema. The retry loop catches that to decide whether to
    re-prompt.
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

    `extractor` is a zero-argument function that returns one raw model reply. Passing
    it in (instead of calling the LLM directly) lets us test the retry logic with a
    fake extractor — no API calls, no tokens. The real pipeline passes
    `lambda: extract_receipt(doc.image)`.

    Returns a validated Receipt, or None if every attempt failed. A failure is
    logged, not raised, so one bad document never crashes the run.
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
