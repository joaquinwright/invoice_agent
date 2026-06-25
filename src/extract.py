"""
The LLM extraction component: a receipt image -> structured fields (as text).

This is the ONLY file in the project that talks to the LLM (CLAUDE.md §2.6, §2.9).
Keeping all Claude calls here means the "smart but unreliable" part of the system
is one small, replaceable box; everything around it stays boring and deterministic.

WHAT THIS DOES (Phase 3): send the receipt photo to Claude and ask it to return our
fields as JSON. We return Claude's RAW TEXT reply, unparsed and unvalidated, on
purpose. Phase 3 is where we feel the "the model returns slightly-off JSON" problem
firsthand (CLAUDE.md Phase 3.3); Phase 4 adds the Pydantic parsing + bounded retries
that tame it. The SDK can also *force* schema-valid output via structured outputs
(`client.messages.parse(...)`) — the production-grade approach — but adopting it now
would skip that lesson, so we hold it for a later improvement.

Auth: the ANTHROPIC_API_KEY is read from the gitignored .env file by python-dotenv;
the Anthropic SDK then picks it up from the environment automatically. The key is
never written in code (CLAUDE.md §2.3).
"""

import base64
import io

import anthropic
from dotenv import load_dotenv
from PIL import Image

# Load .env into the process environment so the SDK finds ANTHROPIC_API_KEY.
load_dotenv()

# Claude is vision-capable, so it reads the receipt photo directly (no OCR step).
# We use Haiku — the cheapest current model (~$1/$5 per 1M input/output tokens, ~5x
# cheaper than Opus) — deliberately, to stretch a small API budget on a learning-scale
# project. The production default would be a stronger model like claude-opus-4-8; if
# our eval later shows Haiku is the accuracy bottleneck, upgrading the model is one of
# the improvements we can measure (CLAUDE.md Phase 8).
_MODEL = "claude-haiku-4-5"

# The instruction we send alongside the image. We describe the EXACT fields from our
# locked schema (see src/schema.py) and demand JSON-only output — no prose — so the
# reply is something we can later parse. Note: this is a request, not a guarantee;
# that gap is exactly what Phase 4 handles.
_SYSTEM_PROMPT = """You extract structured data from receipt images.

Return ONLY a single JSON object — no prose, no markdown fences, no explanation —
with exactly these fields:

  - "line_items": a list of objects, each with:
      - "name":     the item's name (string)
      - "price":    the item's price as a number (no currency symbols, no thousands
                    separators — e.g. write 75000, not "75,000")
      - "quantity": how many were bought (integer), or null if not shown
  - "total":    the receipt's grand total as a number
  - "subtotal": the subtotal as a number, or null if not shown
  - "tax":      the tax amount as a number, or null if not shown

Output the JSON object and nothing else."""

# One Anthropic client, reused across calls. Reads ANTHROPIC_API_KEY from the env.
_client = anthropic.Anthropic()


def _image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string for the Anthropic image block.

    The API takes image bytes as base64 text inside the request JSON, so we render
    the in-memory PIL image to PNG bytes and base64-encode them.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def extract_receipt(image: Image.Image) -> str:
    """Send one receipt image to Claude and return its RAW text reply (unparsed).

    The reply *should* be a JSON object matching our schema, but we deliberately do
    not parse or validate it here — that is Phase 4's job. A caller in Phase 3 just
    prints this to see what the model actually produces.
    """
    image_b64 = _image_to_base64_png(image)

    response = _client.messages.create(
        model=_MODEL,
        max_tokens=4096,  # a receipt's JSON is small; this is comfortable headroom
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": "Extract the fields from this receipt."},
                ],
            }
        ],
    )

    # response.content is a list of typed blocks; for a plain text reply we want the
    # text of the first (and here, only) text block.
    return next(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    # Demo: extract ONE document and print Claude's raw output beside the ground truth.
    # Hard-capped at a single document — every call costs tokens (CLAUDE.md §2.3).
    import json

    from src.loader import load_cord

    doc = next(load_cord(split="train", limit=1))
    print(f"document_id: {doc.document_id}\n")

    print("=== Claude's RAW reply (unparsed) ===")
    raw = extract_receipt(doc.image)
    print(raw)

    print("\n=== Ground truth (for eyeballing only) ===")
    print(json.dumps(doc.ground_truth, indent=2, ensure_ascii=False))
