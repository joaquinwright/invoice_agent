"""
The LLM extraction component: a receipt image -> structured fields (as text).

This is the only file that talks to the LLM. Keeping all Claude calls here means
the unreliable part of the system is one small, replaceable box.

It sends the receipt photo to Claude and returns the raw text reply, unparsed and
unvalidated — parsing and retries happen in validate.py. (The SDK can also force
schema-valid output via client.messages.parse(...); we keep the manual approach
for now.)

Auth: ANTHROPIC_API_KEY is loaded from the gitignored .env by python-dotenv and
picked up by the SDK automatically. The key never appears in code.
"""

import base64
import io

import anthropic
from dotenv import load_dotenv
from PIL import Image

# Load .env so the SDK finds ANTHROPIC_API_KEY.
load_dotenv()

# Claude reads the photo directly, so there's no separate OCR step. We use Haiku,
# the cheapest current model, to stretch a small budget; a stronger model like
# claude-opus-4-8 would be the production default and is an easy thing to try
# later if accuracy needs it.
_MODEL = "claude-haiku-4-5"

# The instruction sent with the image. It names the exact fields from our schema
# (see src/schema.py) and demands JSON-only output. This is a request, not a
# guarantee — validate.py handles the cases where the model doesn't comply.
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

    The API takes image bytes as base64 text, so we render the image to PNG bytes
    and base64-encode them.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def extract_receipt(image: Image.Image) -> str:
    """Send one receipt image to Claude and return its raw text reply (unparsed).

    The reply should be JSON matching our schema, but we don't parse or validate it
    here — that's validate.py's job.
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

    # response.content is a list of typed blocks; we want the text of the first
    # text block.
    return next(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    # Demo: extract one document and print Claude's raw output beside the ground
    # truth. Capped at a single document since every call costs tokens.
    import json

    from src.loader import load_cord

    doc = next(load_cord(split="train", limit=1))
    print(f"document_id: {doc.document_id}\n")

    print("=== Claude's RAW reply (unparsed) ===")
    raw = extract_receipt(doc.image)
    print(raw)

    print("\n=== Ground truth (for eyeballing only) ===")
    print(json.dumps(doc.ground_truth, indent=2, ensure_ascii=False))
