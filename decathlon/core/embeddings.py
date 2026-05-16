"""Embeddings via the OpenAI-compatible endpoint (bge-m3).

Reuses the same OPENAI_BASE_URL / OPENAI_API_KEY config as the chat app;
the embedding model is configured separately via OPENAI_EMBED_MODEL.
"""

import logging
import os

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load a local .env if present (real env vars take precedence).
load_dotenv()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "http://localhost:1234/v1"
).rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "bge-m3")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
REQUEST_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed `texts`, returning one vector per input in input order."""
    if not texts:
        return []

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    vectors: list[list[float]] = []

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            resp = client.post(
                f"{OPENAI_BASE_URL}/embeddings",
                json={"model": OPENAI_EMBED_MODEL, "input": batch},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json().get("data") or []
            # The endpoint may not preserve order; sort by `index` to be safe.
            data.sort(key=lambda d: d.get("index", 0))
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch: asked {len(batch)}, "
                    f"got {len(data)}"
                )
            vectors.extend(d["embedding"] for d in data)

    return vectors
