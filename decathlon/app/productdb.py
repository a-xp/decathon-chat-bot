"""Runtime read access to `products.db` for the `get_product` tool.

The Chroma `products` collection only carries the lean metadata needed for
search/cards; the rich characteristics a shopper asks about (composition,
technical specs, benefits, available sizes/colours) live only in the scraped
`raw_json` blob. This is the single place that reads the SQLite db at request
time. Best-effort and read-only: returns ``None`` rather than raising when a
product is missing.
"""

import json
import logging
import sqlite3

from decathlon.app.catalog import DB_PATH
from decathlon.core.documents import parse_document
from decathlon.core.vectordb import PRODUCTS, get_client, get_collection

logger = logging.getLogger(__name__)


def _unwrap(raw: dict, key: str):
    """Pull a Decathlon ``{"value": ...}`` field, JSON-decoding stringified
    lists/objects. Returns the parsed value, or None when absent/empty."""
    node = raw.get(key)
    val = node.get("value") if isinstance(node, dict) else node
    if val in (None, "", "null"):
        return None
    if isinstance(val, str):
        s = val.strip()
        if s and s[0] in "[{":
            try:
                return json.loads(s)
            except ValueError:
                return val
    return val


def _section_path(product_id: str) -> str:
    """The product's category path(s), reused from the indexed Chroma doc."""
    try:
        got = get_collection(get_client(), PRODUCTS).get(
            ids=[product_id], include=["documents"]
        )
        docs = got.get("documents") or []
        if docs and docs[0]:
            return parse_document(docs[0])[1]
    except Exception as e:  # noqa: BLE001 - best-effort enrichment
        logger.warning("Could not load section path for %s: %s", product_id, e)
    return ""


def get_product_details(product_id: str) -> dict | None:
    """Full characteristics for one product, for exploration/comparison.

    Returns a compact, model-friendly dict (no stock/availability claims), or
    ``None`` if the id is unknown.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute(
                "SELECT id, handle, title, description, brand, model_code, "
                "price, compare_at_price, available, image_url, raw_json "
                "FROM products WHERE id = ?",
                (product_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("products.db read failed for %s: %s", product_id, e)
        return None

    if row is None:
        return None

    (pid, handle, title, description, brand, model_code,
     price, compare_at_price, available, image_url, raw_json) = row

    try:
        raw = json.loads(raw_json) if raw_json else {}
    except (TypeError, ValueError):
        raw = {}

    # Distinct variant option titles (e.g. "Зеленый / S") = sizes/colours.
    variants = []
    for v in (raw.get("variants") or {}).get("nodes") or []:
        t = (v or {}).get("title")
        if t and t not in variants:
            variants.append(t)

    details: dict = {
        "id": pid,
        "handle": handle,
        "title": title or "",
        "section_path": _section_path(str(pid)),
        "brand": brand,
        "model_code": model_code,
        "price": price,
        "compare_at_price": compare_at_price or None,
        "available": bool(available) if available is not None else None,
        "image_url": image_url,
        "description": description or _unwrap(raw, "description"),
        "catch_line": _unwrap(raw, "web_catch_line"),
        "plus_point": _unwrap(raw, "plus_point"),
        "designed_for": _unwrap(raw, "designed_for"),
        "composition": _unwrap(raw, "composition"),
        "benefits": _unwrap(raw, "benefits"),
        "technical_specs": _unwrap(raw, "technicals"),
        "characteristics": _unwrap(raw, "characteristics"),
        "care_instructions": _unwrap(raw, "care_instructions"),
        "variants": variants or None,
    }
    # Drop empty keys so the tool payload stays compact.
    return {k: v for k, v in details.items() if v not in (None, "", [], {})}
