"""Embed products + categories from products.db into the Chroma store.

Run: `python -m decathlon.indexing.index`  (or `mise run index-vectors`)

`products.category_id` is always a top-level Shopify collection id, so the
real (leaf) section is resolved from the product's `category:<n>` tags, where
`<n>` is the numeric prefix of `categories.slug` (e.g. `category:232` ->
slug `232-turizm-...`). Section paths are stored/embedded WITHOUT the root
level (e.g. "Виды спорта"), which carries no signal.

A product usually belongs to several branches, so its document carries one
section path per branch (root level stripped).

Product document  = title + one section path per branch (no root).
Category document = ancestor path (no root).
"""

import json
import logging
import sqlite3

from decathlon.core.embeddings import EMBED_BATCH, embed_texts
from decathlon.core.vectordb import CATEGORIES, PRODUCTS, get_client, get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "products.db"


def load_categories(conn: sqlite3.Connection) -> dict[int, dict]:
    cats: dict[int, dict] = {}
    for cid, name, slug, parent_id, level in conn.execute(
        "SELECT id, name, slug, parent_id, level FROM categories"
    ):
        cats[cid] = {
            "name": name,
            "slug": slug,
            "parent_id": parent_id,
            "level": level,
        }
    return cats


def legacy_index(cats: dict[int, dict]) -> dict[int, int]:
    """Map the slug's numeric prefix (used in `category:` tags) -> cid."""
    by_legacy: dict[int, int] = {}
    for cid, c in cats.items():
        slug = c["slug"] or ""
        head = slug.split("-", 1)[0]
        if head.isdigit():
            by_legacy[int(head)] = cid
    return by_legacy


def ancestors(cats: dict[int, dict], cat_id) -> list[tuple[int, str]]:
    """Ordered [(id, name), ...] from root -> the given category."""
    chain: list[tuple[int, str]] = []
    seen: set[int] = set()
    cur = cat_id
    while cur is not None and cur in cats and cur not in seen:
        seen.add(cur)
        chain.append((cur, cats[cur]["name"]))
        cur = cats[cur]["parent_id"]
    chain.reverse()
    return chain


def path_without_root(cats: dict[int, dict], chain) -> list[str]:
    """Names along the chain, dropping level-0 (root) sections."""
    return [name for cid, name in chain if cats[cid]["level"] not in (0, None)]


def product_chains(cats, by_legacy, tags, fallback_cat_id):
    """Resolve a product's section path(s) from its `category:` tags.

    The category tree is not linear: a product's tags usually span several
    branches, each fully expanded (every ancestor is also a tag). We keep
    only the *leaf* candidates (those that are not an ancestor of another
    candidate) and return one chain per leaf.
    """
    candidates = set()
    for t in tags:
        if t.startswith("category:"):
            try:
                legacy = int(t.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            cid = by_legacy.get(legacy)
            if cid is not None:
                candidates.add(cid)
    if not candidates:
        chain = ancestors(cats, fallback_cat_id)
        return [chain] if chain else []

    # Drop any candidate that is a strict ancestor of another candidate.
    inner = set()
    for c in candidates:
        for cid, _ in ancestors(cats, c)[:-1]:
            if cid in candidates:
                inner.add(cid)
    leaves = candidates - inner
    return [ancestors(cats, leaf) for leaf in leaves]


def _tag_id(tags: list[str], prefix: str):
    for t in tags:
        if t.startswith(prefix):
            try:
                return int(t.split(":", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


def _flush(collection, ids, docs, metas) -> None:
    if not ids:
        return
    embeddings = embed_texts(docs)
    collection.upsert(
        ids=ids, documents=docs, embeddings=embeddings, metadatas=metas
    )
    logger.info("Upserted batch of %d into '%s'", len(ids), collection.name)


def index_categories(conn, cats, collection) -> None:
    ids, docs, metas = [], [], []
    total = 0
    for cid, c in cats.items():
        chain = ancestors(cats, cid)
        path_names = path_without_root(cats, chain)
        meta = {
            "name": c["name"] or "",
            "slug": c["slug"] or "",
            "level": c["level"] if c["level"] is not None else -1,
            "root_section_id": chain[0][0] if chain else cid,
            "path_ids": " ".join(str(i) for i, _ in chain),
        }
        if c["parent_id"] is not None:
            meta["parent_id"] = c["parent_id"]
        ids.append(f"cat-{cid}")
        docs.append(" / ".join(path_names) or (c["name"] or ""))
        metas.append(meta)
        if len(ids) >= EMBED_BATCH:
            _flush(collection, ids, docs, metas)
            total += len(ids)
            ids, docs, metas = [], [], []
    _flush(collection, ids, docs, metas)
    total += len(ids)
    logger.info("Indexed %d categories.", total)


def index_products(conn, cats, by_legacy, collection) -> None:
    ids, docs, metas = [], [], []
    total = 0
    rows = conn.execute(
        "SELECT id, title, category_id, tags, brand, price, available, "
        "handle, image_url, raw_json FROM products"
    )
    for (pid, title, cat_id, tags_json, brand, price, available,
         handle, image_url, raw_json_str) in rows:
        try:
            tags = json.loads(tags_json) if tags_json else []
        except (TypeError, ValueError):
            tags = []

        chains = product_chains(cats, by_legacy, tags, cat_id)
        # One readable path per branch, de-duplicated, order-stable.
        paths, seen_paths = [], set()
        for ch in chains:
            p = " / ".join(path_without_root(cats, ch))
            if p and p not in seen_paths:
                seen_paths.add(p)
                paths.append(p)
        title = title or ""
        document = f"{title}\n" + "\n".join(paths) if paths else title

        all_ids = sorted({i for ch in chains for i, _ in ch})
        # Primary branch = the deepest one (for the scalar section filters).
        primary = max(chains, key=len) if chains else []
        meta = {
            "available": bool(available),
            "path_ids": " ".join(str(i) for i in all_ids),
        }
        if cat_id is not None:
            meta["category_id"] = cat_id
        if primary:
            meta["root_section_id"] = primary[0][0]
            meta["section_id"] = primary[-1][0]
        for key, prefix in (
            ("sport_id", "sport:"),
            ("gender_id", "gender:"),
        ):
            v = _tag_id(tags, prefix)
            if v is not None:
                meta[key] = v
        if brand:
            meta["brand"] = brand
        if price is not None:
            meta["price"] = float(price)
        if handle:
            meta["handle"] = handle
        if image_url:
            meta["image_url"] = image_url
        try:
            raw = json.loads(raw_json_str) if raw_json_str else {}
        except (TypeError, ValueError):
            raw = {}
        size_vals: list[str] = []
        seen_sizes: set[str] = set()
        color_vals: list[str] = []
        seen_colors: set[str] = set()
        for v in (raw.get("variants") or {}).get("nodes") or []:
            for opt in (v or {}).get("selectedOptions") or []:
                name = opt.get("name")
                val = (opt.get("value") or "").strip()
                if not val:
                    continue
                if name == "Size":
                    if val != "no_size:1" and val not in seen_sizes:
                        seen_sizes.add(val)
                        size_vals.append(val)
                elif name == "Color":
                    lc = val.lower()
                    if lc not in seen_colors:
                        seen_colors.add(lc)
                        color_vals.append(lc)
        if size_vals:
            meta["sizes"] = "\x1f".join(size_vals)
        if color_vals:
            meta["colors"] = "\x1f".join(color_vals)

        ids.append(str(pid))
        docs.append(document)
        metas.append(meta)
        if len(ids) >= EMBED_BATCH:
            _flush(collection, ids, docs, metas)
            total += len(ids)
            ids, docs, metas = [], [], []
    _flush(collection, ids, docs, metas)
    total += len(ids)
    logger.info("Indexed %d products.", total)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cats = load_categories(conn)
        by_legacy = legacy_index(cats)
        logger.info(
            "Loaded %d categories (%d legacy-mapped) from %s",
            len(cats), len(by_legacy), DB_PATH,
        )
        client = get_client()
        index_categories(conn, cats, get_collection(client, CATEGORIES))
        index_products(
            conn, cats, by_legacy, get_collection(client, PRODUCTS)
        )
    finally:
        conn.close()
    logger.info("Vector indexing finished.")


if __name__ == "__main__":
    main()
