"""Product vector search exposed to the chat agent as the `search_products`
tool.

Thin wrapper over the Chroma `products` collection: embeds the query with
bge-m3, optionally applies a structured `gender_id` pre-filter, a hard category
ancestor filter on `path_ids` (any of the given categories), and a brand
substring filter, then returns structured product dicts.

Category is a hard filter here (the agent picks categories deliberately via the
`find_categories` tool), but it relaxes to unfiltered when it would starve
results — the same shape as the gender fallback.
"""

import logging

from decathlon.app.catalog import DISPLAY_TO_ID
from decathlon.core.documents import ancestor_match, parse_document
from decathlon.core.embeddings import embed_texts
from decathlon.core.vectordb import PRODUCTS, get_client, get_collection

logger = logging.getLogger(__name__)

# Gender -> the set of `gender_id` metadata values that satisfy it.
# `gender_id` is a clean, reliable facet, so it is applied as a near-hard
# pre-filter (with a relax fallback when it would starve results). 5 = generic
# kids (not split by sex), so it is included for boys/girls too. 13 = unisex
# gear and is intentionally NOT mapped (a gendered request rarely wants
# generic equipment).
GENDER_IDS: dict[str, list[int]] = {
    "men": [2, 1],
    "women": [3],
    "boys": [8, 5],
    "girls": [6, 9, 5],
    "kids": [5, 6, 8, 9, 4],
}

# Valid gender facet values (the `gender` tool-arg enum).
GENDERS: list[str] = list(GENDER_IDS)


def search_products(
    query: str,
    categories: list[str] | None = None,
    n: int = 10,
    gender: str | None = None,
    brand: str | None = None,
    size: str | None = None,
) -> list[dict]:
    """Semantic search over the catalog.

    `categories` is a list of values from `catalog.CATEGORY_DISPLAYS`; a result
    is kept if it falls under ANY of them (hard filter, relaxed if it starves
    results). `gender` is a `GENDERS` value applied as a structured `gender_id`
    pre-filter so the query embedding can stay a pure product description.
    `brand` is a case-insensitive substring matched against the product brand.
    """
    vec = embed_texts([query])[0]

    ancestor_ids: list[int] = []
    for disp in categories or []:
        cid = DISPLAY_TO_ID.get(disp)
        if cid is None:
            logger.warning("Unknown category %r, ignoring", disp)
        else:
            ancestor_ids.append(cid)

    gender_ids = GENDER_IDS.get(gender) if gender else None
    where = {"gender_id": {"$in": gender_ids}} if gender_ids else None

    brand_lc = brand.strip().lower() if brand and brand.strip() else None
    size_lc = size.strip().lower() if size and size.strip() else None

    # Over-fetch when post-filtering client-side (category, brand, and/or size).
    post_filtered = bool(ancestor_ids) or brand_lc is not None or size_lc is not None
    n_results = n * 5 if post_filtered else n

    client = get_client()
    collection = get_collection(client, PRODUCTS)

    def run(where_clause):
        return collection.query(
            query_embeddings=[vec],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

    res = run(where)
    # Relax the gender filter if it starved results (e.g. an item only tagged
    # unisex): a thinner gender signal beats returning almost nothing.
    if where is not None and len(res["ids"][0]) < n:
        logger.info(
            "Gender filter %r left %d (<%d) results; relaxing",
            gender, len(res["ids"][0]), n,
        )
        res = run(None)

    rows = list(zip(  # already distance-sorted
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0],
    ))

    def to_product(rid, doc, meta, dist) -> dict:
        title, section_path = parse_document(doc)
        return {
            "id": rid,
            "title": title,
            "section_path": section_path,
            "price": meta.get("price"),
            "brand": meta.get("brand"),
            "handle": meta.get("handle"),
            "image_url": meta.get("image_url"),
            "available": meta.get("available"),
            "distance": dist,
        }

    def matches(meta) -> bool:
        if ancestor_ids and not any(
            ancestor_match(meta, aid) for aid in ancestor_ids
        ):
            return False
        if brand_lc is not None and brand_lc not in (
            (meta.get("brand") or "").lower()
        ):
            return False
        if size_lc is not None:
            stored = (meta.get("sizes") or "").lower()
            if not any(size_lc in tok for tok in stored.split()):
                return False
        return True

    if not post_filtered:
        return [to_product(*r) for r in rows[:n]]

    filtered = [r for r in rows if matches(r[2])]
    # The category/brand pick is the agent's deliberate intent, so it is a
    # hard filter — but never let an over-specific pick collapse results:
    # if it leaves too few, fall back to the raw semantic ranking.
    if len(filtered) < n:
        logger.info(
            "Category/brand filter left %d (<%d) results; relaxing to "
            "semantic ranking", len(filtered), n,
        )
        return [to_product(*r) for r in rows[:n]]
    return [to_product(*r) for r in filtered[:n]]
