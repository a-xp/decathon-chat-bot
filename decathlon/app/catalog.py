"""Category catalog used to ground the router's category choice.

Level-0 (root) sections carry no useful signal on their own, but their name
disambiguates the deeper categories for the router (e.g. "Мужчины / Обувь" vs
"Детям / Детская обувь", which would otherwise both collapse to "Обувь"). We
therefore present each category as its full ancestor path
``"<root> / <level-1> [ / <level-2>]"`` and map that display string back to the
category id, which is then used as an ancestor filter over `path_ids`.

Both level-1 and level-2 categories are exposed so the router can pick a
fine-grained subtree when it is confident, or fall back to a broad level-1 one
otherwise. An over-specific pick is safe: `search.search_products` treats the
category as a soft preference and backfills, so it can never collapse results.
"""

import sqlite3

DB_PATH = "products.db"
PRODUCT_URL = "https://decathlon.kz/p/{handle}"

# Deepest category level offered to the router. Level 1 is broad ("Обувь"),
# level 2 is the useful shopping granularity ("Обувь / Кроссовки для бега").
MAX_CATEGORY_LEVEL = 2


def load_categories(
    db_path: str = DB_PATH, max_level: int = MAX_CATEGORY_LEVEL
) -> list[dict]:
    """Return categories of level 1..`max_level` with their full ancestor path.

    Each item: ``{"id", "name", "level", "display"}`` where ``display`` is the
    ``" / "``-joined chain of names from the level-0 root down to the category.
    """
    conn = sqlite3.connect(db_path)
    try:
        names = dict(conn.execute("SELECT id, name FROM categories"))
        parents = dict(conn.execute("SELECT id, parent_id FROM categories"))
        rows = conn.execute(
            "SELECT id, name, level FROM categories "
            "WHERE level BETWEEN 1 AND ? ORDER BY level, id",
            (max_level,),
        ).fetchall()
    finally:
        conn.close()

    def path_display(cid: int) -> str:
        chain = []
        node: int | None = cid
        while node is not None:
            chain.append(names.get(node) or "")
            node = parents.get(node)
        return " / ".join(p for p in reversed(chain) if p)

    cats: list[dict] = []
    for cid, name, level in rows:
        cats.append(
            {
                "id": cid,
                "name": name or "",
                "level": level,
                "display": path_display(cid),
            }
        )
    return cats


_CATS = load_categories()

# Display strings (level-1 first, then level-2; ready to drop into a prompt).
CATEGORY_DISPLAYS: list[str] = [c["display"] for c in _CATS]

# Display string -> category id (used as the ancestor filter).
DISPLAY_TO_ID: dict[str, int] = {c["display"]: c["id"] for c in _CATS}

# Category id -> display string. Used to turn a semantic hit in the
# `categories` Chroma collection (id `cat-<n>`) back into a full path.
ID_TO_DISPLAY: dict[int, str] = {c["id"]: c["display"] for c in _CATS}
