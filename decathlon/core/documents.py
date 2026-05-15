"""Read-side helpers for the shape of indexed vector-store documents.

`indexing.index` writes products as ``title`` + one section path per branch,
with a space-joined ``path_ids`` union in the metadata. Both `app.search` and
`indexing.query` need to parse that back, so the readers live in `core` to
keep `indexing` from importing `app`.
"""


def ancestor_match(meta: dict, ancestor_id: int) -> bool:
    """True if `ancestor_id` is anywhere on the result's section path.

    `path_ids` is a space-separated union of every category id across all of a
    product's branches, so this matches cross-branch (see the vector-stack
    notes / `indexing.query`).
    """
    return str(ancestor_id) in (meta.get("path_ids") or "").split()


def parse_document(doc: str) -> tuple[str, str]:
    """Split a product document into (title, section path).

    The indexed document is ``title`` followed by one section path per branch
    (see `indexing.index.index_products`).
    """
    if not doc:
        return "", ""
    title = doc.splitlines()[0]
    path = doc.split("\n", 1)[1] if "\n" in doc else ""
    return title, path
