"""Semantic search over the Chroma store.

Examples:
  python -m decathlon.indexing.query "ботинки для бега зимой" -n 5
  python -m decathlon.indexing.query "куртка" --section 1
  python -m decathlon.indexing.query "обувь" --collection categories
  python -m decathlon.indexing.query "шорты" --ancestor 583
"""

import argparse

from decathlon.core.documents import ancestor_match, parse_document
from decathlon.core.embeddings import embed_texts
from decathlon.core.vectordb import CATEGORIES, PRODUCTS, get_client, get_collection


def build_where(args) -> dict | None:
    clauses = []
    if args.section is not None:
        clauses.append({"root_section_id": args.section})
    if args.sport is not None:
        clauses.append({"sport_id": args.sport})
    if args.gender is not None:
        clauses.append({"gender_id": args.gender})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector search over the catalog.")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument(
        "--collection", choices=[PRODUCTS, CATEGORIES], default=PRODUCTS
    )
    parser.add_argument("--section", type=int, help="root_section_id filter")
    parser.add_argument("--sport", type=int, help="sport_id filter")
    parser.add_argument("--gender", type=int, help="gender_id filter")
    parser.add_argument(
        "--ancestor", type=int,
        help="keep only results whose section path contains this category id",
    )
    parser.add_argument("-n", type=int, default=10, help="number of results")
    args = parser.parse_args()

    vec = embed_texts([args.query])[0]
    where = build_where(args)
    # Over-fetch when we need to post-filter by ancestor id client-side.
    n_results = args.n * 5 if args.ancestor is not None else args.n

    client = get_client()
    collection = get_collection(client, args.collection)
    res = collection.query(
        query_embeddings=[vec],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    shown = 0
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        if args.ancestor is not None and not ancestor_match(meta, args.ancestor):
            continue
        first_line, path = parse_document(doc)
        print(f"\n[{dist:.4f}] {first_line}  (id={rid})")
        if path:
            print(f"  section: {path}")
        bits = []
        if meta.get("price") is not None:
            bits.append(f"price={meta['price']}")
        if meta.get("brand"):
            bits.append(f"brand={meta['brand']}")
        if meta.get("handle"):
            bits.append(f"handle={meta['handle']}")
        if bits:
            print(f"  {'  '.join(bits)}")
        shown += 1
        if shown >= args.n:
            break

    if shown == 0:
        print("No results.")


if __name__ == "__main__":
    main()
