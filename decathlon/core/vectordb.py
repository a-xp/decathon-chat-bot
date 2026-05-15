"""Embedded Chroma vector store (no server).

We pass embeddings explicitly (from decathlon.core.embeddings), so collections
are created without an embedding function.
"""

import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

PRODUCTS = "products"
CATEGORIES = "categories"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client: chromadb.ClientAPI, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
