from __future__ import annotations

import uuid

import pytest

from app.embeddings import HashEmbeddings
from app.milvus_store import MilvusStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "milvus_test.db")


@pytest.fixture
def make_test_store(db_path):
    """Factory: open a MilvusStore on the temp Milvus Lite file (hash embeddings, no network)."""
    opened: list[MilvusStore] = []
    collection = f"mem_test_{uuid.uuid4().hex[:8]}"

    def _make() -> MilvusStore:
        s = MilvusStore(uri=db_path, embeddings=HashEmbeddings(), collection_name=collection)
        opened.append(s)
        return s

    yield _make
    for s in opened:
        s.close()


@pytest.fixture
def store(make_test_store):
    return make_test_store()


def raw_query(store: MilvusStore, expr: str) -> list[dict]:
    """Query Milvus with pymilvus directly, bypassing the adapter's read path."""
    return store.client.query(
        collection_name=store.collection_name,
        filter=expr,
        output_fields=["id", "namespace", "key", "value", "vector"],
        limit=1000,
    )
