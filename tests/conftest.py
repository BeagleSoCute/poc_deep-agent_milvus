"""Test fixtures.

By default every test uses a throw-away Milvus Lite file. To run the same tests against a
Milvus server (e.g. the team's URL), set:

    MILVUS_TEST_URI=http://host:19530  [MILVUS_TEST_TOKEN=user:pass]  [MILVUS_TEST_DB=name]

Each test creates its own random collection there and drops it afterwards.
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.embeddings import HashEmbeddings
from app.milvus_store import MilvusStore


SERVER_URI = os.environ.get("MILVUS_TEST_URI", "").strip()
SERVER_KW = {
    "token": os.environ.get("MILVUS_TEST_TOKEN") or None,
    "db_name": os.environ.get("MILVUS_TEST_DB") or None,
    "create_db": True,
}


@pytest.fixture
def milvus_uri(tmp_path):
    """Server URL if MILVUS_TEST_URI is set, otherwise a temp Milvus Lite file."""
    return SERVER_URI or str(tmp_path / "milvus_test.db")


@pytest.fixture
def milvus_kwargs():
    return dict(SERVER_KW) if SERVER_URI else {}


@pytest.fixture
def db_path(milvus_uri):  # backwards-compatible name
    return milvus_uri


@pytest.fixture
def make_test_store(milvus_uri, milvus_kwargs):
    """Factory: open a MilvusStore on a fresh collection (hash embeddings, no LLM calls)."""
    opened: list[MilvusStore] = []
    collection = f"mem_test_{uuid.uuid4().hex[:8]}"

    def _make() -> MilvusStore:
        s = MilvusStore(uri=milvus_uri, embeddings=HashEmbeddings(), collection_name=collection, **milvus_kwargs)
        opened.append(s)
        return s

    yield _make
    if opened:
        cleanup = _make()
        cleanup.drop_collection()
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
