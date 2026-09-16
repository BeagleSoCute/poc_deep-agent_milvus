"""Live end-to-end test with the real LLM endpoint.

    RUN_LIVE=1 pytest -m live -s
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.agent import build_agent, reset_memory
from app.embeddings import make_embeddings
from app.milvus_store import MilvusStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real LLM"),
]

SECRET = f"Zebra-{uuid.uuid4().hex[:6]}"


def ask(store, question: str) -> str:
    agent = build_agent("live-user", store)  # new agent + new checkpointer each time
    res = agent.graph.invoke({"messages": [{"role": "user", "content": question}]},
                             config={"configurable": {"thread_id": uuid.uuid4().hex}})
    content = res["messages"][-1].content
    return content if isinstance(content, str) else str(content)


def test_live_remember_recall_and_forget(milvus_uri, milvus_kwargs):
    store = MilvusStore(uri=milvus_uri, embeddings=make_embeddings(),
                        collection_name=f"live_test_{uuid.uuid4().hex[:6]}", **milvus_kwargs)
    ns = ("poc-agent", "live-user")
    try:
        before = store.count(ns)
        ask(store, f"Please remember permanently: my project codename is {SECRET}.")
        rows = store.raw_rows(ns)
        print("\nrows after session A (Milvus):")
        for r in rows:
            hit = [ln for ln in r["value"]["content"].splitlines() if SECRET in ln]
            print(f"  /memories{r['key']}  id={r['id'][:10]}…  line with codename: {hit}")
        assert any(SECRET in r["value"]["content"] for r in rows), "codename was not written to Milvus"
        assert len(rows) >= max(before, 1)

        answer = ask(store, "What is my project codename?")
        print("session B:", answer)
        assert SECRET in answer

        reset_memory(store, ns)
        answer = ask(store, "What is my project codename?")
        print("session C (after wipe):", answer)
        assert SECRET not in answer
    finally:
        store.drop_collection()
        store.close()
