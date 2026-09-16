"""Checkpointer tests (short-term memory per thread).

Postgres tests need a database:

    docker compose up -d postgres
    POSTGRES_TEST_URI=postgresql://agent:agent@localhost:5432/agent_checkpoints python -m pytest tests/test_postgres_checkpointer.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import uuid

import pytest
from langchain_core.messages import AIMessage

from app.agent import build_agent
from app.checkpointer import list_threads, open_checkpointer, thread_config
from tests.test_agent_memory import ScriptedModel, call

PG_URI = os.environ.get("POSTGRES_TEST_URI", "").strip()
needs_pg = pytest.mark.skipif(not PG_URI, reason="set POSTGRES_TEST_URI to run Postgres checkpointer tests")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def say(*texts):
    """Scripted model that answers each turn with the given texts."""
    return ScriptedModel(messages=iter([AIMessage(content=t) for t in texts]), disable_streaming=True)


def messages(agent, thread, user):
    return agent.graph.get_state(thread_config(thread, user)).values.get("messages", [])


def ask(agent, thread, user, text):
    agent.graph.invoke({"messages": [{"role": "user", "content": text}]}, config=thread_config(thread, user))


# --------------------------------------------------------------------------- baseline
def test_inmemory_checkpointer_forgets_thread_in_new_instance(store):
    thread = f"t-{uuid.uuid4().hex[:6]}"
    with open_checkpointer("memory") as cp1:
        a1 = build_agent("earth", store, model=say("hi"), tools=[], checkpointer=cp1)
        ask(a1, thread, "earth", "hello")
        assert len(messages(a1, thread, "earth")) == 2
    with open_checkpointer("memory") as cp2:  # "restart"
        a2 = build_agent("earth", store, model=say("x"), tools=[], checkpointer=cp2)
        assert messages(a2, thread, "earth") == []


# --------------------------------------------------------------------------- postgres
@needs_pg
def test_postgres_thread_survives_new_connection_pool(store):
    thread, user = f"t-{uuid.uuid4().hex[:6]}", "earth"
    with open_checkpointer("postgres", PG_URI) as cp1:
        a1 = build_agent(user, store, model=say("Nice to meet you, Earth."), tools=[], checkpointer=cp1)
        ask(a1, thread, user, "my name is Earth")
    with open_checkpointer("postgres", PG_URI) as cp2:  # brand-new pool = simulated restart
        a2 = build_agent(user, store, model=say("You said your name is Earth."), tools=[], checkpointer=cp2)
        msgs = messages(a2, thread, user)
        assert [m.content for m in msgs] == ["my name is Earth", "Nice to meet you, Earth."]
        ask(a2, thread, user, "what did I say?")
        assert len(messages(a2, thread, user)) == 4  # conversation continued in the same thread
        cp2.delete_thread(thread)
        assert messages(a2, thread, user) == []


@needs_pg
def test_postgres_thread_survives_process_exit(tmp_path, milvus_uri, milvus_kwargs):
    """Strongest proof: a child *process* writes the thread and exits; the parent reads it back."""
    thread, user = f"t-{uuid.uuid4().hex[:6]}", "earth"
    collection = f"mem_test_{uuid.uuid4().hex[:8]}"
    child = textwrap.dedent(f"""
        from langchain_core.messages import AIMessage
        from app.agent import build_agent
        from app.checkpointer import open_checkpointer, thread_config
        from app.embeddings import HashEmbeddings
        from app.milvus_store import MilvusStore
        from tests.test_agent_memory import ScriptedModel
        store = MilvusStore(uri={milvus_uri!r}, embeddings=HashEmbeddings(), collection_name={collection!r},
                            **{milvus_kwargs!r})
        model = ScriptedModel(messages=iter([AIMessage(content="stored in postgres")]), disable_streaming=True)
        with open_checkpointer("postgres", {PG_URI!r}) as cp:
            agent = build_agent({user!r}, store, model=model, tools=[], checkpointer=cp)
            agent.graph.invoke({{"messages": [{{"role": "user", "content": "remember this turn"}}]}},
                               config=thread_config({thread!r}, {user!r}))
        store.close()
    """)
    subprocess.run([sys.executable, "-c", child], cwd=ROOT, check=True, timeout=120)

    from app.embeddings import HashEmbeddings
    from app.milvus_store import MilvusStore

    store = MilvusStore(uri=milvus_uri, embeddings=HashEmbeddings(), collection_name=collection, **milvus_kwargs)
    try:
        with open_checkpointer("postgres", PG_URI) as cp:
            agent = build_agent(user, store, model=say("-"), tools=[], checkpointer=cp)
            assert [m.content for m in messages(agent, thread, user)] == ["remember this turn", "stored in postgres"]
            cp.delete_thread(thread)
    finally:
        store.drop_collection()
        store.close()


@needs_pg
def test_threads_are_isolated_but_milvus_memory_is_shared(store):
    user = "earth"
    t1, t2 = f"t-{uuid.uuid4().hex[:6]}", f"t-{uuid.uuid4().hex[:6]}"
    with open_checkpointer("postgres", PG_URI) as cp:
        writer = build_agent(
            user, store, tools=[], checkpointer=cp,
            model=ScriptedModel(messages=iter([
                call("write_file", 1, file_path="/memories/fact.md", content="likes Milvus"),
                AIMessage(content="saved"),
            ]), disable_streaming=True),
        )
        ask(writer, t1, user, "remember I like Milvus")

        reader = build_agent(
            user, store, tools=[], checkpointer=cp,
            model=ScriptedModel(messages=iter([
                call("read_file", 1, file_path="/memories/fact.md"),
                AIMessage(content="you like Milvus"),
            ]), disable_streaming=True),
        )
        ask(reader, t2, user, "what do I like?")

        t2_msgs = messages(reader, t2, user)
        assert all("remember I like Milvus" not in str(m.content) for m in t2_msgs)   # thread state separate
        assert any("likes Milvus" in str(m.content) for m in t2_msgs if m.type == "tool")  # Milvus shared
        cp.delete_thread(t1)
        cp.delete_thread(t2)


@needs_pg
def test_list_threads_by_user(store):
    mine, other = f"t-{uuid.uuid4().hex[:6]}", f"t-{uuid.uuid4().hex[:6]}"
    user = f"u{uuid.uuid4().hex[:6]}"
    with open_checkpointer("postgres", PG_URI) as cp:
        a = build_agent(user, store, model=say("ok"), tools=[], checkpointer=cp)
        ask(a, mine, user, "hello from me")
        b = build_agent("someone-else", store, model=say("ok"), tools=[], checkpointer=cp)
        ask(b, other, "someone-else", "hello from other")

        threads = list_threads(cp, a.graph, user)
        assert [t[0] for t in threads] == [mine]
        assert threads[0][1] == 2 and threads[0][2] == "hello from me"
        cp.delete_thread(mine)
        cp.delete_thread(other)
