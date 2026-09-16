"""create_milvus_memory(): the drop-in helper the team plugs into create_deep_agent."""
from __future__ import annotations

from typing import TypedDict

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.embeddings import HashEmbeddings
from app.milvus_memory import create_milvus_memory
from app.milvus_store import MilvusStore
from tests.conftest import raw_query
from tests.test_agent_memory import ScriptedModel, call


def scripted(*steps):
    return ScriptedModel(messages=iter([*steps, AIMessage(content="done")]), disable_streaming=True)


def test_agent_kwargs_wire_milvus_into_create_deep_agent(store):
    mem = create_milvus_memory(uri=store.uri, embeddings=HashEmbeddings(), namespace=("team-agent", "u1"),
                               store=store)
    kw = mem.agent_kwargs()
    assert set(kw) == {"backend", "store", "memory"}
    assert kw["store"] is store and isinstance(kw["store"], MilvusStore)
    assert kw["memory"] == ["/memories/AGENTS.md"]
    assert len(raw_query(store, 'namespace == "team-agent.u1" and key == "/AGENTS.md"')) == 1  # seeded

    agent = create_deep_agent(
        model=scripted(call("write_file", 1, file_path="/memories/notes.md", content="from helper")),
        tools=[], checkpointer=InMemorySaver(), **kw,
    )
    agent.invoke({"messages": [{"role": "user", "content": "save"}]}, config={"configurable": {"thread_id": "h"}})
    rows = raw_query(store, 'namespace == "team-agent.u1" and key == "/notes.md"')
    assert rows and rows[0]["value"]["content"] == "from helper"


def test_connects_by_uri(milvus_uri, milvus_kwargs):
    """The helper opens its own connection from a URI (Lite path or server URL)."""
    import uuid

    mem = create_milvus_memory(uri=milvus_uri, embeddings=HashEmbeddings(), namespace=("team-agent", "u9"),
                               collection_name=f"mem_uri_{uuid.uuid4().hex[:6]}", **milvus_kwargs)
    try:
        assert mem.store.get(("team-agent", "u9"), "/AGENTS.md") is not None
        assert mem.reset(("team-agent", "u9")) == 1
    finally:
        mem.store.drop_collection()
        mem.close()


class Ctx(TypedDict):
    user_id: str


def test_per_request_user_namespace_from_context(store):
    """One agent serving many users: namespace comes from invoke(context=...)."""
    mem = create_milvus_memory(
        uri=store.uri, embeddings=HashEmbeddings(), store=store,
        namespace=lambda rt: ("team-agent", rt.context["user_id"]),
    )
    model = scripted(
        call("write_file", 1, file_path="/memories/AGENTS.md", content="alice likes Go"),
    )
    agent = create_deep_agent(model=model, tools=[], context_schema=Ctx, checkpointer=InMemorySaver(),
                              **mem.agent_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "remember"}]},
                 config={"configurable": {"thread_id": "a"}}, context={"user_id": "alice"})

    alice = raw_query(store, 'namespace == "team-agent.alice"')
    assert [r["value"]["content"] for r in alice] == ["alice likes Go"]
    assert raw_query(store, 'namespace == "team-agent.bob"') == []
