"""T6–T11 with a scripted fake model: deterministic, no network.

The fake model emits exactly the tool calls a real model would, so these tests check
the wiring (Deep Agents -> CompositeBackend -> StoreBackend -> MilvusStore -> Milvus),
not the LLM's judgement. tests/test_live_llm.py covers the real model.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import AGENTS_MD_KEY, build_agent, reset_memory
from app.milvus_store import MilvusStore
from tests.conftest import raw_query


class ScriptedModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts bind_tools (Deep Agents requires tool calling)."""

    def bind_tools(self, tools: Any, **kwargs: Any):  # noqa: ARG002
        return self


def call(name: str, i: int, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}])


def run(agent, text: str, thread: str) -> list:
    result = agent.graph.invoke({"messages": [{"role": "user", "content": text}]},
                                config={"configurable": {"thread_id": thread}})
    return result["messages"]


def tool_outputs(messages, name: str) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == name]


def agent_with_script(store, *steps, user="earth"):
    # disable_streaming: GenericFakeChatModel's _stream drops tool_calls
    model = ScriptedModel(messages=iter([*steps, AIMessage(content="done")]), disable_streaming=True)
    return build_agent(user, store, model=model, tools=[])


def test_t11_agent_uses_milvus_store_not_inmemory(store):
    agent = agent_with_script(store)
    assert isinstance(agent.store, MilvusStore)
    assert agent.memory_backend._get_store() is store  # the bug in the blog: InMemoryStore here
    # seeding AGENTS.md already went through StoreBackend into Milvus
    assert len(raw_query(store, f'key == "{AGENTS_MD_KEY}"')) == 1


def test_t6_write_under_memories_lands_in_milvus(store):
    agent = agent_with_script(
        store, call("write_file", 1, file_path="/memories/user_prefs.md", content="Earth likes Python")
    )
    before = store.count(agent.namespace)
    run(agent, "remember that I like Python", thread="t6")

    rows = raw_query(store, 'namespace == "poc-agent.earth" and key == "/user_prefs.md"')
    assert len(rows) == 1, "file written under /memories/ must be a row in Milvus"
    assert "Earth likes Python" in rows[0]["value"]["content"]
    assert store.count(agent.namespace) == before + 1


def test_t10_scratch_files_do_not_reach_milvus(store):
    agent = agent_with_script(store, call("write_file", 1, file_path="/scratch/notes.md", content="temp only"))
    before = store.count()
    msgs = run(agent, "take a scratch note", thread="t10")

    assert store.count() == before
    assert raw_query(store, 'key like "%notes.md"') == []
    assert any("notes.md" in o for o in tool_outputs(msgs, "write_file"))


def test_t7_new_agent_new_thread_reads_memory_from_milvus(make_test_store):
    s1 = make_test_store()
    writer = agent_with_script(
        s1, call("write_file", 1, file_path="/memories/user_prefs.md", content="favorite language: Python")
    )
    run(writer, "remember my favorite language", thread="session-A")
    s1.close()  # "process" A ends

    s2 = make_test_store()  # fresh client, fresh agent, fresh checkpointer, new thread
    reader = agent_with_script(s2, call("read_file", 1, file_path="/memories/user_prefs.md"))
    msgs = run(reader, "what is my favorite language?", thread="session-B")

    assert any("favorite language: Python" in o for o in tool_outputs(msgs, "read_file"))


def test_t8_negative_control_after_deleting_rows(store):
    writer = agent_with_script(
        store, call("write_file", 1, file_path="/memories/user_prefs.md", content="favorite language: Python")
    )
    run(writer, "remember", thread="a")
    assert reset_memory(store, writer.namespace) >= 2  # AGENTS.md + user_prefs.md

    reader = agent_with_script(store, call("read_file", 1, file_path="/memories/user_prefs.md"))
    msgs = run(reader, "what is my favorite language?", thread="b")
    outputs = tool_outputs(msgs, "read_file")
    assert outputs and all("favorite language: Python" not in o for o in outputs)


def test_edit_file_updates_the_same_row(store):
    agent = agent_with_script(
        store,
        call("edit_file", 1, file_path="/memories/AGENTS.md", old_string="- (nothing yet)",
             new_string="- Name: Earth"),
    )
    run(agent, "my name is Earth", thread="e")
    rows = raw_query(store, f'namespace == "poc-agent.earth" and key == "{AGENTS_MD_KEY}"')
    assert len(rows) == 1
    assert "- Name: Earth" in rows[0]["value"]["content"]
