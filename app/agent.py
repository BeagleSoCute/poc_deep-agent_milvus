"""Deep Agent whose /memories/ directory is persisted in Milvus."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.embeddings import make_embeddings
from app.milvus_store import MilvusStore
from app.tools import internet_search, make_memory_search_tool

MEMORY_ROUTE = "/memories/"
AGENTS_MD_KEY = "/AGENTS.md"  # CompositeBackend strips "/memories" before it reaches the store

DEFAULT_AGENTS_MD = """# Long-term memory about the user
(The agent keeps this file up to date. It is loaded at the start of every session.)

## User profile & preferences
- (nothing yet)
"""

SYSTEM_PROMPT = """You are a research assistant with long-term memory.

Long-term memory = files under /memories/ (persisted in a Milvus database, shared across sessions).
Everything outside /memories/ is scratch space that disappears when the session ends.

Rules:
1. When the user tells you something durable about themselves (name, role, preferences,
   tools they use), update /memories/AGENTS.md right away with edit_file (or write_file).
   Keep the file short and organised.
2. When you finish researching a topic, save a concise summary to
   /memories/research/<short-topic-slug>.md with write_file.
3. When the user asks about something from an earlier session, first check
   /memories/ (ls, read_file) or use search_memories before answering.
4. Never claim you remembered something unless it is in /memories/.
5. Use scratch files (e.g. /scratch/...) only for temporary notes.
"""


def make_chat_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # No temperature: GPT-5 family models only accept the default.
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openai_base_url or None,
        api_key=settings.openai_api_key or None,
        timeout=120,
        max_retries=2,
    )


def make_store(**overrides: Any) -> MilvusStore:
    kwargs: dict[str, Any] = dict(
        uri=settings.resolved_milvus_uri,
        token=settings.milvus_token or None,
        collection_name=settings.milvus_collection,
        embeddings=make_embeddings(),
    )
    kwargs.update(overrides)
    return MilvusStore(**kwargs)


_USER_ID_RE = re.compile(r"^[A-Za-z0-9_\-@+:~]{1,64}$")


def user_namespace(user_id: str) -> tuple[str, ...]:
    # "." is excluded: LangGraph namespace labels cannot contain periods (and MilvusStore joins with ".").
    if not _USER_ID_RE.match(user_id):
        raise ValueError(
            f"Invalid user id {user_id!r}: use letters, digits, '-', '_', '@', '+', ':' or '~' (max 64 chars)."
        )
    return (settings.agent_namespace, user_id)


def memory_backend(store: MilvusStore, namespace: tuple[str, ...]) -> StoreBackend:
    return StoreBackend(namespace=lambda _rt: namespace, store=store)


def seed_memory(store: MilvusStore, namespace: tuple[str, ...]) -> None:
    """Create /memories/AGENTS.md for a new user (goes through StoreBackend -> Milvus)."""
    if store.get(namespace, AGENTS_MD_KEY) is None:
        memory_backend(store, namespace).write(AGENTS_MD_KEY, DEFAULT_AGENTS_MD)


def reset_memory(store: MilvusStore, namespace: tuple[str, ...]) -> int:
    """Delete every memory of this user, then re-seed an empty AGENTS.md. Returns rows deleted."""
    items = store.search(namespace, limit=10_000)
    for it in items:
        store.delete(it.namespace, it.key)
    seed_memory(store, namespace)
    return len(items)


@dataclass
class MemoryAgent:
    graph: Any
    store: MilvusStore
    namespace: tuple[str, ...]
    backend: CompositeBackend
    memory_backend: StoreBackend


def build_agent(
    user_id: str,
    store: MilvusStore,
    *,
    model: BaseChatModel | None = None,
    tools: list | None = None,
    checkpointer: Any | None = None,
) -> MemoryAgent:
    namespace = user_namespace(user_id)
    seed_memory(store, namespace)

    mem = memory_backend(store, namespace)
    backend = CompositeBackend(default=StateBackend(), routes={MEMORY_ROUTE: mem})

    if tools is None:
        tools = [internet_search, make_memory_search_tool(store, namespace)]

    graph = create_deep_agent(
        model=model or make_chat_model(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        memory=[MEMORY_ROUTE + AGENTS_MD_KEY.lstrip("/")],  # "/memories/AGENTS.md"
        backend=backend,
        store=store,  # the blog passed InMemoryStore() here by mistake
        checkpointer=checkpointer or InMemorySaver(),
        name="milvus-memory-poc",
    )
    return MemoryAgent(graph=graph, store=store, namespace=namespace, backend=backend, memory_backend=mem)
