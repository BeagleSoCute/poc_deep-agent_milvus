"""Drop-in long-term memory for ``create_deep_agent`` backed by a Milvus URL.

Usage in any project::

    from deepagents import create_deep_agent
    from langchain_openai import OpenAIEmbeddings
    from app.milvus_memory import create_milvus_memory

    mem = create_milvus_memory(
        uri="http://milvus.internal:19530",       # or ./memory.db (Milvus Lite)
        token="root:Milvus",                      # or user=/password=
        db_name="agents",
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        namespace=("my-agent", "user-123"),       # fixed per agent instance
    )
    agent = create_deep_agent(model=..., tools=[...], system_prompt=..., **mem.agent_kwargs())

Multi-user service (one agent, user taken from the invoke context)::

    mem = create_milvus_memory(..., namespace=lambda rt: ("my-agent", rt.context["user_id"]))
    agent = create_deep_agent(..., context_schema=Ctx, **mem.agent_kwargs())
    agent.invoke({"messages": [...]}, config=..., context={"user_id": "user-123"})

Every file the agent writes under ``/memories/`` becomes one row in the Milvus collection;
everything else stays in the thread state (``StateBackend``).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.embeddings import Embeddings

from app.milvus_store import MilvusStore

Namespace = tuple[str, ...]
NamespaceArg = Namespace | Callable[[Any], Namespace]

DEFAULT_ROUTE = "/memories/"
AGENTS_MD = "AGENTS.md"
DEFAULT_AGENTS_MD = """# Long-term memory about the user
(The agent keeps this file up to date. It is loaded at the start of every session.)

## User profile & preferences
- (nothing yet)
"""


@dataclass
class MilvusMemory:
    store: MilvusStore
    backend: CompositeBackend
    memory_backend: StoreBackend
    namespace: NamespaceArg
    route: str = DEFAULT_ROUTE
    memory_files: list[str] = field(default_factory=list)

    def agent_kwargs(self) -> dict[str, Any]:
        """``backend``, ``store`` and ``memory`` for ``create_deep_agent(**...)``."""
        return {"backend": self.backend, "store": self.store, "memory": list(self.memory_files)}

    def backend_for(self, namespace: Namespace) -> StoreBackend:
        return StoreBackend(namespace=lambda _rt: namespace, store=self.store)

    def seed(self, namespace: Namespace, content: str = DEFAULT_AGENTS_MD) -> bool:
        """Create AGENTS.md for this namespace if missing. Returns True if created."""
        key = "/" + AGENTS_MD
        if self.store.get(namespace, key) is not None:
            return False
        self.backend_for(namespace).write(key, content)
        return True

    def reset(self, namespace: Namespace, reseed: bool = True) -> int:
        """Delete all memories in ``namespace``. Returns the number of rows deleted."""
        items = self.store.search(namespace, limit=10_000)
        for it in items:
            self.store.delete(it.namespace, it.key)
        if reseed:
            self.seed(namespace)
        return len(items)

    def close(self) -> None:
        self.store.close()


def create_milvus_memory(
    *,
    uri: str,
    embeddings: Embeddings,
    namespace: NamespaceArg,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    create_db: bool = False,
    collection_name: str = "agent_memories",
    route: str = DEFAULT_ROUTE,
    default_backend: Any | None = None,
    seed_agents_md: bool = True,
    store: MilvusStore | None = None,
) -> MilvusMemory:
    """Connect to Milvus and build the backend/store pair for ``create_deep_agent``."""
    if not route.startswith("/") or not route.endswith("/"):
        raise ValueError("route must look like '/memories/'")

    store = store or MilvusStore(
        uri=uri,
        token=token,
        user=user,
        password=password,
        db_name=db_name,
        create_db=create_db,
        embeddings=embeddings,
        collection_name=collection_name,
    )
    ns_factory = namespace if callable(namespace) else (lambda _rt, _ns=tuple(namespace): _ns)
    memory_backend = StoreBackend(namespace=ns_factory, store=store)
    backend = CompositeBackend(default=default_backend or StateBackend(), routes={route: memory_backend})

    mem = MilvusMemory(
        store=store,
        backend=backend,
        memory_backend=memory_backend,
        namespace=namespace,
        route=route,
        memory_files=[route + AGENTS_MD],
    )
    # With a callable namespace the user is only known at invoke time; a missing AGENTS.md is skipped
    # by Deep Agents' MemoryMiddleware and the agent creates it on first write.
    if seed_agents_md and not callable(namespace):
        mem.seed(tuple(namespace))
    return mem
