"""Minimal example: plug a Milvus URL into create_deep_agent's memory.

    MILVUS_DB_URI=http://<host>:19530 MILVUS_DB_TOKEN=root:Milvus python -m examples.team_agent
"""
from __future__ import annotations

import uuid
from typing import TypedDict

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import make_chat_model  # any BaseChatModel works
from app.config import settings
from app.embeddings import make_embeddings
from app.milvus_memory import create_milvus_memory


class Ctx(TypedDict):
    user_id: str


def main() -> None:
    mem = create_milvus_memory(
        uri=settings.resolved_milvus_uri,         # http://host:19530 | https://...zillizcloud.com | ./x.db
        token=settings.milvus_token or None,      # "user:password" or API key
        user=settings.milvus_user or None,
        password=settings.milvus_password or None,
        db_name=settings.milvus_db_name or None,
        create_db=settings.milvus_db_create,
        collection_name=settings.milvus_collection,
        embeddings=make_embeddings(),
        namespace=lambda rt: ("team-agent", rt.context["user_id"]),  # per-request user
    )

    agent = create_deep_agent(
        model=make_chat_model(),
        tools=[],
        system_prompt=(
            "Files under /memories/ are long-term memory stored in Milvus. "
            "Save durable facts about the user to /memories/AGENTS.md and read it when asked."
        ),
        context_schema=Ctx,
        checkpointer=InMemorySaver(),
        **mem.agent_kwargs(),  # -> backend=CompositeBackend(/memories/ -> Milvus), store=MilvusStore, memory=[...]
    )

    user = "demo-user"
    for question in ["Remember: my favourite database is Milvus.", "What is my favourite database?"]:
        res = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": uuid.uuid4().hex}},  # new thread each time
            context={"user_id": user},
        )
        print(f"you> {question}\nagent> {res['messages'][-1].content}\n")

    rows = mem.store.raw_rows(("team-agent", user))
    print(f"{len(rows)} row(s) in Milvus {mem.store.uri} db={mem.store.db_name} "
          f"collection={mem.store.collection_name}:")
    for r in rows:
        print(f"  /memories{r['key']}: {r['value']['content'][:100]!r}")
    mem.close()


if __name__ == "__main__":
    main()
