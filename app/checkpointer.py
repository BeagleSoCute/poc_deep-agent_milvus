"""Checkpointer factory = short-term memory (full graph state per thread_id).

    with open_checkpointer() as cp:            # CHECKPOINTER=memory | postgres
        agent = build_agent(user, store, checkpointer=cp)

Postgres keeps every thread across restarts / multiple API instances.
Milvus (the store) keeps /memories/ across threads. They are independent.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings


@contextmanager
def open_checkpointer(kind: str | None = None, postgres_uri: str | None = None) -> Iterator[Any]:
    kind = (kind or settings.checkpointer).lower()
    if kind == "memory":
        yield InMemorySaver()
        return
    if kind != "postgres":
        raise ValueError(f"Unknown CHECKPOINTER={kind!r} (use 'memory' or 'postgres')")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"{e}. CHECKPOINTER=postgres needs extra packages; inside the venv run:\n"
            "    uv pip install -r requirements.txt\n"
            "(or: uv pip install langgraph-checkpoint-postgres 'psycopg[binary,pool]')"
        ) from e

    uri = postgres_uri or settings.postgres_uri
    # autocommit + dict_row are required by PostgresSaver (same as PostgresSaver.from_conn_string)
    pool = ConnectionPool(
        conninfo=uri,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    try:
        pool.open(wait=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        pool.close()
        raise ConnectionError(
            f"Cannot connect to Postgres at {redact(uri)}: {e}\n"
            "Start it with `docker compose up -d postgres` or fix POSTGRES_URI."
        ) from e
    try:
        saver = PostgresSaver(pool)
        saver.setup()  # creates/migrates checkpoint tables (idempotent)
        yield saver
    finally:
        pool.close()


def redact(uri: str) -> str:
    """postgresql://user:***@host/db"""
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def thread_config(thread_id: str, user_id: str) -> dict:
    """Invoke config. user_id is stored in checkpoint metadata so threads can be listed per user."""
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}, "metadata": {"user_id": user_id}}


def list_threads(checkpointer: Any, graph: Any, user_id: str, limit: int = 20) -> list[tuple[str, int, str]]:
    """[(thread_id, message_count, last_user_message)] newest first.

    checkpointer.list() finds the user's thread ids (metadata filter). The thread state is then read
    with graph.get_state(): a raw checkpoint does not necessarily hold the full message list
    (newer LangGraph versions rebuild channel values from several checkpoints/writes).
    """
    thread_ids: list[str] = []
    for tup in checkpointer.list(None, filter={"user_id": user_id}, limit=500):
        tid = tup.config["configurable"]["thread_id"]
        if tid not in thread_ids:
            thread_ids.append(tid)
        if len(thread_ids) >= limit:
            break

    out = []
    for tid in thread_ids:
        msgs = graph.get_state(thread_config(tid, user_id)).values.get("messages", []) or []
        last_user = next((m for m in reversed(msgs) if getattr(m, "type", "") == "human"), None)
        text = str(getattr(last_user, "content", ""))[:60] if last_user else ""
        out.append((tid, len(msgs), text))
    return out
