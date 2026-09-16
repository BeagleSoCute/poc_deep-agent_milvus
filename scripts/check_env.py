"""Smoke-test the environment: LLM, embeddings, Milvus.

    python -m scripts.check_env
"""
from __future__ import annotations

import sys
import time

from app.config import settings


def step(name, fn) -> bool:
    t = time.time()
    try:
        detail = fn()
        print(f"[OK]   {name} ({time.time() - t:.1f}s) {detail or ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def check_models():
    from openai import OpenAI

    client = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
    ids = sorted(m.id for m in client.models.list().data)
    return f"\n       models: {', '.join(ids[:40])}{' …' if len(ids) > 40 else ''}"


def check_llm():
    from app.agent import make_chat_model

    reply = make_chat_model().invoke("Reply with the single word: pong")
    return f"model={settings.llm_model} reply={reply.content!r}"


def check_embeddings():
    from app.embeddings import make_embeddings

    vec = make_embeddings().embed_query("hello milvus")
    return f"provider={settings.embedding_provider} model={settings.embedding_model} dim={len(vec)}"


def check_milvus():
    from pymilvus import MilvusClient

    uri = settings.resolved_milvus_uri
    if not settings.is_milvus_server:
        client = MilvusClient(uri=uri)
        cols = client.list_collections()
        client.close()
        return f"mode=milvus-lite uri={uri} collections={cols}"

    kw = settings.milvus_client_kwargs()
    kw.pop("db_name", None)
    client = MilvusClient(**kw, timeout=10)
    info = f"mode=server uri={uri} version={client.get_server_version()} databases={client.list_databases()}"
    db = settings.milvus_db_name or "default"
    if db != "default" and db not in client.list_databases():
        client.close()
        hint = "set MILVUS_DB_CREATE=true to create it" if not settings.milvus_db_create else "will be created on first use"
        return f"{info}\n       database {db!r} does not exist yet ({hint})"
    client.close()
    client = MilvusClient(**settings.milvus_client_kwargs(), timeout=10)
    cols = client.list_collections()
    client.close()
    return f"{info}\n       db={db} collections={cols}"


def check_checkpointer():
    from app.checkpointer import open_checkpointer, redact

    with open_checkpointer() as cp:
        if settings.checkpointer == "memory":
            return "checkpointer=memory (RAM only; set CHECKPOINTER=postgres to persist threads)"
        with cp.conn.connection() as conn:
            version = conn.execute("select version()").fetchone()["version"].split(",")[0]
            n = conn.execute("select count(distinct thread_id) as n from checkpoints").fetchone()["n"]
        return f"checkpointer=postgres {redact(settings.postgres_uri)}  {version}  threads={n}"


if __name__ == "__main__":
    print(f"OPENAI_BASE_URL={settings.openai_base_url}  LLM_MODEL={settings.llm_model}")
    results = [
        step("list models on endpoint (optional)", check_models),
        step("chat completion", check_llm),
        step("embeddings", check_embeddings),
        step("milvus connection", check_milvus),
        step("checkpointer", check_checkpointer),
    ]
    if not results[2]:
        print("\nTip: if the endpoint has no embedding model, set EMBEDDING_PROVIDER=hash in .env "
              "(or EMBEDDING_MODEL to one listed above).")
    sys.exit(0 if all(results[1:]) else 1)
