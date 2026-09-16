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

    client = MilvusClient(uri=settings.resolved_milvus_uri, token=settings.milvus_token or "")
    cols = client.list_collections()
    client.close()
    return f"uri={settings.resolved_milvus_uri} collections={cols}"


if __name__ == "__main__":
    print(f"OPENAI_BASE_URL={settings.openai_base_url}  LLM_MODEL={settings.llm_model}")
    results = [
        step("list models on endpoint (optional)", check_models),
        step("chat completion", check_llm),
        step("embeddings", check_embeddings),
        step("milvus connection", check_milvus),
    ]
    if not results[2]:
        print("\nTip: if the endpoint has no embedding model, set EMBEDDING_PROVIDER=hash in .env "
              "(or EMBEDDING_MODEL to one listed above).")
    sys.exit(0 if all(results[1:]) else 1)
