"""Look inside Milvus directly with pymilvus — no agent involved.

    python -m scripts.inspect_memory                 # all rows, all users
    python -m scripts.inspect_memory --user earth    # one user
    python -m scripts.inspect_memory --user earth --full
    python -m scripts.inspect_memory --user earth --query "milvus index types"
    python -m scripts.inspect_memory --stats

Milvus Lite keeps a lock on its .db file: run this while the chat CLI is NOT running
(inside the CLI use /mem instead).
"""
from __future__ import annotations

import argparse

from app.config import settings  # noqa: I001  (must load before pymilvus, see app/config.py)
from pymilvus import MilvusClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user")
    p.add_argument("--full", action="store_true", help="print full content")
    p.add_argument("--query", help="vector search (uses the configured embedding model)")
    p.add_argument("--stats", action="store_true", help="collection schema + row count")
    a = p.parse_args()

    uri, col = settings.resolved_milvus_uri, settings.milvus_collection
    client = MilvusClient(uri=uri, token=settings.milvus_token or "")
    print(f"uri={uri}\ncollection={col}  exists={client.has_collection(col)}")
    if not client.has_collection(col):
        return
    client.load_collection(col)

    ns = f"{settings.agent_namespace}.{a.user}" if a.user else None
    expr = f'namespace == "{ns}"' if ns else 'id != ""'

    if a.stats:
        desc = client.describe_collection(col)
        for f in desc["fields"]:
            print(f"  field {f['name']:<11} {f['type']!s:<24} {f.get('params', '')}")
        print("  indexes:", client.list_indexes(col))
        print("  stats:", client.get_collection_stats(col))

    if a.query:
        from app.embeddings import make_embeddings

        vec = make_embeddings().embed_query(a.query)
        hits = client.search(col, data=[vec], anns_field="vector", filter=f"({expr}) and indexed == true",
                             limit=5, output_fields=["namespace", "key"],
                             search_params={"metric_type": "COSINE"})[0]
        print(f"\nvector search: {a.query!r}")
        for h in hits:
            print(f"  score={h['distance']:.3f}  {h['entity']['namespace']}  /memories{h['entity']['key']}")
        return

    rows = client.query(col, filter=expr, limit=16384,
                        output_fields=["id", "namespace", "key", "value", "created_at", "updated_at", "vector"])
    rows.sort(key=lambda r: (r["namespace"], r["key"]))
    print(f"\n{len(rows)} row(s){' for ' + ns if ns else ''}\n")
    for r in rows:
        content = r["value"].get("content", "")
        if isinstance(content, list):
            content = "\n".join(content)
        vec = list(r["vector"])
        print(f"- {r['namespace']}  /memories{r['key']}")
        print(f"  id={r['id']}  created_at={r['created_at']}  updated_at={r['updated_at']}")
        print(f"  vector dim={len(vec)} head={[round(float(x), 4) for x in vec[:4]]}")
        body = content if a.full else content.replace("\n", " ")[:200]
        print(f"  content: {body}\n")
    client.close()


if __name__ == "__main__":
    main()
