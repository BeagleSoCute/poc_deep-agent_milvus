"""LangGraph ``BaseStore`` backed by Milvus (Milvus Lite or a Milvus server).

The Milvus blog imports ``langchain_milvus.storage.MilvusStore``, which does not
exist in ``langchain-milvus`` (0.4.0). Deep Agents' ``StoreBackend`` needs a
LangGraph ``BaseStore``, so this module provides one.

One Milvus row == one store item (for Deep Agents: one file under /memories/).

    id          VARCHAR  PK   sha256(namespace + key)   -> put() is an upsert
    namespace   VARCHAR       "poc-agent.earth"          (LangGraph forbids "." in labels)
    key         VARCHAR       "/AGENTS.md"               (CompositeBackend strips "/memories")
    value       JSON          {"content", "encoding", "created_at", "modified_at"}
    text        VARCHAR       text that was embedded
    indexed     BOOL          False when put(..., index=False)
    created_at  INT64         epoch ms
    updated_at  INT64         epoch ms
    vector      FLOAT_VECTOR  embedding of ``text``
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    get_text_at_path,
)
from pymilvus import DataType, MilvusClient

# Milvus Lite runs an in-process gRPC server that logs "Method not implemented" for calls newer
# pymilvus clients probe (e.g. AllocTimestamp). Harmless; keep it out of test/CLI output.
logging.getLogger("grpc._server").setLevel(logging.CRITICAL)

NS_SEP = "."
MAX_QUERY_WINDOW = 16384  # Milvus limit for offset + limit
TEXT_MAX_CHARS = 16000  # VARCHAR max_length is in bytes; Thai is 3 bytes/char
OUTPUT_FIELDS = ["id", "namespace", "key", "value", "created_at", "updated_at"]


# --------------------------------------------------------------------------- helpers
def ns_to_str(namespace: tuple[str, ...]) -> str:
    return NS_SEP.join(namespace)


def str_to_ns(value: str) -> tuple[str, ...]:
    return tuple(value.split(NS_SEP)) if value else ()


def make_id(namespace: tuple[str, ...], key: str) -> str:
    return hashlib.sha256(json.dumps([list(namespace), key]).encode()).hexdigest()


def lit(value: str) -> str:
    """Milvus string literal (double-quoted, escaped)."""
    return json.dumps(value, ensure_ascii=False)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _apply_op(item_value: Any, op: str, target: Any) -> bool:
    try:
        if op == "$eq":
            return item_value == target
        if op == "$ne":
            return item_value != target
        if op == "$gt":
            return item_value > target
        if op == "$gte":
            return item_value >= target
        if op == "$lt":
            return item_value < target
        if op == "$lte":
            return item_value <= target
    except TypeError:
        return False
    raise ValueError(f"Unsupported filter operator: {op}")


def compare_values(item_value: Any, filter_value: Any) -> bool:
    """Same semantics as LangGraph's InMemoryStore filter."""
    if isinstance(filter_value, dict):
        if filter_value and all(k.startswith("$") for k in filter_value):
            return all(_apply_op(item_value, k, v) for k, v in filter_value.items())
        if not isinstance(item_value, dict):
            return False
        return all(compare_values(item_value.get(k), v) for k, v in filter_value.items())
    return item_value == filter_value


def matches_filter(value: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    return not flt or all(compare_values(value.get(k), v) for k, v in flt.items())


def does_match(cond: MatchCondition, namespace: tuple[str, ...]) -> bool:
    path = tuple(cond.path)
    if len(path) > len(namespace):
        return False
    segment = namespace[: len(path)] if cond.match_type == "prefix" else namespace[len(namespace) - len(path) :]
    return all(p == "*" or p == s for p, s in zip(path, segment))


def _in_prefix(namespace: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return namespace[: len(prefix)] == tuple(prefix)


def text_for_embedding(value: dict[str, Any], fields: list[str] | None) -> str:
    if fields:
        parts: list[str] = []
        for field in fields:
            for t in get_text_at_path(value, field):
                parts.append(t if isinstance(t, str) else json.dumps(t, ensure_ascii=False))
        text = "\n".join(parts)
    else:
        content = value.get("content")
        if isinstance(content, list):
            content = "\n".join(map(str, content))
        text = content if isinstance(content, str) else json.dumps(value, ensure_ascii=False)
    return (text or " ")[:TEXT_MAX_CHARS]


# --------------------------------------------------------------------------- store
class MilvusStore(BaseStore):
    """Persistent LangGraph store on Milvus with vector search."""

    supports_ttl = False

    def __init__(
        self,
        *,
        uri: str,
        embeddings: Embeddings,
        collection_name: str = "agent_memories",
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        db_name: str | None = None,
        create_db: bool = False,
        timeout: float | None = 10.0,
        index_fields: list[str] | None = None,
    ) -> None:
        """
        Args:
            uri: Milvus Lite file path (``./data/memory.db``) or server URL
                (``http://host:19530``, ``https://xxx.zillizcloud.com``).
            token: ``"user:password"`` for a Milvus server with auth, or a Zilliz Cloud API key.
            user / password: alternative to ``token``.
            db_name: Milvus database (server only). ``None`` -> ``default``.
            create_db: create ``db_name`` if it does not exist.
        """
        self.uri = uri
        self.db_name = db_name or "default"
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.index_fields = index_fields  # None -> embed value["content"]
        is_server = uri.startswith(("http://", "https://", "tcp://", "grpc://"))
        auth = dict(uri=uri, token=token or "", user=user or "", password=password or "", timeout=timeout)
        use_db = bool(is_server and db_name and db_name != "default")
        try:
            if use_db:
                admin = MilvusClient(**auth)
                existing = admin.list_databases()
                if db_name not in existing:
                    if not create_db:
                        admin.close()
                        raise ValueError(
                            f"Milvus database {db_name!r} not found (have: {existing}). "
                            "Create it or set create_db=True / MILVUS_DB_CREATE=true."
                        )
                    admin.create_database(db_name)
                admin.close()
            self.client = MilvusClient(**auth, db_name=db_name if use_db else "")
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ConnectionError(f"Cannot connect to Milvus at {uri!r} (db={self.db_name}): {e}") from e
        if not is_server:
            self.db_name = "(milvus-lite)"
        self._lock = threading.RLock()
        self.dim = self._ensure_collection()

    # ----------------------------------------------------------------- setup
    def _ensure_collection(self) -> int:
        c = self.client
        if c.has_collection(self.collection_name):
            desc = c.describe_collection(self.collection_name)
            dim = None
            for f in desc.get("fields", []):
                if f.get("name") == "vector":
                    dim = int(f.get("params", {}).get("dim"))
            if dim is None:
                raise RuntimeError(f"Collection {self.collection_name!r} exists but has no 'vector' field")
            probe = len(self.embeddings.embed_query("dimension probe"))
            if probe != dim:
                raise RuntimeError(
                    f"Collection {self.collection_name!r} has dim={dim} but the embedding model returns {probe}. "
                    "Use another MILVUS_COLLECTION or delete the Milvus Lite file."
                )
            c.load_collection(self.collection_name)
            return dim

        dim = len(self.embeddings.embed_query("dimension probe"))
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("namespace", DataType.VARCHAR, max_length=1024)
        schema.add_field("key", DataType.VARCHAR, max_length=2048)
        schema.add_field("value", DataType.JSON)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("indexed", DataType.BOOL)
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("updated_at", DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

        index_params = c.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        c.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",  # read-your-writes: the demo checks rows right after writing
        )
        c.load_collection(self.collection_name)
        return dim

    def drop_collection(self) -> None:
        """Delete the whole collection (tests / cleanup)."""
        with self._lock:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------- raw helpers
    def _query(self, expr: str, limit: int = MAX_QUERY_WINDOW) -> list[dict[str, Any]]:
        return self.client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=OUTPUT_FIELDS,
            limit=limit,
        )

    @staticmethod
    def _prefix_expr(prefix: tuple[str, ...]) -> str:
        if not prefix:
            return 'id != ""'
        p = ns_to_str(prefix)
        # LIKE may over-match ('_' is a wildcard); callers post-filter exactly with _in_prefix().
        return f"(namespace == {lit(p)} or namespace like {lit(p + NS_SEP + '%')})"

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> Item:
        return Item(
            value=row["value"],
            key=row["key"],
            namespace=str_to_ns(row["namespace"]),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    # ----------------------------------------------------------------- ops
    def _get(self, op: GetOp) -> Item | None:
        rows = self._query(f"id == {lit(make_id(op.namespace, op.key))}", limit=1)
        return self._row_to_item(rows[0]) if rows else None

    def _put_many(self, ops: list[PutOp]) -> None:
        # Last write wins inside one batch (same as InMemoryStore).
        final: dict[str, PutOp] = {}
        for op in ops:
            final[make_id(op.namespace, op.key)] = op

        deletes = [i for i, op in final.items() if op.value is None]
        upserts = {i: op for i, op in final.items() if op.value is not None}

        if deletes:
            self.client.delete(collection_name=self.collection_name, ids=deletes)
        if not upserts:
            return

        ids_expr = "id in [" + ", ".join(lit(i) for i in upserts) + "]"
        existing = {r["id"]: r["created_at"] for r in self._query(ids_expr, limit=len(upserts))}

        now = _now_ms()
        texts, rows = [], []
        for pk, op in upserts.items():
            value = dict(op.value)
            indexed = op.index is not False
            fields = op.index if isinstance(op.index, list) else self.index_fields
            text = text_for_embedding(value, fields) if indexed else ""
            texts.append(text or " ")
            rows.append(
                {
                    "id": pk,
                    "namespace": ns_to_str(op.namespace),
                    "key": op.key,
                    "value": value,
                    "text": text,
                    "indexed": indexed,
                    "created_at": existing.get(pk, now),
                    "updated_at": now,
                }
            )
        vectors = self.embeddings.embed_documents(texts)
        for row, vec in zip(rows, vectors):
            row["vector"] = [float(x) for x in vec]
        self.client.upsert(collection_name=self.collection_name, data=rows)

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        expr = self._prefix_expr(prefix)
        window = op.offset + op.limit

        if op.query:
            fetch = MAX_QUERY_WINDOW if op.filter else min(MAX_QUERY_WINDOW, max(window * 4, window))
            vec = self.embeddings.embed_query(op.query)
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[vec],
                anns_field="vector",
                filter=f"{expr} and indexed == true",
                limit=min(fetch, MAX_QUERY_WINDOW),
                output_fields=OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE"},
            )[0]
            scored: list[tuple[dict[str, Any], float | None]] = []
            for hit in hits:
                row = hit["entity"]
                if _in_prefix(str_to_ns(row["namespace"]), prefix) and matches_filter(row["value"], op.filter):
                    scored.append((row, float(hit["distance"])))
        else:
            rows = self._query(expr)
            scored = [
                (r, None)
                for r in rows
                if _in_prefix(str_to_ns(r["namespace"]), prefix) and matches_filter(r["value"], op.filter)
            ]
            scored.sort(key=lambda rs: (rs[0]["namespace"], rs[0]["key"]))

        return [
            SearchItem(
                namespace=str_to_ns(r["namespace"]),
                key=r["key"],
                value=r["value"],
                created_at=_dt(r["created_at"]),
                updated_at=_dt(r["updated_at"]),
                score=score,
            )
            for r, score in scored[op.offset : window]
        ]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = self.client.query(
            collection_name=self.collection_name,
            filter='id != ""',
            output_fields=["namespace"],
            limit=MAX_QUERY_WINDOW,
        )
        namespaces = {str_to_ns(r["namespace"]) for r in rows}
        if op.match_conditions:
            namespaces = {ns for ns in namespaces if all(does_match(c, ns) for c in op.match_conditions)}
        if op.max_depth is not None:
            namespaces = {ns[: op.max_depth] for ns in namespaces}
        return sorted(namespaces)[op.offset : op.offset + op.limit]

    # ----------------------------------------------------------------- BaseStore API
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        results: list[Result] = [None] * len(ops)
        with self._lock:
            puts = [op for op in ops if isinstance(op, PutOp)]
            # Reads in a batch see the state *before* its puts (InMemoryStore behaves the same).
            for i, op in enumerate(ops):
                if isinstance(op, GetOp):
                    results[i] = self._get(op)
                elif isinstance(op, SearchOp):
                    results[i] = self._search(op)
                elif isinstance(op, ListNamespacesOp):
                    results[i] = self._list_namespaces(op)
                elif isinstance(op, PutOp):
                    pass
                else:
                    raise ValueError(f"Unknown operation type: {type(op)}")
            if puts:
                self._put_many(puts)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    # ----------------------------------------------------------------- demo helpers
    def raw_rows(self, namespace_prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        """Rows exactly as stored in Milvus (for inspection / tests)."""
        with self._lock:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=self._prefix_expr(namespace_prefix),
                output_fields=[*OUTPUT_FIELDS, "text", "indexed"],
                limit=MAX_QUERY_WINDOW,
            )
        return sorted(
            (r for r in rows if _in_prefix(str_to_ns(r["namespace"]), namespace_prefix)),
            key=lambda r: (r["namespace"], r["key"]),
        )

    def count(self, namespace_prefix: tuple[str, ...] = ()) -> int:
        return len(self.raw_rows(namespace_prefix))
