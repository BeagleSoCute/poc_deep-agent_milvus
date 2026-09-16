"""T1–T5 + persistence: the adapter really reads/writes Milvus."""
from __future__ import annotations

from langgraph.store.base import MatchCondition
from langgraph.store.memory import InMemoryStore

from tests.conftest import raw_query

NS = ("poc-agent", "u1")


def file_value(content: str) -> dict:
    return {"content": content, "encoding": "utf-8",
            "created_at": "2026-01-01T00:00:00+00:00", "modified_at": "2026-01-01T00:00:00+00:00"}


def test_t1_put_creates_a_row_in_milvus(store):
    assert store.count() == 0
    store.put(NS, "/prefs.md", file_value("prefers Python"))

    rows = raw_query(store, 'namespace == "poc-agent.u1" and key == "/prefs.md"')
    assert len(rows) == 1
    assert rows[0]["value"]["content"] == "prefers Python"
    assert len(rows[0]["vector"]) == store.dim


def test_t2_put_same_key_is_upsert(store):
    store.put(NS, "/prefs.md", file_value("v1"))
    first = store.get(NS, "/prefs.md")
    store.put(NS, "/prefs.md", file_value("v2"))

    rows = raw_query(store, 'key == "/prefs.md"')
    assert len(rows) == 1
    assert rows[0]["value"]["content"] == "v2"
    second = store.get(NS, "/prefs.md")
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_t3_get_delete_search_list(store):
    store.put(NS, "/a.md", file_value("alpha"))
    store.put(NS, "/b.md", file_value("beta"))
    store.put(("poc-agent", "u2"), "/c.md", file_value("gamma"))

    assert store.get(NS, "/a.md").value["content"] == "alpha"
    assert store.get(NS, "/missing.md") is None
    assert [i.key for i in store.search(NS)] == ["/a.md", "/b.md"]
    assert len(store.search(("poc-agent",))) == 3
    assert [i.key for i in store.search(NS, limit=1, offset=1)] == ["/b.md"]
    assert [i.key for i in store.search(NS, filter={"content": "beta"})] == ["/b.md"]
    assert store.list_namespaces() == [("poc-agent", "u1"), ("poc-agent", "u2")]
    assert store.list_namespaces(max_depth=1) == [("poc-agent",)]
    assert store.list_namespaces(suffix=("u2",)) == [("poc-agent", "u2")]

    store.delete(NS, "/a.md")
    assert store.get(NS, "/a.md") is None
    assert raw_query(store, 'key == "/a.md"') == []


def test_t3_differential_against_inmemorystore(store):
    """Same operations on MilvusStore and LangGraph's InMemoryStore give the same answers."""
    ref = InMemoryStore()
    script = [
        ("put", ("poc-agent", "u1"), "/AGENTS.md", file_value("name: Earth")),
        ("put", ("poc-agent", "u1"), "/research/milvus.md", file_value("HNSW, IVF")),
        ("put", ("poc-agent", "u2"), "/AGENTS.md", file_value("name: Joe")),
        ("put", ("poc-agent", "u1"), "/AGENTS.md", file_value("name: Earth, likes Python")),
        ("put", ("other", "x"), "/k", {"n": 5, "tag": "a"}),
        ("put", ("other", "x"), "/j", {"n": 9, "tag": "b"}),
        ("delete", ("poc-agent", "u1"), "/research/milvus.md", None),
    ]
    for op, ns, key, value in script:
        for s in (store, ref):
            s.put(ns, key, value) if op == "put" else s.delete(ns, key)

    def snapshot(s):
        return {
            "get": [(i.value if (i := s.get(ns, key)) else None) for _, ns, key, _ in script],
            "search_all": sorted((i.namespace, i.key, str(i.value)) for i in s.search((), limit=100)),
            "search_u1": sorted((i.key, str(i.value)) for i in s.search(("poc-agent", "u1"), limit=100)),
            "filter_gt": sorted(i.key for i in s.search(("other",), filter={"n": {"$gt": 6}})),
            "filter_eq": sorted(i.key for i in s.search(("other",), filter={"tag": "a"})),
            "ns": sorted(s.list_namespaces()),
            "ns_prefix": sorted(s.list_namespaces(prefix=("poc-agent", "*"))),
        }

    assert snapshot(store) == snapshot(ref)


def test_t4_semantic_search_ranks_relevant_memory_first(store):
    store.put(NS, "/research/milvus.md", file_value("Milvus index types: HNSW, IVF_FLAT, DiskANN"))
    store.put(NS, "/AGENTS.md", file_value("User prefers short answers in Thai"))
    store.put(NS, "/lunch.md", file_value("Lunch today was pad kra pao with rice"))

    hits = store.search(NS, query="which index types does milvus support", limit=3)
    assert hits[0].key == "/research/milvus.md"
    assert hits[0].score is not None and hits[0].score >= hits[-1].score


def test_t5_namespaces_are_isolated(store):
    store.put(("poc-agent", "alice"), "/AGENTS.md", file_value("alice secret"))
    store.put(("poc-agent", "bob"), "/AGENTS.md", file_value("bob stuff"))
    store.put(("poc-agent", "alice_2"), "/AGENTS.md", file_value("lookalike namespace"))

    assert [i.value["content"] for i in store.search(("poc-agent", "bob"))] == ["bob stuff"]
    assert [i.value["content"] for i in store.search(("poc-agent", "bob"), query="secret")] == ["bob stuff"]
    assert store.get(("poc-agent", "bob"), "/AGENTS.md").value["content"] == "bob stuff"
    assert len(store.search(("poc-agent", "alice"))) == 1  # "alice_2" must not leak in via LIKE


def test_index_false_is_stored_but_not_vector_searchable(store):
    store.put(NS, "/raw.json", {"content": "milvus"}, index=False)
    assert store.get(NS, "/raw.json") is not None
    assert store.search(NS, query="milvus") == []


def test_data_survives_reopening_the_database(make_test_store):
    s1 = make_test_store()
    s1.put(NS, "/AGENTS.md", file_value("persisted"))
    s1.close()

    s2 = make_test_store()  # new client on the same Milvus Lite file + collection
    assert s2.get(NS, "/AGENTS.md").value["content"] == "persisted"


def test_async_api(store):
    import asyncio

    async def go():
        await store.aput(NS, "/async.md", file_value("async ok"))
        return await store.aget(NS, "/async.md")

    assert asyncio.run(go()).value["content"] == "async ok"
