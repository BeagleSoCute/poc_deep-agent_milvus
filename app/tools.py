"""Agent tools: web search (Tavily or offline mock) and semantic memory search."""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool

from app.config import settings

_MOCK_CORPUS = {
    "Milvus overview": "Milvus is an open-source vector database (LF AI & Data) built for similarity search "
    "over billions of vectors. It separates storage and compute and scales horizontally.",
    "Milvus deployment modes": "Milvus Lite is a Python library storing data in a local file, Milvus Standalone "
    "runs in one Docker container, Milvus Distributed runs on Kubernetes, and Zilliz Cloud is the managed service.",
    "Milvus index types": "Milvus supports FLAT, IVF_FLAT, IVF_PQ, HNSW, DiskANN, SCANN and GPU indexes, with "
    "metrics L2, IP and COSINE for dense vectors and BM25 for sparse full-text search.",
    "Milvus hybrid search": "Milvus supports hybrid search across multiple vector fields (dense + sparse) with "
    "rerankers such as RRF and weighted ranking, plus scalar filtering with boolean expressions.",
    "Milvus data model": "Collections have schemas with scalar fields (VARCHAR, INT64, BOOL, JSON, ARRAY) and "
    "vector fields; partitions and partition keys support multi-tenancy.",
}


def _mock_search(query: str, max_results: int) -> str:
    words = {w for w in query.lower().split() if len(w) > 2}
    ranked = sorted(
        _MOCK_CORPUS.items(),
        key=lambda kv: -sum(w in (kv[0] + " " + kv[1]).lower() for w in words),
    )
    lines = [f"{title}: {body}" for title, body in ranked[:max_results]]
    return "[MOCK SEARCH - no TAVILY_API_KEY set]\n" + "\n".join(lines)


@tool
def internet_search(query: str, max_results: int = 5) -> str:
    """Search the internet and return short text results."""
    if not settings.tavily_api_key:
        return _mock_search(query, max_results)
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(query, max_results=max_results)
    return "\n".join(f"{r['title']}: {r['content']}" for r in results["results"])


def make_memory_search_tool(store, namespace: tuple[str, ...]) -> Callable:
    """Vector search over this user's long-term memories stored in Milvus."""

    @tool
    def search_memories(query: str, limit: int = 3) -> str:
        """Semantic search over long-term memories (files under /memories/). Returns file paths and snippets."""
        hits = store.search(namespace, query=query, limit=limit)
        if not hits:
            return "No memories found."
        out = []
        for h in hits:
            content = h.value.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            score = f"{h.score:.3f}" if h.score is not None else "-"
            out.append(f"/memories{h.key} (score={score})\n{content[:500]}")
        return "\n\n".join(out)

    return search_memories
