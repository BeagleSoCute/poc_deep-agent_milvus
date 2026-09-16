"""Interactive CLI.

    python -m app.cli --user earth

Slash commands:
    /mem              rows stored in Milvus for this user (queried directly, not via the agent)
    /show <path>      full content of one memory, e.g. /show /AGENTS.md
    /search <text>    vector search in Milvus
    /new              start a new thread (= new session, empty short-term state)
    /reset            delete this user's memories in Milvus (negative-control demo)
    /exit
"""
from __future__ import annotations

import argparse
import json
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import build_agent, make_store, reset_memory

DIM, BOLD, CYAN, GREEN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


def print_rows(store, namespace) -> None:
    rows = store.raw_rows(namespace)
    print(f"{GREEN}Milvus collection '{store.collection_name}' @ {store.uri}{RESET}")
    print(f"{GREEN}{len(rows)} row(s) in namespace {'.'.join(namespace)}{RESET}")
    for r in rows:
        content = r["value"].get("content", "")
        if isinstance(content, list):
            content = "\n".join(content)
        preview = content.replace("\n", " ")[:90]
        print(f"  {BOLD}/memories{r['key']}{RESET}  id={r['id'][:10]}…  updated={r['updated_at']}  {DIM}{preview}{RESET}")


def print_turn(new_messages) -> None:
    for m in new_messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                args = json.dumps(tc.get("args", {}), ensure_ascii=False)
                color = YELLOW if "/memories" in args or tc["name"] == "search_memories" else CYAN
                print(f"{color}  → tool {tc['name']}({args[:160]}){RESET}")
            text = _text(m)
            if text and not m.tool_calls:
                print(f"\n{BOLD}agent>{RESET} {text}\n")
        elif isinstance(m, ToolMessage):
            print(f"{DIM}  ← {m.name}: {_text(m)[:160]!r}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser()
    from app.config import settings

    parser.add_argument("--user", default=settings.default_user_id)
    parser.add_argument("--thread", default=None)
    args = parser.parse_args()

    store = make_store()
    agent = build_agent(args.user, store)
    thread = args.thread or uuid.uuid4().hex[:8]
    print(f"{BOLD}Deep Agent + Milvus memory PoC{RESET}  user={args.user}  thread={thread}")
    print(f"{DIM}Type /mem, /show, /search, /new, /reset, /exit{RESET}\n")

    while True:
        try:
            line = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/exit":
            break
        if line == "/mem":
            print_rows(store, agent.namespace)
            continue
        if line.startswith("/show"):
            key = line.split(maxsplit=1)[1] if " " in line else "/AGENTS.md"
            key = key.removeprefix("/memories")
            item = store.get(agent.namespace, key)
            print(item.value.get("content") if item else f"(no row for key {key})")
            continue
        if line.startswith("/search "):
            for h in store.search(agent.namespace, query=line[8:], limit=5):
                print(f"  {h.score:.3f}  /memories{h.key}")
            continue
        if line == "/new":
            thread = uuid.uuid4().hex[:8]
            print(f"{GREEN}new thread {thread} (short-term state is empty; only Milvus remembers){RESET}")
            continue
        if line == "/reset":
            n = reset_memory(store, agent.namespace)
            print(f"{GREEN}deleted {n} row(s) from Milvus and re-seeded AGENTS.md{RESET}")
            continue

        config = {"configurable": {"thread_id": thread}}
        before = agent.graph.get_state(config).values.get("messages", [])
        result = agent.graph.invoke({"messages": [{"role": "user", "content": line}]}, config=config)
        print_turn(result["messages"][len(before) :])

    store.close()


if __name__ == "__main__":
    main()
