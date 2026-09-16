"""Interactive CLI.

    python -m app.cli --user earth

Slash commands:
    /mem              rows stored in Milvus for this user (queried directly, not via the agent)
    /show <path>      full content of one memory, e.g. /show /AGENTS.md
    /search <text>    vector search in Milvus
    /new              start a new thread (= new session, empty short-term state)
    /thread [id]      show current thread id, or switch to (resume) another thread
    /threads          threads of this user saved by the checkpointer
    /history          messages stored in the checkpoint of the current thread
    /reset            delete this user's memories in Milvus (negative-control demo)
    /exit
"""
from __future__ import annotations

import argparse
import json
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import build_agent, make_store, reset_memory
from app.checkpointer import list_threads, open_checkpointer, redact, thread_config

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
    from app.config import settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=settings.default_user_id)
    parser.add_argument("--thread", default=None)
    args = parser.parse_args()

    store = make_store()
    with open_checkpointer() as checkpointer:
        agent = build_agent(args.user, store, checkpointer=checkpointer)
        where = "RAM (lost on exit)" if settings.checkpointer == "memory" else redact(settings.postgres_uri)
        print(f"{BOLD}Deep Agent + Milvus memory PoC{RESET}  user={args.user}")
        print(f"{DIM}long-term memory : Milvus {store.uri} / {store.collection_name}{RESET}")
        print(f"{DIM}checkpointer     : {settings.checkpointer} -> {where}{RESET}")
        thread = args.thread or uuid.uuid4().hex[:8]
        _announce_thread(agent, thread, args.user)
        print(f"{DIM}Type /mem, /show, /search, /new, /thread, /threads, /history, /reset, /exit{RESET}\n")
        _loop(agent, store, checkpointer, thread, args.user)
    store.close()


def _announce_thread(agent, thread: str, user: str) -> None:
    n = len(agent.graph.get_state(thread_config(thread, user)).values.get("messages", []))
    state = f"resuming, {n} message(s) in checkpoint" if n else "new"
    print(f"{GREEN}thread={thread} ({state}){RESET}")


def _loop(agent, store, checkpointer, thread: str, user: str) -> None:
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
        if line.startswith("/thread") and not line.startswith("/threads"):
            parts = line.split()
            if len(parts) > 1:
                thread = parts[1]
                _announce_thread(agent, thread, user)
            else:
                print(f"thread={thread}   (resume later: python -m app.cli --user {user} --thread {thread})")
            continue
        if line == "/threads":
            rows = list_threads(checkpointer, agent.graph, user)
            if not rows:
                print("(no saved threads for this user)")
            for tid, n, text in rows:
                mark = "*" if tid == thread else " "
                print(f" {mark} {tid}  {n:>3} msgs  last: {text!r}")
            continue
        if line == "/history":
            msgs = agent.graph.get_state(thread_config(thread, user)).values.get("messages", [])
            print(f"{GREEN}{len(msgs)} message(s) in checkpoint of thread {thread}{RESET}")
            for m in msgs:
                label = m.type if not getattr(m, "tool_calls", None) else f"ai→{[t['name'] for t in m.tool_calls]}"
                print(f"  {label:<14} {_text(m).replace(chr(10), ' ')[:100]}")
            continue
        if line == "/reset":
            n = reset_memory(store, agent.namespace)
            print(f"{GREEN}deleted {n} row(s) from Milvus and re-seeded AGENTS.md{RESET}")
            continue

        config = thread_config(thread, user)
        before = agent.graph.get_state(config).values.get("messages", [])
        result = agent.graph.invoke({"messages": [{"role": "user", "content": line}]}, config=config)
        print_turn(result["messages"][len(before) :])


if __name__ == "__main__":
    main()
