"""One question = one new process + one new thread. Used by the demo to prove
that anything remembered must have come from Milvus.

    python -m scripts.ask --user earth "What is my name?"
    python -m scripts.ask --user earth --reset          # wipe this user's memories
"""
from __future__ import annotations

import argparse
import uuid

from app.agent import build_agent, make_store, reset_memory
from app.cli import print_rows, print_turn
from app.config import settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--user", default=settings.default_user_id)
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()

    store = make_store()
    agent = build_agent(a.user, store)
    if a.reset:
        print(f"deleted {reset_memory(store, agent.namespace)} row(s)")
    if a.question:
        thread = uuid.uuid4().hex[:8]
        print(f"[new process, thread={thread}] you> {a.question}")
        result = agent.graph.invoke(
            {"messages": [{"role": "user", "content": a.question}]},
            config={"configurable": {"thread_id": thread}},
        )
        print_turn(result["messages"][1:])
    print_rows(store, agent.namespace)
    store.close()


if __name__ == "__main__":
    main()
