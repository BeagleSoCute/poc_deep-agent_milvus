"""One question = one new process + one new thread. Used by the demo to prove
that anything remembered must have come from Milvus.

    python -m scripts.ask --user earth "What is my name?"
    python -m scripts.ask --user earth --reset          # wipe this user's memories
"""
from __future__ import annotations

import argparse
import uuid

from app.agent import build_agent, make_store, reset_memory
from app.checkpointer import open_checkpointer, thread_config
from app.cli import print_rows, print_turn
from app.config import settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--user", default=settings.default_user_id)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--thread", help="continue this thread (needs CHECKPOINTER=postgres to survive between runs)")
    a = p.parse_args()

    store = make_store()
    with open_checkpointer() as cp:
        agent = build_agent(a.user, store, checkpointer=cp)
        if a.reset:
            print(f"deleted {reset_memory(store, agent.namespace)} row(s)")
        if a.question:
            thread = a.thread or uuid.uuid4().hex[:8]
            config = thread_config(thread, a.user)
            before = agent.graph.get_state(config).values.get("messages", [])
            print(f"[new process, checkpointer={settings.checkpointer}, thread={thread}, "
                  f"{len(before)} msg(s) already in thread] you> {a.question}")
            result = agent.graph.invoke({"messages": [{"role": "user", "content": a.question}]}, config=config)
            print_turn(result["messages"][len(before) + 1 :])
        print_rows(store, agent.namespace)
    store.close()


if __name__ == "__main__":
    main()
