#!/usr/bin/env bash
# Checkpointer demo: two separate processes continue the SAME thread via Postgres.
#   docker compose up -d postgres
#   CHECKPOINTER=postgres ./scripts/demo_resume.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export CHECKPOINTER="${CHECKPOINTER:-postgres}"
THREAD="resume-$(date +%H%M%S)"
USER_ID="${1:-earth}"
echo "checkpointer=$CHECKPOINTER thread=$THREAD user=$USER_ID"

echo; echo "---- process 1: say something only to this thread (not a durable fact)"
python -m scripts.ask --user "$USER_ID" --thread "$THREAD" \
  "For this conversation only (do NOT save it to /memories/): the temporary code word is PINEAPPLE-42. Just acknowledge."

echo; echo "---- process 2: same thread -> answer comes from the Postgres checkpoint"
python -m scripts.ask --user "$USER_ID" --thread "$THREAD" "What is the temporary code word I gave you?"

echo; echo "---- process 3: NEW thread -> should not know (it was never written to Milvus)"
python -m scripts.ask --user "$USER_ID" "What is the temporary code word I gave you?"
