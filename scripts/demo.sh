#!/usr/bin/env bash
# End-to-end demo: every step is a separate Python process with a fresh thread.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_ID="${1:-demo-$(date +%H%M%S)}"
if [[ ! "$USER_ID" =~ ^[A-Za-z0-9_@+:~-]{1,64}$ ]]; then
  echo "Invalid user id '$USER_ID' (tip: zsh passes '# comments' as arguments). Usage: ./scripts/demo.sh [user-id]" >&2
  exit 1
fi
echo "Demo user: $USER_ID"
ask() { python -m scripts.ask --user "$USER_ID" "$@"; }
pause() { echo; read -r -p "---- $1 [enter] " _; echo; }

pause "1) Start clean: Milvus has only the seeded AGENTS.md for $USER_ID"
ask --reset

pause "2) Session A: tell the agent about yourself"
ask "Hi, I'm Earth. My team builds AI agents in Python and I prefer short answers in Thai. Please remember this."

pause "3) Session B: research something (should write /memories/research/...)"
ask "Research the index types Milvus supports and save a short summary for later."

pause "4) Look inside Milvus directly (no agent)"
python -m scripts.inspect_memory --user "$USER_ID"

pause "5) Session C (new process, new thread): does it remember?"
ask "What's my name, which language does my team use, and what did you research for me before?"

pause "6) Vector search straight from Milvus"
python -m scripts.inspect_memory --user "$USER_ID" --query "vector index types"

pause "7) Negative control: wipe Milvus rows, ask again -> it should NOT know"
ask --reset "What's my name and what did you research for me before?"
