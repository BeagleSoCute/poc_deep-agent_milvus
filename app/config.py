"""Settings loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Milvus Lite talks gRPC; hide its "skipping fork() handlers" info logs.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# pymilvus parses MILVUS_URI from the environment when it is imported and only accepts http(s)://.
# This project uses MILVUS_DB_URI instead; drop a stray local-path MILVUS_URI so the import does not fail.
if os.environ.get("MILVUS_URI", "").strip() and not os.environ["MILVUS_URI"].startswith(("http://", "https://")):
    os.environ.pop("MILVUS_URI")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    openai_base_url: str = _env("OPENAI_BASE_URL")
    openai_api_key: str = _env("OPENAI_API_KEY")
    llm_model: str = _env("LLM_MODEL", "gpt-5")

    embedding_provider: str = _env("EMBEDDING_PROVIDER", "openai").lower()
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-3-large")

    milvus_uri: str = _env("MILVUS_DB_URI", "./data/milvus_memory.db")
    milvus_token: str = _env("MILVUS_DB_TOKEN")  # "user:password" or Zilliz Cloud API key
    milvus_user: str = _env("MILVUS_DB_USER")
    milvus_password: str = _env("MILVUS_DB_PASSWORD")
    milvus_db_name: str = _env("MILVUS_DB_NAME")  # server only; empty -> "default"
    milvus_db_create: bool = _env("MILVUS_DB_CREATE", "false").lower() in ("1", "true", "yes")
    milvus_collection: str = _env("MILVUS_COLLECTION", "agent_memories")

    agent_namespace: str = _env("AGENT_NAMESPACE", "poc-agent")
    default_user_id: str = _env("DEFAULT_USER_ID", "earth")

    tavily_api_key: str = _env("TAVILY_API_KEY")

    # Short-term memory (thread state): "memory" (RAM, lost on exit) or "postgres"
    checkpointer: str = _env("CHECKPOINTER", "memory").lower()
    postgres_uri: str = _env("POSTGRES_URI", "postgresql://agent:agent@localhost:5432/agent_checkpoints")

    @property
    def is_milvus_server(self) -> bool:
        return self.milvus_uri.startswith(("http://", "https://", "tcp://", "grpc://"))

    def milvus_client_kwargs(self) -> dict:
        """kwargs for pymilvus.MilvusClient (used by scripts that bypass MilvusStore)."""
        kw = {
            "uri": self.resolved_milvus_uri,
            "token": self.milvus_token,
            "user": self.milvus_user,
            "password": self.milvus_password,
        }
        if self.is_milvus_server and self.milvus_db_name:
            kw["db_name"] = self.milvus_db_name
        return kw

    @property
    def resolved_milvus_uri(self) -> str:
        """Relative Milvus Lite paths are resolved against the project root."""
        uri = self.milvus_uri
        if uri.startswith(("http://", "https://", "tcp://", "unix:")):
            return uri
        path = Path(uri)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


settings = Settings()
