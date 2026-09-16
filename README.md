# Deep Agent + Milvus Long-Term Memory (PoC)

PoC จาก Milvus blog *How to Build Production-Ready AI Agents with Deep Agents and Milvus*
(หัวข้อ Hands-on) พร้อม test ที่พิสูจน์ได้ว่า memory ถูกเก็บลง Milvus จริง

- Agent: `deepagents` (`create_deep_agent`) + LLM `gpt-5` ผ่าน LiteLLM (OpenAI-compatible)
- Memory: ไฟล์ใต้ `/memories/` → `StoreBackend` → **`MilvusStore`** (adapter ที่เขียนเอง) → Milvus Lite (`data/milvus_memory.db`)
- ไฟล์อื่น (เช่น `/scratch/...`) → `StateBackend` หายเมื่อจบ session
- Search tool: Tavily ถ้ามี key, ไม่มีก็ใช้ mock (ข้อมูล Milvus สำเร็จรูป)
- UI: CLI (+ Attu ถ้าเปิด Milvus standalone ด้วย `docker-compose.yml`)

## ต่างจากโค้ดใน blog ยังไง

| blog | PoC นี้ |
|---|---|
| `from langchain_milvus.storage import MilvusStore` (ไม่มีใน langchain-milvus 0.4.0) | `app/milvus_store.py` — LangGraph `BaseStore` บน pymilvus |
| `StoreBackend(store=InMemoryStore())` → memory **ไม่ได้ลง Milvus** | `StoreBackend(namespace=..., store=milvus_store)` + test T11 กันพลาด |
| ไม่ import `create_deep_agent`, ไม่ส่ง `store=` | ครบ |
| ไม่มี `namespace=` | `("poc-agent", <user_id>)` แยก memory รายผู้ใช้ |

## Setup (macOS)

```bash
cd ~/Desktop/deep-agent_milvus
uv venv -p 3.12 && source .venv/bin/activate      # หรือ python3.12 -m venv .venv
uv pip install -r requirements.txt                 # หรือ pip install -r requirements.txt
cp .env.example .env   # (มี .env ให้แล้ว) แก้ค่าได้ตามต้องการ
python -m scripts.check_env
```

`check_env` จะลอง: list models, chat `gpt-5`, embeddings, Milvus
ถ้า endpoint ไม่มี embedding model → ตั้ง `EMBEDDING_PROVIDER=hash` ใน `.env`
(ยังพิสูจน์การเก็บลง Milvus ได้ แต่ semantic search จะเป็นแค่ระดับคำ)

> เปลี่ยน embedding model ภายหลัง = dimension เปลี่ยน → ลบ `data/milvus_memory.db` หรือเปลี่ยน `MILVUS_COLLECTION`

## ใช้งาน

```bash
python -m app.cli --user earth
```

คำสั่งใน CLI: `/mem` (ดู row ใน Milvus ตรงๆ), `/show /AGENTS.md`, `/search <text>`,
`/new` (thread ใหม่), `/reset` (ลบ memory ของ user นี้), `/exit`
tool call ที่แตะ `/memories` จะเป็นสีเหลือง

ดู Milvus ตรงๆ (ปิด CLI ก่อน — Milvus Lite lock ไฟล์):

```bash
python -m scripts.inspect_memory --user earth [--full] [--stats] [--query "..."]
```

## Checkpointer (short-term memory) บน Postgres

| | Checkpointer | Store (Milvus) |
|---|---|---|
| เก็บ | state ทั้ง thread (messages, tool calls, ไฟล์นอก `/memories/`, todos) | ไฟล์ใต้ `/memories/` |
| key | `thread_id` | namespace + path |
| ข้าม thread | ไม่ | ได้ |

ค่าเริ่มต้น `CHECKPOINTER=memory` (RAM หายเมื่อปิดโปรแกรม) เปลี่ยนเป็น Postgres:

```bash
docker compose up -d postgres                  # (ถ้า 5432 ชน: POSTGRES_HOST_PORT=5433 ... แล้วแก้ POSTGRES_URI)
# .env
CHECKPOINTER=postgres
POSTGRES_URI=postgresql://agent:agent@localhost:5432/agent_checkpoints

python -m scripts.check_env                    # checkpointer=postgres ... threads=N
python -m app.cli --user earth                 # /thread, /threads, /history
python -m app.cli --user earth --thread <id>   # กลับมาคุยต่อ thread เดิมหลังปิดโปรแกรม
./scripts/demo_resume.sh                       # 2 process คุยต่อ thread เดียวกัน, thread ใหม่จำไม่ได้
```

Tests (ต้องมี Postgres):

```bash
POSTGRES_TEST_URI=postgresql://agent:agent@localhost:5432/agent_checkpoints python -m pytest tests/test_postgres_checkpointer.py -v
```

- thread อยู่รอดหลังเปิด connection pool ใหม่ และหลัง **process ลูกจบไปแล้ว**
- thread แยกกัน แต่ memory ใน Milvus ใช้ร่วมกัน
- `list_threads` กรองตาม user, `delete_thread` ลบได้
- baseline: `InMemorySaver` ลืม thread เมื่อสร้าง instance ใหม่

ใช้ในโค้ดของทีม: `with open_checkpointer() as cp: create_deep_agent(..., checkpointer=cp, **mem.agent_kwargs())`
(แอป async/FastAPI ให้ใช้ `AsyncPostgresSaver`) — checkpoint เก็บทุก step ข้อมูลโตเร็ว ควรมีงานลบ thread เก่า

## ต่อ Milvus ผ่าน URL เข้ากับ `create_deep_agent`

ใช้ `app/milvus_memory.py` → `create_milvus_memory()` คืน `backend` + `store` + `memory` ให้ส่งเข้า `create_deep_agent` ได้เลย

```python
from app.milvus_memory import create_milvus_memory

mem = create_milvus_memory(
    uri="http://milvus.internal:19530",      # Milvus server / https://...zillizcloud.com / ./x.db (Lite)
    token="root:Milvus",                     # หรือ user=..., password=...
    db_name="agents",                        # optional, create_db=True เพื่อสร้างให้
    collection_name="agent_memories",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-large", ...),
    namespace=("my-agent", "user-123"),      # หรือ lambda rt: ("my-agent", rt.context["user_id"])
)
agent = create_deep_agent(model=..., tools=[...], system_prompt=..., **mem.agent_kwargs())
```

- ไฟล์ใต้ `/memories/` → 1 row ใน Milvus, ไฟล์อื่นอยู่ใน state ของ thread
- namespace แบบคงที่ = 1 agent ต่อ 1 user; แบบ `lambda rt:` = agent เดียวหลาย user (ส่ง `context={"user_id": ...}` ตอน invoke + `context_schema`)
- ตัวอย่างเต็ม: `examples/team_agent.py`

### เปลี่ยนทั้งโปรเจกต์ไปใช้ server — แก้แค่ `.env`

```
MILVUS_DB_URI=http://<host>:19530
MILVUS_DB_TOKEN=root:Milvus        # ถ้าเปิด auth
MILVUS_DB_NAME=agents              # ถ้าไม่ใช้ default
MILVUS_DB_CREATE=true              # ให้สร้าง database ถ้ายังไม่มี
```

```bash
python -m scripts.check_env                       # จะแสดง mode=server, version, databases, collections
python -m examples.team_agent                     # remember → recall → แสดง row ใน Milvus
MILVUS_TEST_URI=http://<host>:19530 MILVUS_TEST_TOKEN=root:Milvus python -m pytest   # test ทั้งชุดกับ server
```

test บน server สร้าง collection `mem_test_*` ชั่วคราวแล้ว drop ทิ้งเอง
ถ้าเปลี่ยน embedding model (dimension เปลี่ยน) ต้องใช้ `MILVUS_COLLECTION` ใหม่

## Tests

```bash
python -m pytest                    # T1–T11: ไม่ใช้ network (Milvus Lite ชั่วคราว + fake model)
RUN_LIVE=1 python -m pytest -m live -s   # end-to-end กับ LLM จริง
```

> ใช้ `python -m pytest` เสมอ — `pytest` เฉยๆ อาจไปเรียก pytest ของ Python ระบบแทน venv


| ID | ไฟล์ | พิสูจน์อะไร |
|---|---|---|
| T1 | test_milvus_store | put → query Milvus ด้วย pymilvus เจอ row + vector |
| T2 | 〃 | put key เดิม = upsert (1 row), created_at คงเดิม |
| T3 | 〃 | get/delete/search/filter/list_namespaces + **differential test เทียบ InMemoryStore** |
| T4 | 〃 | vector search จัดอันดับ memory ที่เกี่ยวข้องขึ้นก่อน |
| T5 | 〃 | namespace ของแต่ละ user ไม่รั่วหากัน |
| persist | 〃 | ปิด client เปิดใหม่ ข้อมูลยังอยู่ |
| T6 | test_agent_memory | agent `write_file /memories/...` → มี row ใน Milvus |
| T7 | 〃 | agent ใหม่ + client ใหม่ + thread ใหม่ อ่าน memory ได้ |
| T8 | 〃 | **negative control**: ลบ row แล้วอ่านไม่เจอ |
| T10 | 〃 | `/scratch/...` ไม่ลง Milvus |
| T11 | 〃 | backend ใช้ `MilvusStore` จริง ไม่ใช่ `InMemoryStore` |
| live | test_live_llm | LLM จำ codename → session ใหม่ตอบได้ → wipe แล้วตอบไม่ได้ |

## Demo ให้ทีม (~10 นาที)

```bash
./scripts/demo.sh            # แต่ละขั้นเป็น process ใหม่ + thread ใหม่
```

1. reset → 2. บอกข้อมูลตัวเอง → 3. ให้ research Milvus → 4. `inspect_memory` เห็น row/vector ใน Milvus
→ 5. process ใหม่ถาม "ผมชื่ออะไร/เคย research อะไร" → ตอบได้ → 6. vector search ตรงจาก Milvus
→ 7. wipe แล้วถามซ้ำ → จำไม่ได้

อยากโชว์ใน Attu: `docker compose up -d`, ตั้ง `MILVUS_DB_URI=http://localhost:19530`, เปิด http://localhost:8000
แล้ว `docker compose restart milvus-standalone` เพื่อโชว์ว่าข้อมูลยังอยู่หลัง restart

## โครงสร้าง

```
app/config.py         settings จาก .env
app/embeddings.py     OpenAI-compatible หรือ hash embeddings
app/milvus_store.py   MilvusStore(BaseStore)  ← หัวใจ
app/milvus_memory.py  create_milvus_memory() ← ใช้ต่อกับ create_deep_agent
examples/team_agent.py ตัวอย่างต่อ Milvus URL
app/tools.py          internet_search (Tavily/mock), search_memories (vector search)
app/agent.py          build_agent(): CompositeBackend + StoreBackend + create_deep_agent
app/checkpointer.py   open_checkpointer(): memory | postgres
app/cli.py            chat CLI
scripts/              check_env, inspect_memory, ask (1 คำถาม = 1 process), demo.sh
tests/                T1–T11 + live
```

## Collection `agent_memories`

1 row = 1 ไฟล์ใต้ `/memories/`
`id` (sha256 ของ namespace+key) · `namespace` (`poc-agent.earth`) · `key` (`/AGENTS.md` — CompositeBackend ตัด `/memories` ออก)
· `value` JSON (`content`, `encoding`, `created_at`, `modified_at`) · `text` · `indexed` · `created_at`/`updated_at` (ms) · `vector` (COSINE, AUTOINDEX)

## ข้อจำกัดที่รู้อยู่

- ยังไม่ได้รันจริงตอนเขียน (sandbox ไม่มี PyPI) — เขียนตาม source ของ deepagents 0.7.13 / langgraph store API
- Milvus Lite: เปิดได้ทีละ process, ไม่รองรับ Attu, ไม่รองรับ Windows (ใช้ server URL แทนได้)
- ทั้งไฟล์ = 1 vector; `value` JSON จำกัด ~64KB ต่อไฟล์
- ไม่รองรับ TTL
