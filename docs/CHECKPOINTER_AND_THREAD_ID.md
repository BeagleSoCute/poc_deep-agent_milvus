# Checkpointer & `thread_id` — สรุปสำหรับทีม

> Short-term memory ของ agent = **state ของบทสนทนาหนึ่ง (thread)**
> ในโปรเจกต์นี้เลือกเก็บได้ 2 แบบ: **RAM** (`InMemorySaver`) หรือ **Postgres** (`PostgresSaver`)

---

## 1. TL;DR

| คำถาม | คำตอบ |
|---|---|
| checkpointer คืออะไร | ตัวที่ LangGraph ใช้ **บันทึก state ทั้งก้อนของ graph** หลังจบแต่ละ step แยกตาม `thread_id` |
| `thread_id` คืออะไร | กุญแจของ "บทสนทนา" — ส่งมากับทุก `invoke()` ใน `config["configurable"]` |
| ใช้ `langgraph-checkpoint-postgres` ไหม | **ใช้** — `PostgresSaver` (ใน `app/checkpointer.py`) เมื่อ `CHECKPOINTER=postgres` |
| ใช้ทำอะไร | เก็บ checkpoint ลงตาราง Postgres (สร้างเองด้วย `setup()`), โหลดกลับตอน invoke ครั้งถัดไป, list/ลบ thread |
| ผลลัพธ์สุดท้าย | ปิดโปรแกรม / เปิด process ใหม่ / deploy ใหม่ แล้ว **คุยต่อ thread เดิมได้** — thread ใหม่เริ่มว่าง แต่ยังเห็น memory ระยะยาวใน Milvus |
| ต่างจาก Milvus ยังไง | Checkpointer = ความจำ **ใน** thread · Milvus store = ความจำ **ข้าม** thread (`/memories/`) |

---

## 2. สองชั้นของความจำ

```mermaid
flowchart LR
    U["User"] -->|"invoke(messages, thread_id, user_id)"| G["Deep Agent graph<br/>create_deep_agent"]

    subgraph ST["Short-term: Checkpointer (ต่อ thread_id)"]
        direction TB
        CP{"CHECKPOINTER"}
        RAM["InMemorySaver<br/>RAM - หายเมื่อปิด process"]
        PG[("Postgres<br/>PostgresSaver")]
        CP -->|memory| RAM
        CP -->|postgres| PG
    end

    subgraph LT["Long-term: Store (ต่อ namespace)"]
        MS[("Milvus<br/>MilvusStore")]
    end

    G <-->|"messages, tool calls,<br/>ไฟล์นอก /memories/, todos"| CP
    G <-->|"ไฟล์ใต้ /memories/"| MS
```

| | Checkpointer | Store (Milvus) |
|---|---|---|
| เก็บอะไร | state ทั้ง thread: `messages`, tool calls/results, ไฟล์ของ `StateBackend` (เช่น `/scratch/...`), todos, `memory_contents` | ไฟล์ใต้ `/memories/` |
| key | `thread_id` (+ `checkpoint_ns`, `checkpoint_id`) | `namespace` + path |
| อายุ | ต่อ thread | ข้ามทุก thread ของ user |
| เปลี่ยน thread แล้ว | เริ่มว่าง | ยังอยู่ |
| backend | RAM หรือ Postgres | Milvus Lite / Milvus server |

---

## 3. `thread_id` ทำงานยังไง

```mermaid
sequenceDiagram
    autonumber
    participant App as CLI / ask.py
    participant G as LangGraph graph
    participant CP as Checkpointer
    participant LLM as LLM (gpt-5)

    App->>G: invoke({messages:[user msg]}, config{thread_id="abc", user_id="earth"})
    G->>CP: get_tuple(thread_id="abc")
    CP-->>G: checkpoint ล่าสุด (หรือว่างถ้า thread ใหม่)
    Note over G: รวม messages เดิม + ข้อความใหม่
    loop ทุก super-step (model → tools → model ...)
        G->>LLM: messages ทั้งหมดของ thread
        LLM-->>G: คำตอบ / tool_calls
        G->>CP: put(checkpoint ใหม่) + put_writes(...)
    end
    G-->>App: state ล่าสุด (messages ทั้ง thread)
```

- `thread_id` เดิม → ต่อบทสนทนาเดิม
- `thread_id` ใหม่ → state ว่าง (แต่ AGENTS.md ใน Milvus ยังถูกโหลดเข้า prompt)
- LangGraph จะไม่รู้ว่า "user" คือใคร — thread เป็นแค่ string เราจึงแนบ `user_id` ไว้ใน **metadata** ของ checkpoint เพื่อ list thread ต่อ user ได้

---

## 4. สิ่งที่เปลี่ยนในโค้ด (changes รอบนี้)

### 4.1 ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | เปลี่ยนอะไร |
|---|---|
| `requirements.txt` | + `langgraph-checkpoint-postgres>=3.0`, `psycopg[binary,pool]>=3.2` |
| `docker-compose.yml` | + service `postgres` (postgres:16, user/pass/db = `agent`/`agent`/`agent_checkpoints`, volume `./volumes/postgres`, healthcheck `pg_isready`) และ port ปรับได้ด้วย `POSTGRES_HOST_PORT` |
| `.env` / `.env.example` | + `CHECKPOINTER=memory\|postgres`, `POSTGRES_URI=postgresql://agent:agent@localhost:5432/agent_checkpoints` |
| `app/config.py` | + `settings.checkpointer`, `settings.postgres_uri` |
| **`app/checkpointer.py`** (ใหม่) | `open_checkpointer()`, `thread_config()`, `list_threads()`, `redact()` |
| `app/cli.py` | ใช้ `open_checkpointer()`, แสดง checkpointer ตอนเริ่ม, `--thread` เพื่อ resume, คำสั่ง `/thread [id]`, `/threads`, `/history` |
| `scripts/ask.py` | ใช้ `open_checkpointer()` + `--thread` |
| `scripts/check_env.py` | + `check_checkpointer()` แสดง Postgres version และจำนวน thread |
| `scripts/demo_resume.sh` (ใหม่) | demo 3 process: บอก code word → process ใหม่ thread เดิมตอบได้ → thread ใหม่ตอบไม่ได้ |
| `tests/test_postgres_checkpointer.py` (ใหม่) | 5 tests (ดูข้อ 6) |

### 4.2 `app/checkpointer.py` — ทำอะไรบ้าง

```python
with open_checkpointer() as cp:                 # อ่าน CHECKPOINTER จาก .env
    agent = build_agent("earth", store, checkpointer=cp)
    agent.graph.invoke(..., config=thread_config("abc", "earth"))
```

| ฟังก์ชัน | หน้าที่ |
|---|---|
| `open_checkpointer(kind, postgres_uri)` | context manager: `memory` → `InMemorySaver()` · `postgres` → สร้าง `psycopg_pool.ConnectionPool` → `PostgresSaver(pool)` → `saver.setup()` → yield → ปิด pool ตอนจบ |
| `thread_config(thread_id, user_id)` | `{"configurable": {"thread_id", "user_id"}, "metadata": {"user_id"}}` — ใช้ทุกครั้งที่ invoke / get_state |
| `list_threads(cp, graph, user_id)` | `cp.list(None, filter={"user_id": ...})` หา thread ของ user → `graph.get_state()` นับ messages + ข้อความล่าสุด |
| `redact(uri)` | ซ่อน password เวลา print |

---

## 5. `langgraph-checkpoint-postgres` ถูกเรียกตรงไหน / ทำอะไร

**ถูกเรียกเฉพาะเมื่อ `CHECKPOINTER=postgres`** (import แบบ lazy ใน `open_checkpointer`) — ถ้าเป็น `memory` ไม่แตะ lib นี้เลย

```mermaid
flowchart TB
    A["open_checkpointer()"] --> B{"CHECKPOINTER"}
    B -->|memory| M["InMemorySaver()"]
    B -->|postgres| C["ConnectionPool(POSTGRES_URI)<br/>autocommit=True<br/>row_factory=dict_row<br/>prepare_threshold=0"]
    C --> D["PostgresSaver(pool)"]
    D --> E["saver.setup()<br/>สร้าง/ migrate ตาราง"]
    E --> F["create_deep_agent(checkpointer=saver)"]
    F --> G["LangGraph เรียกเองระหว่างรัน:<br/>get_tuple · put · put_writes"]
    F --> H["โค้ดเราเรียกเอง:<br/>list(filter=user_id) · delete_thread"]
```

| API ของ `PostgresSaver` | ใครเรียก | ใช้ทำอะไร |
|---|---|---|
| `PostgresSaver(pool)` | `open_checkpointer` | สร้าง saver บน connection pool |
| `setup()` | `open_checkpointer` | สร้างตาราง `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` (idempotent — เรียกซ้ำได้) |
| `get_tuple(config)` | LangGraph (ตอน `invoke` / `get_state`) | โหลด checkpoint ล่าสุดของ thread |
| `put(...)` / `put_writes(...)` | LangGraph (ทุก step) | บันทึก state ใหม่ + ผลของแต่ละ node |
| `list(None, filter={"user_id": ...})` | `list_threads()` | หา thread ของ user จาก metadata |
| `delete_thread(thread_id)` | tests | ลบ thread ทั้งหมด |
| `conn.connection()` | `check_env` | query `select version()` และนับ `distinct thread_id` |

**ทำไมต้อง `autocommit=True` + `dict_row`** — lib กำหนดไว้: `setup()` ต้อง commit ตาราง และโค้ดใน lib อ่านผลแบบ `row["column"]`
(ค่าเดียวกับที่ `PostgresSaver.from_conn_string()` ใช้ แต่เราใช้ **pool** เพื่อรองรับหลาย request)

**สิ่งที่เรียนรู้ระหว่างทำ:** checkpoint ดิบจาก `list()` / `get_tuple()` ของ lib เวอร์ชันนี้ **ไม่ได้มี messages ครบในก้อนเดียว**
→ ต้องอ่าน state ผ่าน `graph.get_state(config)` ซึ่ง LangGraph ประกอบ state ให้ถูกต้อง

---

## 6. ผลลัพธ์สุดท้าย (พิสูจน์ด้วย test + demo)

```mermaid
sequenceDiagram
    participant P1 as Process 1
    participant P2 as Process 2
    participant P3 as Process 3
    participant PG as Postgres (checkpoints)
    participant MV as Milvus (/memories/)

    P1->>PG: thread=resume-X : "code word = PINEAPPLE-42" (ไม่ให้เซฟลง memory)
    Note over P1: process จบ
    P2->>PG: get_tuple(resume-X) → 2 messages
    P2-->>P2: ตอบ "PINEAPPLE-42" ✅
    P3->>PG: thread ใหม่ → ว่าง
    P3->>MV: ls / read AGENTS.md → ไม่มี code word
    P3-->>P3: "ไม่มี code word ในความจำ" ✅
```

| Test (`tests/test_postgres_checkpointer.py`) | พิสูจน์ |
|---|---|
| `test_inmemory_checkpointer_forgets_thread_in_new_instance` | baseline: RAM saver ใหม่ = thread หาย |
| `test_postgres_thread_survives_new_connection_pool` | ปิด pool เปิดใหม่ → messages เดิมครบ, คุยต่อได้, `delete_thread` ลบได้ |
| `test_postgres_thread_survives_process_exit` | **process ลูก** เขียน thread แล้วจบ → process แม่อ่านได้ |
| `test_threads_are_isolated_but_milvus_memory_is_shared` | thread B ไม่เห็นข้อความของ thread A แต่อ่าน `/memories/fact.md` ที่ A เขียนได้ |
| `test_list_threads_by_user` | filter thread ตาม `user_id` ได้ถูกคน + นับ messages |

```bash
POSTGRES_TEST_URI=postgresql://agent:agent@localhost:5432/agent_checkpoints \
  python -m pytest tests/test_postgres_checkpointer.py -v
CHECKPOINTER=postgres ./scripts/demo_resume.sh
```

---

## 7. ใช้งาน

```bash
docker compose up -d postgres
# .env → CHECKPOINTER=postgres
python -m scripts.check_env                   # checkpointer=postgres ... threads=N
python -m app.cli --user earth                # /thread · /threads · /history · /new
python -m app.cli --user earth --thread abc   # กลับมาคุยต่อ
```

ดูข้อมูลดิบใน Postgres:

```bash
docker compose exec postgres psql -U agent -d agent_checkpoints \
  -c "select thread_id, count(*) as checkpoints, max(metadata->>'user_id') as user_id from checkpoints group by thread_id;"
```

## 8. ข้อควรรู้ก่อนขึ้น production

- checkpoint เก็บ **ทุก step** → ข้อมูลโตเร็ว ต้องมีงานลบ thread เก่า (`delete_thread`)
- แอป async (FastAPI) ใช้ `AsyncPostgresSaver` + `AsyncConnectionPool`
- `thread_id` ควรผูกกับ user ในฝั่งแอป (ตรวจสิทธิ์ก่อน resume) — LangGraph ไม่ได้กันให้
- human-in-the-loop (`interrupt_on`) ต้องมี checkpointer ถาวรถึงจะรอข้ามวันได้
