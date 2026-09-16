# Milvus Store — memory ของ `create_deep_agent` ลง Milvus ได้ยังไง

> ไฟล์ที่ agent เขียนใต้ **`/memories/`** = 1 row ใน Milvus collection `agent_memories`
> ทุกอย่างอื่น (`/scratch/...`, messages) อยู่ใน state ของ thread (ดู `CHECKPOINTER_AND_THREAD_ID.md`)

---

## 1. TL;DR

| คำถาม | คำตอบ |
|---|---|
| Deep Agents คุยกับ Milvus ตรงๆ ไหม | **ไม่** — Deep Agents รู้จักแค่ `BaseStore` ของ LangGraph เราจึงเขียน **`MilvusStore(BaseStore)`** (`app/milvus_store.py`) เป็นตัวแปล |
| ต่อ Milvus ยังไง | `MilvusStore(uri=...)` — path ไฟล์ = Milvus Lite, `http(s)://` = Milvus server/Zilliz Cloud (+ token / user,password / db_name) |
| agent รู้ได้ยังไงว่าไฟล์ไหนต้องไป Milvus | `CompositeBackend(routes={"/memories/": StoreBackend(...)})` — path ขึ้นต้น `/memories/` ถูก route ไป store, ที่เหลือไป `StateBackend` |
| key ที่อยู่ใน Milvus หน้าตาแบบไหน | `CompositeBackend` **ตัด prefix** ออก: `/memories/research/milvus.md` → key `/research/milvus.md` |
| แยก user ยังไง | `namespace = ("poc-agent", user_id)` → เก็บเป็น string `poc-agent.earth` |
| data ถูกยัดลงยังไง | `write_file` → `StoreBackend.write` → `store.put()` → `MilvusStore.batch([PutOp])` → embed content → `client.upsert()` |
| ค้นหายังไง | ไม่มี query → scalar filter ตาม namespace · มี query → vector search (COSINE) + filter namespace |
| ใช้ในโค้ดทีม | `mem = create_milvus_memory(uri=..., namespace=...)` → `create_deep_agent(..., **mem.agent_kwargs())` |

---

## 2. ภาพรวม (layers)

```mermaid
flowchart TB
    subgraph AGENT["create_deep_agent (deepagents)"]
        LLM["LLM gpt-5"]
        FSM["FilesystemMiddleware<br/>tools: ls · read_file · write_file · edit_file · glob · grep"]
        MM["MemoryMiddleware<br/>memory=['/memories/AGENTS.md']<br/>โหลดเข้า system prompt ตอนเริ่ม thread"]
        T["tool: search_memories<br/>(ของโปรเจกต์นี้)"]
    end

    subgraph BACKEND["Backend routing (deepagents.backends)"]
        CB{"CompositeBackend"}
        SB["StateBackend<br/>ไฟล์ใน thread state"]
        STB["StoreBackend<br/>namespace = ('poc-agent', user_id)"]
    end

    subgraph ADAPTER["app/milvus_store.py"]
        MS["MilvusStore(BaseStore)<br/>batch(): GetOp · PutOp · SearchOp · ListNamespacesOp"]
        EMB["Embeddings<br/>text-embedding-3-large (3072d)<br/>หรือ HashEmbeddings (tests)"]
    end

    MV[("Milvus<br/>collection agent_memories")]

    LLM -->|tool calls| FSM
    FSM --> CB
    MM --> CB
    CB -->|"path ขึ้นต้น /memories/<br/>(ตัด prefix)"| STB
    CB -->|"path อื่น เช่น /scratch/x.md"| SB
    STB -->|"store.get / put / search"| MS
    T -->|"store.search(query=...)"| MS
    MS --> EMB
    MS -->|"pymilvus MilvusClient<br/>query · upsert · delete · search"| MV
    SB -.->|"บันทึกกับ checkpointer"| CP[("Checkpointer<br/>RAM / Postgres")]
```

---

## 3. Connection: URI → Milvus

`MilvusStore.__init__` (`app/milvus_store.py`) — ค่ามาจาก `.env` ผ่าน `make_store()` (`app/agent.py`)

| `.env` | ความหมาย |
|---|---|
| `MILVUS_DB_URI` | `./data/milvus_memory.db` (Lite) · `http://localhost:19531` (Docker ของเรา) · `https://xxx.zillizcloud.com` |
| `MILVUS_DB_TOKEN` | `user:password` หรือ API key |
| `MILVUS_DB_USER` / `MILVUS_DB_PASSWORD` | ทางเลือกแทน token |
| `MILVUS_DB_NAME` / `MILVUS_DB_CREATE` | database (server เท่านั้น) / สร้างให้ถ้าไม่มี |
| `MILVUS_COLLECTION` | ชื่อ collection (default `agent_memories`) |

> ชื่อต้องเป็น `MILVUS_DB_URI` ไม่ใช่ `MILVUS_URI` — pymilvus อ่าน `MILVUS_URI` เองตอน import และรับเฉพาะ `http(s)://` (เคยทำให้ import พังทั้งโปรเจกต์)

```mermaid
flowchart TB
    A["MilvusStore(uri, token, user, password, db_name, create_db)"] --> B{"uri ขึ้นต้น http:// https:// ?"}
    B -->|ไม่ใช่| L["Milvus Lite<br/>MilvusClient(uri=path ไฟล์)<br/>db_name = (milvus-lite)"]
    B -->|ใช่| C{"db_name กำหนด<br/>และไม่ใช่ default?"}
    C -->|ไม่| S["MilvusClient(uri, token/user/password)"]
    C -->|ใช่| D["admin client: list_databases()"]
    D -->|มีอยู่| S2["MilvusClient(..., db_name)"]
    D -->|ไม่มี + create_db| E["create_database()"] --> S2
    D -->|ไม่มี + ไม่ create| X["ValueError พร้อมคำแนะนำ"]
    L & S & S2 --> F["_ensure_collection()"]
    F --> G{"collection มีอยู่?"}
    G -->|มี| H["ตรวจ dim ของ field vector<br/>== len(embed_query('dimension probe'))<br/>ไม่ตรง → RuntimeError"] --> LD["load_collection()"]
    G -->|ไม่มี| I["create schema + index AUTOINDEX/COSINE<br/>consistency_level=Strong"] --> LD
```

ต่อไม่ได้ → `ConnectionError("Cannot connect to Milvus at ... (db=...)")`

### Schema ของ collection

| field | type | ตัวอย่าง | ใช้ทำอะไร |
|---|---|---|---|
| `id` | VARCHAR(64) **PK** | `sha256(["poc-agent","earth"], "/AGENTS.md")` | id คงที่ต่อ namespace+key → `put` ซ้ำ = **upsert** |
| `namespace` | VARCHAR | `poc-agent.earth` | แยก user (join ด้วย `.` เพราะ LangGraph ห้ามมี `.` ใน namespace) |
| `key` | VARCHAR | `/research/milvus.md` | path หลังตัด `/memories` |
| `value` | JSON | `{"content": "...", "encoding": "utf-8", "created_at": "...", "modified_at": "..."}` | ข้อมูลไฟล์ตามรูปแบบของ `StoreBackend` |
| `text` | VARCHAR | เนื้อหาที่ใช้ embed (ตัดที่ 16,000 ตัวอักษร) | debug / ดูใน Attu |
| `indexed` | BOOL | `true` | `put(..., index=False)` → ไม่ถูก vector search |
| `created_at` / `updated_at` | INT64 (ms) | `1789539109055` | created คงเดิมเวลา update |
| `vector` | FLOAT_VECTOR(3072) | embedding ของ `text` | semantic search (AUTOINDEX, COSINE) |

---

## 4. Path matching: `/memories/...` ไปเป็น row ได้ยังไง

`create_milvus_memory()` (`app/milvus_memory.py`) ประกอบของให้:

```python
mem = create_milvus_memory(uri=..., embeddings=..., namespace=("poc-agent", "earth"))
# mem.agent_kwargs() ==
{
  "backend": CompositeBackend(
      default=StateBackend(),
      routes={"/memories/": StoreBackend(namespace=lambda rt: ("poc-agent", "earth"), store=milvus_store)},
  ),
  "store":  milvus_store,                 # store เดียวกับใน StoreBackend (blog ส่ง InMemoryStore() ให้ StoreBackend → ไม่ลง Milvus)
  "memory": ["/memories/AGENTS.md"],      # MemoryMiddleware โหลดไฟล์นี้เข้า prompt
}
```

| agent เรียก | CompositeBackend ส่งไป | key ใน store | ลง Milvus? |
|---|---|---|---|
| `write_file("/memories/AGENTS.md")` | StoreBackend | `/AGENTS.md` | ✅ |
| `write_file("/memories/research/milvus.md")` | StoreBackend | `/research/milvus.md` | ✅ |
| `ls("/memories")` | StoreBackend | `/` (ทั้ง namespace) | อ่าน |
| `write_file("/scratch/notes.md")` | StateBackend | — | ❌ อยู่ใน checkpoint ของ thread |
| `write_file("/memoriesX/a.md")` | StateBackend | — | ❌ (prefix ต้องตรง `/memories/`) |

namespace มาจากไหน:

- **คงที่** (โปรเจกต์นี้): `build_agent(user_id)` → `("poc-agent", user_id)` — ตรวจ user id ด้วย regex (ห้าม `.` `#` ฯลฯ)
- **ต่อ request** (service หลาย user): `namespace=lambda rt: ("my-agent", rt.context["user_id"])` + `invoke(..., context={"user_id": ...})`

---

## 5. Write flow: agent เขียน memory → row ใน Milvus

```mermaid
sequenceDiagram
    autonumber
    participant LLM as LLM
    participant FS as FilesystemMiddleware
    participant CB as CompositeBackend
    participant SB as StoreBackend
    participant MS as MilvusStore
    participant E as Embeddings
    participant MV as Milvus

    LLM->>FS: write_file(file_path="/memories/research/milvus.md", content="...")
    FS->>CB: write("/memories/research/milvus.md", content)
    CB->>CB: match route "/memories/" → ตัด prefix
    CB->>SB: write("/research/milvus.md", content)
    SB->>SB: namespace = factory(runtime) → ("poc-agent","earth")
    SB->>MS: get(ns, "/research/milvus.md") (มีไฟล์เดิมไหม)
    MS->>MV: query(id == sha256(ns,key))
    SB->>MS: put(ns, key, {content, encoding, created_at, modified_at})
    MS->>MS: batch([PutOp]) → _put_many()
    MS->>MV: query(id in [...]) เพื่อเก็บ created_at เดิม
    MS->>E: embed_documents([content])
    E-->>MS: vector (3072d)
    MS->>MV: upsert({id, namespace, key, value, text, indexed, created_at, updated_at, vector})
    SB-->>CB: WriteResult(path="/research/milvus.md")
    CB-->>FS: WriteResult(path="/memories/research/milvus.md")
    FS-->>LLM: ToolMessage (ผลการเขียนไฟล์)
```

`edit_file` ทำเหมือนกัน: `get` → แทนข้อความ → `put` (upsert row เดิม, `created_at` คงเดิม)
`delete` → `PutOp(value=None)` → `client.delete(ids=[...])`

---

## 6. Read flow: memory กลับเข้าไปใน agent ได้ 3 ทาง

```mermaid
flowchart LR
    subgraph START["ตอนเริ่ม thread"]
        A1["MemoryMiddleware.before_agent"] --> A2["backend.download_files(['/memories/AGENTS.md'])"]
        A2 --> A3["StoreBackend → store.get(ns,'/AGENTS.md')"]
        A3 --> A4["ใส่เข้า system prompt<br/>(cache ใน state: memory_contents)"]
    end
    subgraph TOOLS["ระหว่างคุย - agent เลือกเอง"]
        B1["ls /memories/research"] --> B2["store.search(ns) ไม่มี query<br/>→ Milvus query filter namespace"]
        B3["read_file /memories/x.md"] --> B4["store.get(ns,'/x.md')<br/>→ Milvus query id == ..."]
        B5["search_memories('vector index')"] --> B6["store.search(ns, query=...)<br/>→ embed_query → Milvus vector search<br/>filter namespace and indexed == true"]
    end
```

| store call | Milvus call | หมายเหตุ |
|---|---|---|
| `get(ns, key)` | `query(filter='id == "<sha256>"', limit=1)` | |
| `search(ns)` (ไม่มี query) | `query(filter='namespace == "a.b" or namespace like "a.b.%"')` | กรองซ้ำใน Python ให้ตรง prefix เป๊ะ (`_` ใน LIKE เป็น wildcard) · เรียงตาม namespace, key |
| `search(ns, query=q)` | `search(data=[embed(q)], anns_field="vector", filter=... and indexed == true, metric COSINE)` | คืน `score` |
| `search(ns, filter={...})` | query ตาม namespace แล้วกรอง `value` ใน Python | รองรับ `$eq $ne $gt $gte $lt $lte` |
| `list_namespaces()` | `query(output_fields=["namespace"])` | รองรับ prefix/suffix, `*`, `max_depth` |
| `put(ns, key, value)` | `query` (created_at เดิม) + embed + `upsert` | |
| `delete(ns, key)` | `delete(ids=[...])` | |

`batch()` ใช้ lock เดียวกัน, อ่านก่อนเขียนในแต่ละ batch (พฤติกรรมเดียวกับ `InMemoryStore`), `abatch()` = `asyncio.to_thread(batch)`

---

## 7. ทำไมเชื่อได้ว่าลง Milvus จริง

| หลักฐาน | ที่ไหน |
|---|---|
| Query Milvus ตรงด้วย pymilvus (ไม่ผ่าน adapter) หลัง `put` / หลัง agent `write_file` | `tests/test_milvus_store.py` T1, `tests/test_agent_memory.py` T6 |
| Differential test เทียบผลกับ `InMemoryStore` ของ LangGraph | T3 |
| Agent ใหม่ + client ใหม่ + thread ใหม่ อ่าน memory ได้ | T7 |
| **Negative control**: ลบ row แล้ว agent ไม่รู้ | T8 + live test (gpt-5) |
| `/scratch/...` ไม่ลง Milvus | T10 |
| Store ที่ agent ใช้เป็น `MilvusStore` จริง (กันบั๊กแบบ blog) | T11, `test_milvus_memory.py` |
| helper ต่อด้วย URI ได้ / namespace ต่อ request แยก user ได้ | `tests/test_milvus_memory.py` |
| ดูด้วยตา | `/mem` ใน CLI, `python -m scripts.inspect_memory --user earth [--query ...]`, Attu http://localhost:8000 (หรือพอร์ตที่ตั้งไว้) |

รัน test ทั้งชุดกับ Milvus server:

```bash
MILVUS_TEST_URI=http://localhost:19531 python -m pytest
```

---

## 8. ข้อจำกัด / สิ่งที่ต้องคิดต่อ

- 1 ไฟล์ = 1 vector (ไม่ chunk) · `value` JSON ~64KB ต่อไฟล์ · `text` ตัดที่ 16,000 ตัวอักษร
- `put` ทุกครั้งเรียก embedding API (cost/latency)
- list/search แบบไม่มี query ดึงได้สูงสุด 16,384 rows ต่อครั้ง (ข้อจำกัด Milvus)
- เปลี่ยน embedding model = dim เปลี่ยน → ต้องใช้ collection ใหม่
- ไม่รองรับ TTL
- `raw_rows()` / `count()` / `drop_collection()` เป็น helper สำหรับ demo/test ไม่ใช่ API ของ `BaseStore`
