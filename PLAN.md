# PoC Plan: Deep Agent + Milvus Long-Term Memory

> ที่มา: Milvus blog — *How to Build Production-Ready AI Agents with Deep Agents and Milvus*
> (หัวข้อ "Hands-on: Build an AI Agent with Long-Term Memory Using Milvus and Deep Agents")
> เป้าหมาย: PoC ให้ทีมดู และ **พิสูจน์ได้ว่า memory ถูกเขียนลง Milvus จริง** ไม่ใช่อยู่ใน RAM

---

## 0. สิ่งที่ blog บอก (สรุป)

1. `pip install deepagents tavily-python langchain-milvus`
2. ใช้ `CompositeBackend`: path ทั่วไป → `StateBackend` (อยู่แค่ใน session), path `/memories/` → `StoreBackend` (ถาวร)
3. `create_deep_agent(tools=[internet_search], system_prompt="... เขียนสิ่งสำคัญลง /memories/ ...", backend=backend)`

## 0.1 ปัญหาในโค้ดของ blog (ตรวจกับ source code แล้ว)

| # | ปัญหา | ผลกระทบ | วิธีแก้ใน PoC |
|---|---|---|---|
| 1 | สร้าง `milvus_store` แล้วแต่ส่ง `StoreBackend(store=InMemoryStore())` แทน | **memory ไม่ได้ลง Milvus เลย** — ถ้าก๊อปไปรันจะหลอกตัวเองว่าใช้ได้ | ส่ง Milvus store จริงเข้าไป และมี test ยืนยัน |
| 2 | `from langchain_milvus.storage import MilvusStore` — ใน `langchain-milvus` 0.4.0 (ก.ค. 2026) ไม่มี module/class นี้ มีแค่ `Milvus`, `Zilliz` (VectorStore), retriever และ built-in functions | import error | **เขียน adapter `MilvusStore(BaseStore)` เอง** (ดูข้อ 2) |
| 3 | ไม่ได้ import `create_deep_agent` และไม่ได้ส่ง `store=` / `checkpointer` | รันไม่ขึ้น | แก้ตาม docs ของ deepagents |
| 4 | `StoreBackend` เวอร์ชันปัจจุบันต้องมี `namespace=` (factory) — blog ไม่ได้ใส่ | error ตอน init | ใช้ `namespace=lambda rt: ("poc-agent", user_id)` |
| 5 | `CompositeBackend` ตัด prefix ออก: `/memories/notes.md` ถูกเก็บด้วย key `/notes.md` | ถ้า query Milvus ด้วย key ผิดจะหาไม่เจอ | ใช้ key ที่ตัด prefix แล้วเวลา verify |

> หมายเหตุ: API ของ deepagents เปลี่ยนเร็ว (docs บางหน้าเขียน `StoreBackend(rt, namespace=...)`, source ปัจจุบันเป็น keyword-only `StoreBackend(namespace=..., store=None)`) → **pin version ใน requirements** แล้วยึดตามเวอร์ชันที่ติดตั้งจริง

---

## 1. Architecture ของ PoC

```
User (CLI / Streamlit)
   │
   ▼
create_deep_agent(model, tools, backend, store=milvus_store, checkpointer)
   │
   ├── CompositeBackend
   │     ├── default  → StateBackend      (ไฟล์ชั่วคราวใน thread นี้)
   │     └── /memories/ → StoreBackend(namespace=("poc-agent", user_id))
   │                         │
   │                         ▼
   │                  MilvusStore (BaseStore adapter ที่เราเขียน)
   │                         │  pymilvus
   ▼                         ▼
LLM (OpenAI-compatible)   Milvus standalone (Docker) + Attu UI
```

Collection `agent_memories` (schema เสนอ):

| field | type | หมายเหตุ |
|---|---|---|
| `id` | VARCHAR PK | `sha1(namespace + key)` → ทำให้ put ซ้ำ = upsert |
| `namespace` | VARCHAR | `"poc-agent/user-1"` (join ด้วย `/`) |
| `key` | VARCHAR | path ของไฟล์ เช่น `/milvus_research.md` |
| `value` | JSON | `{content, encoding, created_at, modified_at}` ตามที่ StoreBackend เขียน |
| `text` | VARCHAR | content ที่ใช้ทำ embedding |
| `embedding` | FLOAT_VECTOR | สำหรับ semantic search |
| `created_at`, `updated_at` | INT64 | epoch ms |

---

## 2. งานที่ต้องทำ (ตามลำดับ)

### Phase 1 — Infra (≈0.5 วัน)
1. `docker-compose.yml`: Milvus standalone (etcd + minio + milvus) + **Attu** (UI ให้ทีมเห็นข้อมูลในตาด้วยตาตัวเอง) + volume ถาวร
2. `.env`: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL`, `EMBEDDING_MODEL`, `MILVUS_URI=http://localhost:19530`, `TAVILY_API_KEY` (optional)
3. `pyproject.toml` / `requirements.txt` pin version: `deepagents`, `langgraph`, `langchain-openai`, `pymilvus`, `tavily-python`, `pytest`
   - Python ≥ 3.11 (แนะนำ 3.12)
4. `scripts/healthcheck.py`: ต่อ Milvus ได้ + list collections

### Phase 2 — MilvusStore adapter (≈1 วัน) ← หัวใจของ PoC
5. `app/milvus_store.py`: class `MilvusStore(BaseStore)` จาก `langgraph.store.base`
   - implement `batch(ops)` / `abatch(ops)` รองรับ `GetOp`, `PutOp` (value=None → delete), `SearchOp` (filter by namespace prefix + optional query → vector search), `ListNamespacesOp`
   - สร้าง collection + index (HNSW/AUTOINDEX, COSINE) อัตโนมัติถ้ายังไม่มี
   - put = `upsert` (id คงที่ต่อ namespace+key) แล้ว embed `value["content"]`
   - `get` = `query(filter='id == "..."')`
   - search ไม่มี query → `query` ตาม namespace prefix (`namespace like "poc-agent/%"`) + limit/offset
   - search มี query → `search` บน `embedding` + filter namespace
   - `flush()` หลัง write ใน mode demo เพื่อให้ count เห็นทันที
6. ทางเลือกเร็ว (ถ้าอยาก PoC ภายในวันเดียว): ใช้ **Milvus Lite** (`MILVUS_URI=./milvus_demo.db`) ตอน dev แล้วค่อยสลับเป็น standalone ตอน demo — โค้ดเหมือนกัน

### Phase 3 — Agent (≈0.5 วัน)
7. `app/agent.py`: `build_agent(user_id)`
   - `backend = lambda rt: CompositeBackend(default=StateBackend(...), routes={"/memories/": StoreBackend(namespace=lambda rt: ("poc-agent", user_id))})`
   - `create_deep_agent(model=ChatOpenAI(...), tools=[internet_search], system_prompt=..., backend=backend, store=milvus_store, checkpointer=InMemorySaver(), memory=["/memories/AGENTS.md"])`
   - system prompt บังคับ: "ข้อเท็จจริงเกี่ยวกับผู้ใช้/ผลการค้นคว้าสำคัญ → เขียนลง `/memories/` เสมอ, ก่อนตอบให้ `ls /memories/` และอ่านไฟล์ที่เกี่ยวข้อง"
   - tool: `internet_search` (Tavily) — ถ้าไม่มี key ให้ใช้ fake tool เพื่อ demo ได้ offline
8. `app/cli.py`: `python -m app.cli chat --user u1 --thread t1` (Rich console แสดง tool calls ให้เห็นว่า agent เรียก `write_file /memories/...`)
9. (optional) `app/ui.py` Streamlit: ซ้าย = chat, ขวา = **Memory Inspector** ดึงจาก Milvus ตรงๆ แบบ live

### Phase 4 — Tests ที่พิสูจน์ว่าลง Milvus จริง (≈1 วัน) ← สิ่งที่ทีมจะถาม
ดูข้อ 3

### Phase 5 — Demo pack (≈0.5 วัน)
10. `scripts/inspect_memory.py` (อ่าน Milvus ตรงด้วย pymilvus ไม่ผ่าน agent)
11. `scripts/demo.sh` รันสคริปต์ demo ตามข้อ 4
12. `README.md` + screenshot Attu

**รวม ≈ 3–3.5 วันทำงาน** (ถ้าใช้ Milvus Lite + ไม่ทำ UI ≈ 1.5–2 วัน)

---

## 3. Test plan — "memory ถูกเก็บลง Milvus จริง"

หลักคิด: **ตรวจจาก Milvus โดยตรง (pymilvus/Attu) ไม่เชื่อคำตอบของ agent อย่างเดียว** และมี negative control

| ID | ระดับ | ทดสอบอะไร | Pass เมื่อ |
|---|---|---|---|
| T1 | unit | `store.put` → `pymilvus.query` เจอ row ที่ namespace/key ตรง | row count +1, `value.content` ตรง |
| T2 | unit | put key เดิมซ้ำ = upsert ไม่ใช่ insert ซ้ำ | count ไม่เพิ่ม, content เป็นค่าใหม่ |
| T3 | unit | `get` / `search(namespace_prefix)` / `delete` / `list_namespaces` ทำงานตาม `BaseStore` contract | ผลตรงกับ `InMemoryStore` บน input ชุดเดียวกัน (**differential test**) |
| T4 | unit | semantic search: ใส่ 3 memory, query ความหมายใกล้เคียง | อันดับ 1 เป็นอันที่ถูก |
| T5 | unit | namespace isolation: user1 vs user2 | search ของ user1 ไม่เห็นของ user2 |
| T6 | integration | agent คุย 1 turn ("จำไว้ว่าผมชอบคำตอบสั้น ใช้ Python") | มี `write_file`/`edit_file` ไปที่ `/memories/...` ใน messages **และ** Milvus มี row ใหม่ใน namespace นั้น |
| T7 | integration | **Cross-session**: สร้าง agent ใหม่ใน **process ใหม่** + `thread_id` ใหม่ + checkpointer ว่าง แล้วถาม "ผมชอบภาษาอะไร" | ตอบ Python และมี tool call อ่าน `/memories/` |
| T8 | integration | **Negative control**: ลบ row/drop collection แล้วถามซ้ำ | agent ไม่รู้คำตอบ → พิสูจน์ว่าความจำมาจาก Milvus ไม่ใช่จาก context/checkpointer |
| T9 | integration | **Persistence**: `docker compose restart milvus` แล้วรัน T7 ซ้ำ | ยังจำได้ (ข้อมูลอยู่ใน volume) |
| T10 | integration | ไฟล์ที่ไม่อยู่ใต้ `/memories/` (เช่น `/scratch/notes.md`) | **ไม่** ปรากฏใน Milvus (ยืนยันว่า routing ถูก) |
| T11 | guard | ตรวจว่า `StoreBackend` ที่ agent ใช้เป็น `MilvusStore` ไม่ใช่ `InMemoryStore` | assert type — กันบั๊กแบบใน blog |

- T1–T5, T11: รันกับ Milvus Lite ได้ (เร็ว, ใช้ใน CI)
- T6–T10: ต้องมี LLM จริง → mark `@pytest.mark.llm`; ถ้าต้องการ deterministic ให้ใช้ fake chat model ที่ return tool call `write_file` ตายตัวสำหรับ T6/T10
- ทุก test ใช้ collection ชื่อสุ่ม (`agent_memories_test_<uuid>`) แล้ว drop ตอนจบ

---

## 4. Demo script สำหรับทีม (≈10 นาที)

1. เปิด Attu → ให้เห็นว่า `agent_memories` ว่าง (0 rows)
2. Session A: "ผมชื่อ Earth ทีมใช้ Python และอยากได้คำตอบสั้นๆ" + "research ฟีเจอร์ของ Milvus" → ชี้ tool call `write_file /memories/...` บนจอ
3. Refresh Attu / รัน `inspect_memory.py` → เห็น row, namespace, content, vector
4. ปิดโปรแกรม (kill process) → เปิด Session B (thread ใหม่) → "ผมชื่ออะไร สรุปที่เคย research ให้หน่อย" → ตอบได้
5. `docker compose restart milvus` → ถามอีกครั้ง → ยังได้
6. ลบ row ใน Attu → ถามอีกครั้ง → จำไม่ได้ (negative control)
7. (bonus) semantic search: `inspect_memory.py --query "vector index types"`

---

## 5. โครงสร้างโปรเจกต์

```
deep-agent_milvus/
├── docker-compose.yml        # milvus standalone + etcd + minio + attu
├── .env.example
├── pyproject.toml
├── app/
│   ├── config.py
│   ├── embeddings.py         # OpenAI-compatible embeddings
│   ├── milvus_store.py       # MilvusStore(BaseStore)  ← core
│   ├── tools.py              # internet_search (Tavily / fake)
│   ├── agent.py              # build_agent()
│   ├── cli.py
│   └── ui.py                 # optional Streamlit
├── scripts/
│   ├── healthcheck.py
│   ├── inspect_memory.py
│   └── demo.sh
└── tests/
    ├── conftest.py           # milvus fixture, random collection
    ├── test_milvus_store.py  # T1–T5
    ├── test_store_contract.py# T3 differential vs InMemoryStore
    └── test_agent_memory.py  # T6–T11
```

## 6. การตัดสินใจ (อัปเดต 15 ก.ย. 2026)

- Milvus: **Milvus Lite** (`data/milvus_memory.db`) — docker-compose สำหรับ Attu มีให้เป็น optional เพราะ Attu เปิดไฟล์ Lite ไม่ได้
- LLM: **gpt-5.2** ผ่าน LiteLLM `https://gpt.mfec.co.th/litellm`
- Embedding: `text-embedding-3-small` บน endpoint เดียวกัน (ถ้าไม่มี → `EMBEDDING_PROVIDER=hash`)
- Search: ยังไม่มี Tavily key → mock search
- UI: **CLI + Attu** (ไม่ทำ Streamlit)
- ขั้นตอนใช้งาน/ทดสอบดู README.md

## 6.1 คำถามเดิมก่อนเริ่ม

1. **LLM / Embedding**: ใช้ OpenAI-compatible endpoint ตัวไหน, embedding model อะไร (กำหนด `dim` ของ collection)
2. **Milvus**: standalone Docker (ใกล้ production, มี Attu) หรือ Milvus Lite (เร็วสุด) — แนะนำ Lite ตอน dev + standalone ตอน demo
3. **Search tool**: มี Tavily key ไหม ถ้าไม่มีใช้ fake tool
4. **UI**: CLI + Attu พอไหม หรืออยากได้ Streamlit ที่มี Memory Inspector ข้างๆ chat
5. **Memory granularity**: ไฟล์ markdown ทั้งไฟล์ = 1 vector (ง่าย, ตรงกับ deepagents) หรือ chunk ย่อย (search แม่นขึ้น แต่ adapter ซับซ้อนขึ้น) — PoC แนะนำแบบแรก

## 7. ความเสี่ยง

- deepagents API เปลี่ยนบ่อย → pin version + T11 guard
- Milvus consistency: หลัง insert อาจ query ไม่เจอทันที → ใช้ `consistency_level="Strong"` หรือ `flush()` ใน demo/test
- LLM อาจไม่ยอมเขียน memory → system prompt ชัดเจน + seed `/memories/AGENTS.md` + T6 ตรวจ tool call
- Embedding ทุก put มี cost/latency → PoC รับได้; production ค่อยทำ async/batch
