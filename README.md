<div align="center">

<!-- ANIMATED TITLE BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=CortexFlow&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Multi-Agent%20AI%20Orchestration%20System&descAlignY=55&descSize=20" width="100%"/>

<!-- MAIN BADGE ROW -->
<p>
  <a href="https://cortexflow-metn.onrender.com">
    <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-cortexflow--metn.onrender.com-6C63FF?style=for-the-badge&labelColor=1a1a2e"/>
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Orchestration-FF6B35?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-REST%20Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4.1--mini-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Mem0-Long--Term%20Memory-00D4AA?style=for-the-badge&logo=databricks&logoColor=white"/>
  <img src="https://img.shields.io/badge/pgvector-Vector%20Store-336791?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-RAG%20Retrieval-EF4035?style=for-the-badge&logo=meta&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/MCP-Tool%20Integration-00BCD4?style=for-the-badge&logo=protocol&logoColor=white"/>
  <img src="https://img.shields.io/badge/DuckDuckGo-Web%20Search-DE5833?style=for-the-badge&logo=duckduckgo&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Container%20Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white"/>
</p>

<br/>

> ### 🧠 *An AI system that thinks, critiques, remembers, and refines itself — autonomously.*
> **LangGraph multi-agent orchestration · RAG pipeline · Long-term memory · Self-improving critic loop · MCP tool use**

<br/>

</div>

---

## 📌 Table of Contents

- [🌊 What is CortexFlow?](#-what-is-cortexflow)
- [✨ Feature Showcase](#-feature-showcase)
- [🏗️ System Architecture](#️-system-architecture)
- [🔄 Agent Flow Diagram](#-agent-flow-diagram)
- [📚 RAG Pipeline Deep Dive](#-rag-pipeline-deep-dive)
- [🧬 Dual-Memory Architecture](#-dual-memory-architecture)
- [🤖 Agent Reference](#-agent-reference)
- [🛠️ MCP Tool Integration](#️-mcp-tool-integration)
- [🧰 Tech Stack](#-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [🌐 API Reference](#-api-reference)
- [🗂️ Project Structure](#️-project-structure)

---

## 🌊 What is CortexFlow?

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   CortexFlow is a production-ready multi-agent AI system built       ║
║   on LangGraph. It doesn't just answer questions — it PLANS,         ║
║   EXECUTES, GENERATES, CRITIQUES, and IMPROVES its own answers       ║
║   in a self-refining loop before returning them to you.              ║
║                                                                      ║
║   ⚡ Routes queries to specialized agents (math, code, search, RAG)  ║
║   🧠 Remembers you across sessions via Mem0 + pgvector               ║
║   🔁 Refines its own answers through a critic-improve feedback loop   ║
║   🛠️  Calls real tools via the Model Context Protocol (MCP)           ║
║   🌐 Deployed as a live REST API on Render                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

</div>

---

## ✨ Feature Showcase

<div align="center">

| 🔮 Feature | ⚙️ How It Works | 🎯 Why It Matters |
|:---|:---|:---|
| **🔀 Intelligent Routing** | GPT-4.1-mini classifies every query into `math`, `coding`, `search`, `rag`, `notes`, or `reasoning` | Right tool for every job — no wasted tokens |
| **📚 Advanced RAG** | Query rewriting → FAISS MMR retrieval → LLM re-ranking → relevance check → generation | Multi-stage pipeline beats naive chunk retrieval |
| **🧬 Long-Term Memory** | Mem0 + pgvector persists user-specific facts across all sessions | Remembers your name, preferences, and context forever |
| **🔁 Critic-Improve Loop** | GPT-4.1-mini critiques every complex answer; Qwen rewrites if `BAD` | Self-correcting AI that never settles for mediocre |
| **🛠️ MCP Tool Use** | Filesystem read/write + Playwright browser automation via MCP | Extends the agent into the real world |
| **🤖 Dual-Model Design** | GPT-4.1-mini for routing/critic; Qwen3:14b (local) for generation | Speed + quality, cost efficiency + power |
| **💬 Conversation History** | 3-turn sliding window kept in LangGraph state | Coherent multi-turn dialogue without bloat |
| **🌐 REST API** | FastAPI backend deployed on Render | Production-ready, zero-config integration |

</div>

---

## 🏗️ System Architecture

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════╗
║                        🌐 USER QUERY (via REST API)                   ║
╚═══════════════════════════════════╦═══════════════════════════════════╝
                                    ║
                                    ▼
                    ┌───────────────────────────────┐
                    │    🧠  memory_manager.py        │
                    │    retrieve_memory(query)       │
                    │    → pgvector semantic search   │
                    │    → injects into graph state   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                    L A N G G R A P H   G R A P H                     ║
║                                                                       ║
║   ┌──────────────────────────────────────────────────────────────┐   ║
║   │                   plan_decide_agent                          │   ║
║   │                   Model: GPT-4.1-mini                        │   ║
║   │   Classifies query → { type, complexity }                    │   ║
║   └──────────────────────────┬───────────────────────────────────┘   ║
║                              │                                        ║
║                              ▼                                        ║
║   ┌──────────────────────────────────────────────────────────────┐   ║
║   │                     executor_agent                           │   ║
║   │                                                              │   ║
║   │   math      ──► numexpr evaluator                           │   ║
║   │   coding    ──► GPT-4.1-mini (code assistant)               │   ║
║   │   search    ──► DuckDuckGo live web search                  │   ║
║   │   rag       ──► RAG.py pipeline (FAISS + reranking)         │   ║
║   │   notes     ──► MCP filesystem server                       │   ║
║   │   reasoning ──► GPT-4.1-mini (direct)                       │   ║
║   └──────────────────────────┬───────────────────────────────────┘   ║
║                              │                                        ║
║              ┌───────────────┴───────────────┐                       ║
║              │  math / notes? → finalize      │                       ║
║              └───────────────┬───────────────┘                       ║
║                              │ (other types)                         ║
║                              ▼                                        ║
║   ┌──────────────────────────────────────────────────────────────┐   ║
║   │                    generator_agent                           │   ║
║   │                    Model: GPT-4.1-mini                       │   ║
║   │   Synthesizes final answer using:                            │   ║
║   │   • Retrieved executor output                                │   ║
║   │   • Long-term memory from Mem0                               │   ║
║   │   • 3-turn conversation history                              │   ║
║   └──────────────────────────┬───────────────────────────────────┘   ║
║                              │                                        ║
║              ┌───────────────┴───────────────┐                       ║
║              │  simple? → finalize            │                       ║
║              └───────────────┬───────────────┘                       ║
║                              │ (complex)                             ║
║                              ▼                                        ║
║   ┌──────────────────────────────────────────────────────────────┐   ║
║   │                   simplifier_agent                           │   ║
║   │                   Model: GPT-4.1-mini                        │   ║
║   │   Makes complex answers short, clear, beginner-friendly      │   ║
║   └──────────────────────────┬───────────────────────────────────┘   ║
║                              │                                        ║
║                              ▼                                        ║
║   ┌──────────────────────────────────────────────────────────────┐   ║
║   │                     critic_agent                             │   ║
║   │                     Model: GPT-4.1-mini                      │   ║
║   │   Strict evaluation → returns GOOD or BAD: <reason>         │   ║
║   └──────────────┬─────────────────────────────────┬────────────┘   ║
║                  │ GOOD                             │ BAD             ║
║                  ▼                                  ▼                ║
║           ┌─────────────┐              ┌───────────────────────┐     ║
║           │  finalize   │              │    improve_agent       │     ║
║           │  → store    │              │    Model: GPT-4.1-mini │     ║
║           │    memory   │              │    Rewrites based on   │     ║
║           └─────────────┘              │    critic feedback     │     ║
║                                        └──────────┬────────────┘     ║
║                                                   │                  ║
║                                           loops ──┘ back to critic   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    💾  store_memory()           │
                    │    → Mem0 + pgvector           │
                    └───────────────────────────────┘
                                    │
                                    ▼
                          ✅  FINAL ANSWER
```

</div>

---

## 🔄 Agent Flow Diagram

<div align="center">

```
  USER INPUT
      │
      ▼
  ┌─────────────────┐
  │  retrieve_memory │ ◄── pgvector semantic search
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ plan_decide     │ ──► { type: "math|coding|search|rag|notes|reasoning",
  │  (GPT-4.1-mini) │       complexity: "simple|complex" }
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────┐
  │                  executor_agent                  │
  │                                                  │
  │  math ──► numexpr ──────────────────┐            │
  │  code ──► GPT-4.1-mini ────────────┤            │
  │  search ──► DuckDuckGo ────────────┤            │
  │  rag ──► FAISS RAG pipeline ───────┤            │
  │  notes ──► MCP filesystem ─────────┤            │
  │  reason ──► GPT-4.1-mini ──────────┘            │
  └────────┬────────────────────────────────────────┘
           │
     ┌─────┴──────────┐
     │ math or notes? │
     └───┬────────────┘
         │ YES ──► finalize ──► store ──► DONE
         │ NO
         ▼
  ┌─────────────────┐
  │  generator      │ ◄── memory + history + executor output
  │  (GPT-4.1-mini) │
  └────────┬────────┘
           │
     ┌─────┴──────────┐
     │   simple?      │
     └───┬────────────┘
         │ YES ──► finalize ──► DONE
         │ NO
         ▼
  ┌─────────────────┐
  │  simplifier     │
  │  (GPT-4.1-mini) │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │    critic       │ ──► GOOD ──► finalize ──► DONE
  │  (GPT-4.1-mini) │
  └────────┬────────┘
           │ BAD
           ▼
  ┌─────────────────┐
  │    improve      │
  │  (GPT-4.1-mini) │ ──────────────► loops back to critic
  └─────────────────┘
```

</div>

---

## 📚 RAG Pipeline Deep Dive

<div align="center">

```
  📄 KNOWLEDGE BASE
  docs/sample1.pdf + docs/sample2.pdf
         │
         ▼
  ┌─────────────────────────────────────┐
  │  RecursiveCharacterTextSplitter     │
  │  chunk_size=500 │ overlap=50        │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  OpenAI text-embedding-3-small      │
  │  → FAISS vector index (lazy init)   │
  └──────────────────┬──────────────────┘
                     │
  ═══════════════════╪══════ QUERY TIME ══════════════════
                     │
  USER QUERY ──────► re_write_query()
                     │  ↳ resolves pronouns
                     │  ↳ adds history context
                     ▼
  ┌─────────────────────────────────────┐
  │  FAISS MMR Retrieval                │
  │  search_type="mmr" │ k=5            │
  │  (diversity-aware, not just top-k)  │
  └──────────────────┬──────────────────┘
                     │ 5 candidate chunks
                     ▼
  ┌─────────────────────────────────────┐
  │  re_ranking_agent()                 │
  │  GPT-4.1-mini scores each chunk 1–10│
  │  → keeps top 2 most relevant        │
  └──────────────────┬──────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │  is_relevant()                      │
  │  GPT-4.1-mini: context sufficient?  │
  │  YES ──► generator_agent()          │
  │  NO  ──► "Not enough context"       │
  └─────────────────────────────────────┘
```

</div>

---

## 🧬 Dual-Memory Architecture

<div align="center">

```
  ╔═══════════════════════════════════════════════════════════╗
  ║              C O R T E X F L O W   M E M O R Y           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║  LAYER 1: SHORT-TERM (In-Graph State)                     ║
  ║  ─────────────────────────────────────────────────────    ║
  ║  • Sliding 3-turn conversation window                     ║
  ║  • Stored in LangGraph State TypedDict                    ║
  ║  • history: List[{ user: str, assistant: str }]           ║
  ║  • Injected into generator prompt every turn              ║
  ║  • Lives for the duration of the session                  ║
  ║                                                           ║
  ║  LAYER 2: LONG-TERM (Persistent)                          ║
  ║  ─────────────────────────────────────────────────────    ║
  ║  • Powered by Mem0 (mem0ai)                               ║
  ║  • Backed by pgvector on PostgreSQL                       ║
  ║  • Embeddings: text-embedding-3-small                     ║
  ║  • Scoped per user_id                                     ║
  ║                                                           ║
  ║  ON EVERY QUERY:                                          ║
  ║    memory.search(query, user_id) → inject into prompt    ║
  ║                                                           ║
  ║  ON EVERY RESPONSE:                                       ║
  ║    memory.add([user_msg, assistant_msg], user_id)         ║
  ║    → stored as semantic memory forever                    ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
```

</div>

---

## 🤖 Agent Reference

<div align="center">

| 🤖 Agent | 🧠 Model | ⚡ Role | 📤 Output |
|:---|:---:|:---|:---|
| `plan_decide_agent` | GPT-4.1-mini | Routes query to best tool + estimates complexity | `{ type, complexity }` JSON |
| `executor_agent` | — | Dispatches to the right sub-agent | Raw tool output |
| `math_agent` | `numexpr` | Evaluates math expressions via safe eval | Numeric result |
| `coding_agent` | GPT-4.1-mini | Programming help, debugging, code gen | Code + explanation |
| `search_agent` | DuckDuckGo | Live internet search for current info | Web results text |
| `RAG pipeline` | FAISS + GPT-4.1-mini | Retrieval from local PDF knowledge base | Ranked chunks |
| `filesystem_agent` | MCP + GPT-4.1-mini | Read/write local notes via MCP *(disabled on Render)* | File contents |
| `generator_agent` | GPT-4.1-mini | Synthesizes answer using memory + history + context | Full answer |
| `simplifier_agent` | GPT-4.1-mini | Makes complex answers clear and beginner-friendly | Simplified answer |
| `critic_agent` | GPT-4.1-mini | Strict quality evaluator | `GOOD` or `BAD: <reason>` |
| `improve_agent` | GPT-4.1-mini | Rewrites answer based on critic feedback | Improved answer |
| `finalize_node` | — | Updates history + triggers memory storage | Final state |

</div>

---

## 🛠️ MCP Tool Integration

CortexFlow uses the **Model Context Protocol (MCP)** to extend agent capabilities into the real world:

<div align="center">

```
  ┌────────────────────────────────────────────────────┐
  │            MCP Architecture                        │
  │                                                    │
  │   graph.py (LangGraph)                             │
  │        │                                           │
  │        ▼                                           │
  │   mcp_client.py ──► mcp_server.json               │
  │                           │                        │
  │              ┌────────────┴────────────┐           │
  │              │                         │           │
  │   @modelcontextprotocol/    @playwright/mcp        │
  │     server-filesystem                              │
  │              │                         │           │
  │   read_file()         navigate()                   │
  │   write_file()        click()                      │
  │   list_dir()          screenshot()                 │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

| MCP Server | Capability | Status |
|:---|:---|:---:|
| `@modelcontextprotocol/server-filesystem` | Read and write local notes files | ⚠️ Disabled on Render |
| `@playwright/mcp` | Full browser automation | ⚠️ Disabled on Render |

> **Note:** MCP tools require a local filesystem and are disabled in the cloud deployment. They are fully functional in self-hosted setups.

</div>

---

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|:---:|:---|
| 🕸️ **Orchestration** | LangGraph · LangChain |
| 🤖 **LLMs** | OpenAI GPT-4.1-mini |
| 📦 **Vector Store** | FAISS (local RAG) · pgvector (long-term memory) |
| 🧠 **Memory** | Mem0 (`mem0ai`) |
| 🗄️ **Database** | PostgreSQL |
| 🔍 **Embeddings** | OpenAI `text-embedding-3-small` |
| 🌐 **Web Search** | DuckDuckGo Search (`ddgs`) |
| 📄 **PDF Parsing** | PyPDF |
| 🔧 **MCP** | `mcp-use` · `@modelcontextprotocol/server-filesystem` · `@playwright/mcp` |
| ➕ **Math** | `numexpr` |
| 🚀 **API** | FastAPI · Uvicorn |
| 🐳 **Container** | Docker |
| ☁️ **Cloud** | Render |

</div>

---

## 🚀 Quick Start

### Option A — Use the Live API *(zero setup)*

```bash
curl -X POST https://cortexflow-metn.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is recursion in Python?"}'
```

---

### Option B — Self-Host

#### 1️⃣ Clone

```bash
git clone https://github.com/sivakumarviswaa19/CortexFlow.git
cd CortexFlow
```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Configure environment

```bash
cp .env.example .env
# Edit .env with your values:
```

```env
OPENAI_API_KEY=your_openai_api_key
CONNECTION=postgresql://user:password@localhost:5432/cortex_memory
USER=your_name
path=/path/to/your/notes.txt        # for MCP filesystem agent
```

#### 4️⃣ Set up PostgreSQL + pgvector

```sql
CREATE DATABASE cortex_memory;
\c cortex_memory
CREATE EXTENSION vector;
```

#### 5️⃣ Run

```bash
uvicorn main:app --reload
```

---

### Option C — Docker

```bash
docker build -t cortexflow .
docker run -p 8000:8000 --env-file .env cortexflow
```

---

## 🌐 API Reference

**Base URL:** `https://cortexflow-metn.onrender.com`

<div align="center">

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/` | Health check — returns welcome message |
| `POST` | `/chat` | Send a query, receive an AI-refined answer |

</div>

#### `POST /chat`

**Request:**
```json
{
  "query": "Explain binary search with code"
}
```

**Response:**
```json
{
  "response": "Binary search is a divide-and-conquer algorithm..."
}
```

---

## 🗂️ Project Structure

```
CortexFlow/
│
├── 📄 main.py              # FastAPI app · /chat endpoint · session history
├── 🕸️  graph.py             # LangGraph graph · all agents · routing logic
├── 📚 RAG.py               # RAG pipeline (rewrite → retrieve → rerank → generate)
├── 🧠 memory_manager.py    # Mem0 long-term memory (retrieve + store)
├── 🔌 mcp_client.py        # MCP client setup
├── ⚙️  mcp_server.json      # MCP server config (filesystem + playwright)
├── 🐳 Dockerfile           # Docker container config
├── 📦 requirements.txt     # Python dependencies
├── 🔒 .env                 # Environment variables (never commit!)
├── 📋 .env.example         # Template for required env vars
└── 📁 docs/
    ├── sample1.pdf         # Knowledge base document 1
    └── sample2.pdf         # Knowledge base document 2
```

---

<div align="center">

## 🌟 Portfolio Context

> CortexFlow was built as a **production agentic AI system** demonstrating:
>
> ✅ Multi-agent orchestration with LangGraph state machines  
> ✅ Self-improving AI via critic-feedback loops  
> ✅ Hybrid retrieval (MMR + LLM re-ranking)  
> ✅ Persistent cross-session memory with vector DBs  
> ✅ Tool-use via the Model Context Protocol  
> ✅ REST API deployment on cloud infrastructure  

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

<p>
  <b>Built by <a href="https://github.com/sivakumarviswaa19">Viswaa</a></b>
  &nbsp;·&nbsp;
  <a href="https://cortexflow-metn.onrender.com">🚀 Live Demo</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/sivakumarviswaa19/CortexFlow">⭐ Star this repo</a>
</p>

</div>
