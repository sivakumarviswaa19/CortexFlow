<h1 align="center">
  <img src="https://img.shields.io/badge/CortexFlow-Multi--Agent%20AI-blueviolet?style=for-the-badge&logo=openai&logoColor=white"/>
</h1>

<p align="center">
  <b>LangGraph-based multi-agent AI system with RAG pipeline, long-term memory (Mem0 + pgvector), and MCP tool integration</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Orchestration-orange?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Mem0-Long--Term%20Memory-green?style=flat-square&logo=databricks&logoColor=white"/>
  <img src="https://img.shields.io/badge/pgvector-Vector%20Store-blue?style=flat-square&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-RAG%20Retrieval-red?style=flat-square&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/MCP-Tool%20Integration-teal?style=flat-square&logo=protocol&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4.1--mini-412991?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-Qwen3%3A14b-black?style=flat-square&logo=ollama&logoColor=white"/>
</p>

---

## 🧠 What is CortexFlow?

CortexFlow is a **multi-agent AI assistant** orchestrated via LangGraph. It routes user queries across specialized agents — math, coding, search, RAG, notes, and reasoning — then generates, simplifies, critiques, and improves its own answers in a self-refining loop.

It maintains **long-term memory** across sessions using Mem0 backed by pgvector, and connects to external tools (filesystem, browser automation) through the **Model Context Protocol (MCP)**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔀 **Intelligent Routing** | LLM-based plan-and-decide agent classifies every query to the best tool |
| 📚 **Advanced RAG** | Query rewriting → MMR retrieval → LLM re-ranking → relevance check |
| 🧬 **Long-Term Memory** | Mem0 + pgvector stores and retrieves user-specific memories across sessions |
| 🔁 **Critic-Improve Loop** | Self-critique loop refines complex answers before returning them |
| 🛠️ **MCP Tool Use** | Filesystem read/write and Playwright browser automation via MCP |
| 🤖 **Dual-Model Design** | GPT-4.1-mini for routing/critic; Qwen3:14b (local) for generation |
| 💬 **Conversation History** | Sliding 3-turn window keeps context across multi-turn sessions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │     plan_decide_agent   │  ← GPT-4.1-mini
            │  type: math/coding/     │    classifies query
            │  search/rag/notes/      │    + complexity
            │  reasoning              │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │      executor_agent     │
            │                         │
            │  math    → numexpr      │
            │  coding  → Qwen3:14b    │
            │  search  → DuckDuckGo   │
            │  rag     → RAG pipeline │
            │  notes   → MCP fs tool  │
            │  reason  → Qwen3:14b    │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │   math / notes query?   │
            │   YES → finalize        │
            │   NO  → generator       │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │     generator_agent     │  ← Qwen3:14b
            │  synthesizes final      │    uses memory +
            │  answer with memory     │    history + context
            │  + history + context    │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │     simple query?       │
            │   YES → finalize        │
            │   NO  → simplifier      │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │    simplifier_agent     │  ← Qwen3:14b
            │  makes answer clear     │
            │  and beginner-friendly  │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │      critic_agent       │  ← GPT-4.1-mini
            │   returns GOOD / BAD    │
            └────────────┬────────────┘
                         │
            ┌────────────▼────────────┐
            │       GOOD?             │
            │   YES → finalize        │
            │   NO  → improve_agent   │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │      improve_agent      │  ← Qwen3:14b
            │  rewrites based on      │    loops back
            │  critic feedback        │    to critic
            └─────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │      finalize_node      │
            │  updates history        │
            │  stores to Mem0         │
            └─────────────────────────┘
                         │
                         ▼
                   FINAL ANSWER
```

---

## 📚 RAG Pipeline

```
Query
  │
  ▼
re_write_query()         ← resolves pronouns, adds context from history
  │
  ▼
history-augmented retrieval  ← FAISS MMR (top 5 chunks)
  │
  ▼
re_ranking_agent()       ← Qwen3:14b scores each chunk 1–10, keeps top 2
  │
  ▼
is_relevant()            ← GPT-4.1-mini checks if context is sufficient
  │
  ▼
generator_agent()        ← synthesizes final answer
```

---

## 🗂️ Project Structure

```
CortexFlow/
├── main.py              # LangGraph graph, all agents, main loop
├── RAG.py               # Full RAG pipeline (rewrite, retrieve, rerank)
├── memory_manager.py    # Mem0 long-term memory (store + retrieve)
├── mcp_client.py        # MCP client setup
├── mcp_server.json      # MCP server config (filesystem + playwright)
├── .env                 # Environment variables (never commit this)
├── .env.example         # Template for required env vars
└── sample1.pdf          # Knowledge base documents
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/sivakumarviswaa19/CortexFlow.git
cd CortexFlow
```

### 2. Install dependencies

```bash
pip install langchain langchain-openai langchain-ollama langchain-community
pip install langchain-huggingface langchain-text-splitters faiss-cpu
pip install langgraph mem0ai psycopg2-binary mcp-use numexpr python-dotenv
```

### 3. Pull the local model (requires [Ollama](https://ollama.com))

```bash
ollama pull qwen3:14b
```

### 4. Set up PostgreSQL + pgvector

```bash
# Install pgvector extension in your PostgreSQL instance
CREATE EXTENSION vector;
CREATE DATABASE cortex_memory;
```

### 5. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_key
CONNECTION=postgresql://user:password@localhost:5432/cortex_memory
USER=your_name
path=/your/notes/file/path.txt
```

### 6. Run

```bash
python main.py
```

---

## 🤖 Agent Reference

| Agent | Model | Role |
|---|---|---|
| `plan_decide_agent` | GPT-4.1-mini | Classifies query type and complexity |
| `executor_agent` | — | Dispatches to the right tool |
| `math_agent` | numexpr | Evaluates mathematical expressions |
| `coding_agent` | Qwen3:14b | Answers programming questions |
| `search_agent` | DuckDuckGo | Fetches live web results |
| `RAG pipeline` | FAISS + Qwen3:14b | Retrieves and re-ranks from knowledge base |
| `filesystem_agent` | MCP + Qwen3:14b | Reads and writes local notes via MCP |
| `generator_agent` | Qwen3:14b | Synthesizes final answer with memory + history |
| `simplifier_agent` | Qwen3:14b | Makes complex answers beginner-friendly |
| `critic_agent` | GPT-4.1-mini | Evaluates answer quality (GOOD / BAD) |
| `improve_agent` | Qwen3:14b | Rewrites answer based on critic feedback |

---

## 🧬 Memory System

CortexFlow uses **two layers of memory**:

- **Short-term** — a sliding 3-turn conversation history window passed through graph state
- **Long-term** — [Mem0](https://mem0.ai) backed by **pgvector** on PostgreSQL, persisting user-specific facts across sessions

On every query, relevant memories are retrieved and injected into the generator prompt. After every response, the exchange is stored back to Mem0.

---

## 🛠️ MCP Tools

CortexFlow connects to external tools via the **Model Context Protocol**:

| Server | Capability |
|---|---|
| `@modelcontextprotocol/server-filesystem` | Read and write local notes files |
| `@playwright/mcp` | Browser automation |

---

## 🧰 Tech Stack

`LangGraph` · `LangChain` · `OpenAI GPT-4.1-mini` · `Ollama Qwen3:14b` · `FAISS` · `HuggingFace Embeddings` · `Mem0` · `pgvector` · `PostgreSQL` · `MCP (mcp-use)` · `DuckDuckGo Search` · `Playwright` · `Python 3.11+`

---

## 📄 License

MIT License — feel free to use, fork, and build on this.

---

<p align="center">Built by <a href="https://github.com/sivakumarviswaa19">Viswaa</a></p>
