# 🧠 铸脑 (ZHU·NAO) · CEREBROFORGE

**The Strongest Self-Evolving Cognitive Agent Framework — Human-Brain Inspired, Tool-Evolving, Memory-Persistent**

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-CAST_IRON_SKELETON-6366f1" />
  <img src="https://img.shields.io/badge/Memory-L1%2FL2%2FL3_Brain_Like-10b981" />
  <img src="https://img.shields.io/badge/Evolution-Yunjue_Paper_EGL-a78bfa" />
  <img src="https://img.shields.io/badge/Platform-Linux%20%2B%20Windows-0ea5e9" />
  <img src="https://img.shields.io/badge/LLM-NVIDIA_AI-minimaxai%2Fminimax--m2.7-f59e0b" />
</p>

---

## 🌟 What is CerebroForge?

CerebroForge (铸脑 / ZHU·NAO) is an advanced self-evolving cognitive agent framework that mirrors the human brain's information processing architecture. It combines:

1. **Dual-System Cognition** — System 1 (fast, automated) and System 2 (slow, analytical), routed by prediction error
2. **4±1 Cognitive Chunk Working Memory** — Strict limit on active context, preventing cognitive overload
3. **Prediction-Error-Driven Decisions** — Prior prediction before every action; error determines system activation
4. **Three-Layer Brain-Like Memory** — L1 (episodic) → L2 (patterns) → L3 (wisdom), with reversible compression
5. **Cast-Iron Skeleton DAG** — Pre-defined workflows via LangGraph; LLM only fills nodes, never controls flow
6. **Self-Evolving Tool Library** — Inspired by [Yunjue Agent](https://arxiv.org/abs/2601.18226): forge, enhance, cluster, merge skills on demand
7. **Computer Use** — Full terminal, file operations, Python execution, multi-platform (Linux + Windows)
8. **Ambiguity Clarification** — Never guesses; actively asks clarifying questions when intent is fuzzy
9. **EGL Convergence Monitoring** — Evolutionary Generality Loss tracks system maturity
10. **H5 Web Interface** — Professional dark-themed dashboard with real-time cognitive state visualization

---

## 🏗 Architecture

```
                    ┌──────────────────────────────────────────┐
                    │          铸脑 CEREBROFORGE               │
                    └──────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
            ┌──────────────┐ ┌──────────────┐  ┌──────────────┐
            │  CAST-IRON   │ │  DUAL-SYSTEM │  │  3-LAYER     │
            │  SKELETON    │ │  COGNITION   │  │  MEMORY      │
            │  (LangGraph) │ │  (Pred-Err)  │  │  (SQLite+Vec)│
            └──────┬───────┘ └──────┬───────┘  └──────┬───────┘
                   │                │                  │
    ┌──────────────┼────────────────┼──────────────────┤
    ▼              ▼                ▼                  ▼
┌────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐
│Clarify │  │ Manager  │  │   Executor   │  │  Integrator  │
│ Node   │→│   Node   │→│     Node     │→│     Node     │
└────────┘  └────┬─────┘  └──────────────┘  └──────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌──────────┐    ┌──────────────┐
  │  Tool    │    │    Skill     │
  │Developer │    │    Forge     │
  └──────────┘    └──────────────┘
```

### LangGraph Workflow (Cast-Iron Skeleton)

```
START → Clarify → Manager ─┬→ Tool Developer → Executor → Manager (loop)
                           └→ Executor ────────────────→ Manager (loop)
                                                          │
                                              (terminate) ▼
                                                       Integrator → END
```

The LLM never decides "what to do next" — the DAG controls flow. The LLM only fills semantic content within each node.

---

## 🧠 Core Concepts

### Dual-System Cognition

| Condition | System 1 (Fast) | System 2 (Slow) |
|-----------|-----------------|------------------|
| Prediction Error | < 0.3 | ≥ 0.3 |
| Tool History | High-freq (>10 calls, >80% success) | Low-freq or new |
| Ambiguity | Not ambiguous | Ambiguous or uncertain |
| Task Match | Matches pre-defined DAG node | Novel or complex |
| Output | Direct tool execution + response | Chain-of-thought reasoning → plan → execute |

### 4±1 Cognitive Chunks

Working memory is strictly limited to 4 chunks (±1 for flexibility):

| Chunk | Name | Content |
|-------|------|---------|
| A | North Star | Current task goal |
| B | Ground Truth | Last execution result |
| C | Relevant Memory | ≤2 memories retrieved by surprise/relevance |
| D | Prior Prediction | What we expect next; deviation plan |

**Rule**: Exceeding 4 chunks → information must be accessed via tool calls, not stuffed into context.

### Three-Layer Memory

| Layer | Content | Retention | Compression | Retrieval |
|-------|---------|-----------|-------------|-----------|
| **L1** | Episodic (raw interactions) | Until threshold (100 entries / 50K tokens) | Extract QUAD: goal\|action\|result\|error + index key | By keyword + recency |
| **L2** | Compressed patterns (1 typical case + abstract rule) | 30 days | Merge similar (>5 occurrences) | By vector similarity |
| **L3** | Crystallized wisdom (DAG paths,固化 tools, universal knowledge) | Permanent (frozen after 90 days idle) | Only keep freq>10, success>80% | By weight >0.6 |

**Compression Principle**: Every compressed memory includes an **index key** for reversible recall — compress but never lose the ability to "remember".

### Prediction Error Cycle

```
1. Generate Prior Prediction → What should the next tool output contain?
2. Execute & Observe → Call tool, get actual output
3. Compute Error → error = 1 - similarity(predicted, actual)
4. Decide:
   - error < 0.3 → System 1 (high confidence, fast response)
   - 0.3 ≤ error < 0.7 → Mid confidence, correct prediction and retry
   - error ≥ 0.7 → System 2 (low confidence, deep reasoning)
```

### Ambiguity Detection (4 Rules)

When ANY of these trigger, the agent asks clarifying questions instead of guessing:

1. **Pronouns without context** — "this", "那个", "它" with no recent antecedent
2. **Multiple interpretation paths** — Task goal has >2 reasonable interpretations
3. **Missing key parameters** — Time range, comparison object, output format, specific entity
4. **Prediction error oscillating** — Confidence unstable between System 1 and 2

### Self-Evolution (Yunjue-Inspired)

Based on the [Yunjue Agent paper](https://arxiv.org/abs/2601.18226), CerebroForge implements:

- **Tool Forging**: On-demand creation of atomic Python tools with sandbox validation
- **Tool Enhancement**: Automatic fix of failing tools with error context
- **Tool Clustering**: LLM-based semantic grouping of similar tools
- **Tool Merging**: Consolidation of redundant tools into unified versions
- **Batch Evolution**: Parallel processing with post-batch cluster + merge
- **EGL Monitoring**: Evolutionary Generality Loss = cumulative_tools_synthesized / cumulative_invocations
  - EGL > 0.1 → Exploration phase (allow new tool creation)
  - EGL < 0.01 → Maturity phase (force reuse of existing tools)

---

## 🛠 Tools & Skills

### Base Tools (12)

| Tool | Description | Platform |
|------|-------------|----------|
| `web_search` | DuckDuckGo HTML search | All |
| `web_fetch` | URL content extraction (BeautifulSoup) | All |
| `python_exec` | Sandboxed Python execution | All |
| `file_read` | Read workspace files | All |
| `file_write` | Write workspace files | All |
| `text_extract` | PDF/TXT text extraction | All |
| `image_query` | Vision model query (stub) | All |
| `calculate` | Safe math evaluation | All |
| `run_terminal` | **Multi-platform terminal** (bash/cmd/PowerShell) | Linux + Windows |
| `list_files` | Cross-platform directory listing | All |
| `grep_files` | Text search in workspace files | All |
| `computer_use` | Full computer operations toolkit | Linux + Windows |

### Computer Use Capabilities

Like OpenClaw/Claude Computer Use, CerebroForge can:

- Execute terminal commands (Linux bash, Windows cmd, PowerShell)
- Read/write files in workspace
- Execute Python code in sandbox
- List and search files
- Download files from URLs
- Get system information (OS, disk, memory)
- List running processes
- Launch applications

**Safety**: All dangerous commands are blocked (`rm -rf /`, `sudo`, `format`, etc.). File operations restricted to workspace.

### Evolved Skills

The SkillForge engine generates new tools on demand. Each follows the structure:

```python
__TOOL_META__ = {"name": "...", "description": "...", "dependencies": [...]}

from pydantic import BaseModel, Field

class InputModel(BaseModel):
    ...

class OutputModel(BaseModel):
    ...

def run(input: InputModel) -> OutputModel:
    ...
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd CerebroForge
pip install -r requirements.txt
```

### 2. Configure API

Set your NVIDIA API key (or edit `backend/config.py`):

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### 3. Launch

```bash
# From project root
cd CerebroForge
python -m backend.app

# Or from backend directory
cd backend
python app.py
```

The server starts at `http://localhost:8000`.

### 4. Use the H5 Interface

Open `http://localhost:8000` in your browser. The dark-themed dashboard provides:

- Chat with the agent
- Real-time cognitive state monitoring
- Memory stats and management
- Tool library with forge capability
- Evolution tracking (EGL chart)
- Computer use terminal
- Settings management

### 5. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Main chat |
| POST | `/api/chat/stream` | SSE streaming |
| POST | `/api/clarify` | Handle clarification |
| POST | `/api/forge` | Forge new tool |
| POST | `/api/sleep` | Compress memory |
| POST | `/api/evolve` | Trigger evolution |
| GET | `/api/state` | Full agent state |
| GET | `/api/memory` | Memory stats |
| GET | `/api/tools` | Tool listing |
| GET | `/api/evolution` | EGL stats |
| GET | `/api/interactions` | Recent history |
| POST | `/api/upload` | File upload |
| GET | `/api/config` | Current config |
| POST | `/api/config/update` | Update config |
| POST | `/api/computer` | Computer use |
| POST | `/api/batch_evolve` | Batch evolution |
| GET | `/health` | Health check |

---

## 📁 Project Structure

```
CerebroForge/
├── backend/
│   ├── __init__.py           # Package init with re-exports
│   ├── config.py             # All thresholds, paths, model config
│   ├── schemas.py            # Pydantic models, AgentState TypedDict
│   ├── llm_client.py         # NVIDIA API client (OpenAI SDK)
│   ├── cognitive.py          # Dual-system, 4-chunk, prediction engine
│   ├── memory.py             # 3-layer memory (SQLite + ChromaDB)
│   ├── tools.py              # Tool registry with 12 base tools
│   ├── skill_forge.py        # Skill evolution engine
│   ├── computer_use.py       # Computer operations toolkit
│   ├── workflow.py           # LangGraph cast-iron skeleton
│   ├── agent_core.py         # Main agent orchestrator
│   ├── app.py                # FastAPI server
│   └── prompts/
│       ├── __init__.py
│       ├── loader.py         # Jinja2 template loader
│       └── templates/
│           ├── manager.md
│           ├── tool_developer.md
│           ├── worker.md
│           ├── integrator.md
│           ├── critic.md
│           ├── clarify.md
│           ├── compress.md
│           ├── tool_cluster.md
│           └── tool_merge.md
├── frontend/
│   └── index.html            # Complete H5 dashboard (1.7K lines)
├── data/                     # SQLite DB + ChromaDB vectors (gitignored)
├── workspace/                # User workspace (gitignored)
├── dynamic_skills/           # Evolved private skills (gitignored)
├── dynamic_skills_public/    # Shared evolved skills (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

All thresholds are in `backend/config.py` and adjustable at runtime via `/api/config/update`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L1_MAX_ENTRIES` | 100 | Max L1 entries before compression |
| `L1_MAX_TOKENS` | 50000 | Max L1 tokens before compression |
| `HIGH_FREQ_THRESHOLD` | 10 | Tool calls to qualify as high-frequency |
| `SUCCESS_RATE_THRESHOLD` | 0.80 | Min success rate for System 1 fast path |
| `PREDICTION_ERROR_SYS1` | 0.3 | Error below → System 1 |
| `PREDICTION_ERROR_SYS2` | 0.7 | Error above → System 2 |
| `MAX_COGNITIVE_CHUNKS` | 4 | Working memory chunk limit |
| `MAX_TOOL_FORGE_PER_TASK` | 2 | Max new tools per task |
| `TOOL_EVO_BUDGET_TOKENS` | 5000 | Token budget for tool generation |
| `MAX_TASK_EXECUTION_CNT` | 5 | Max execution cycles before terminate |

---

## 🧬 Six Pillars Architecture

Based on the "Cast-Iron Skeleton" framework philosophy:

### 1. Cast-Iron Skeleton — Pre-orchestrated State Machine
LangGraph defines all valid paths. The LLM only fills nodes — it never decides "what to do next".

### 2. Atomic Ant Colony — Task Decomposition to Comfort Zone
Each node is a single-turn, Pydantic-constrained atomic task. Weak models do well on 100 simple tasks, not 1 complex one.

### 3. Sandwich Verification — Zero-Fault Mechanism
Input validation → Execution → Output validation at every step. Pydantic + jsonschema + secondary critic LLM.

### 4. External Brain Memory — Structured Cognitive Offloading
LLM doesn't "remember" through context — it queries memory via tool calls. L1/L2/L3 with vector search.

### 5. Capability Guardrails — Self-Knowledge Routing
Real-time capability map: math → Python, facts → search, creative → LLM, complex → stronger model.

### 6. Recursive Polishing — Adversarial Iteration
Generate → Critic → Fix loop. Critic is a separate prompt designed to find flaws.

---

## 📚 References

- **Yunjue Agent**: [In-situ Self-Evolving Agent System](https://arxiv.org/abs/2601.18226) — Tool evolution, EGL, parallel batch evolution
- **Dual-Process Theory**: Kahneman's System 1/System 2 framework
- **Predictive Processing**: 70% of perception filled by internal model; attention driven by surprise
- **Working Memory**: Miller's 7±2 → refined to 4±1 cognitive chunks

---

## 📜 License

Apache-2.0

---

**铸脑 CEREBROFORGE** — *Forge your brain, evolve your intelligence.* 🧠⚡
