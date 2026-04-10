# 30-Day Agentic AI Mastery

![30 days](https://img.shields.io/badge/curriculum-30%20days-7c3aed) ![203 terms](https://img.shields.io/badge/glossary-203%20terms-0d9488) ![116 Q&A](https://img.shields.io/badge/interview-116%20Q%26A-f59e0b) ![11 deep dives](https://img.shields.io/badge/beyond%2030-11%20deep%20dives-6366f1) ![No build tools](https://img.shields.io/badge/build-none-10b981)

## What This Is

Hands-on, browser-based 30-day curriculum for Senior/Staff AI Engineers. No frameworks, no build step — open `index.html` in a browser and start learning.

Each day is a self-contained HTML page with interactive widgets, code examples, and concept checks. The full curriculum covers everything from LLM fundamentals through production MLOps, fine-tuning, cloud deployment, and capstone architecture design.

## Quick Start

### Browser Only

```bash
git clone https://github.com/your-username/learn-ai.git
cd learn-ai
open index.html   # macOS
# or: xdg-open index.html (Linux)
# or: start index.html (Windows)
```

No dependencies. No npm install. No build step.

### With AI Coach (FastAPI)

```bash
pip install fastapi uvicorn httpx python-dotenv
# Either export the key or create a .env file:
echo "OPENROUTER_API_KEY=sk-or-..." > .env
uvicorn server:app --reload --port 8000
# Then open index.html or ai.html in your browser
```

The AI Coach provides contextual guidance, concept explanations, and coding help — powered by OpenRouter with SSE streaming. The server automatically tries a chain of free Google Gemma 3 models (`gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-4b-it`, `gemma-3n-e4b-it`) before falling back to `arcee-ai/trinity-mini`. You can override the model per request.

## Curriculum Map

| Week | Days | Theme | Key Topics |
|------|------|-------|------------|
| Week 1 | 1–7 | Core Foundations | LLMs & the API, tool use, RAG, LangGraph, multi-agent systems, production engineering, capstone |
| Week 2 | 8–14 | Frameworks & Models | LangChain/LCEL, LLM providers, vector DBs, embeddings, advanced RAG, CrewAI, Week 2 capstone |
| Week 3 | 15–21 | ML & Fine-Tuning | PyTorch, HuggingFace, LoRA/PEFT, MLflow, MLOps pipelines, AI observability, Week 3 capstone |
| Week 4 | 22–28 | Prompt Eng & Cloud | Advanced prompting, guardrails, structured outputs, auth/tenancy, AWS/GCP/Azure, Docker/K8s, RLHF |
| Week 5 | 29–30 | Capstone | AI Copilot Architecture, full-stack capstone project |

## File Structure

```
learn-ai/
├── index.html              # Main dashboard with week-grouped progress tracker
├── glossary.html           # 203-term glossary with category filtering
├── glossary.md             # Markdown source for the glossary
├── interview.html          # 116 Q&A + 20 comparisons + 40 flashcards
├── interview.md            # Markdown source for interview prep
├── ai.html                 # AI Coach interface (connects to FastAPI backend)
├── server.py               # FastAPI backend — OpenRouter API, SSE streaming, Gemma fallback chain
├── bg-animation.js         # Neural network canvas animation (shared across pages)
│
├── days/                   # 30 individual lesson pages
│   ├── day1.html           # How LLMs Work + The API
│   ├── day1.md             # (+ day2.md–day7.md) Markdown notes for Week 1 lessons
│   ├── day2.html           # Tool Use & ReAct
│   ├── day3.html           # Memory & RAG
│   ├── day4.html           # LangGraph: Stateful Agents
│   ├── day5.html           # Multi-Agent Systems
│   ├── day6.html           # Production Engineering
│   ├── day7.html           # Capstone: Code Review Agent
│   ├── day8.html           # LangChain Deep-Dive
│   ├── day9.html           # LLM Provider Landscape
│   ├── day10.html          # Vector Databases in Depth
│   ├── day11.html          # Embedding Models
│   ├── day12.html          # Advanced RAG Patterns
│   ├── day13.html          # CrewAI & Role-Based Agents
│   ├── day14.html          # Week 2 Capstone
│   ├── day15.html          # PyTorch for LLM Engineers
│   ├── day16.html          # HuggingFace Transformers
│   ├── day17.html          # Fine-Tuning with LoRA/PEFT
│   ├── day18.html          # MLflow & Experiment Tracking
│   ├── day19.html          # MLOps Pipelines
│   ├── day20.html          # AI System Observability
│   ├── day21.html          # Week 3 Capstone
│   ├── day22.html          # Advanced Prompt Engineering
│   ├── day23.html          # AI Guardrails & Safety
│   ├── day24.html          # Structured Outputs & Determinism
│   ├── day25.html          # Auth, Tenancy & Authorization
│   ├── day26.html          # Cloud AI Services
│   ├── day27.html          # Docker & Kubernetes for AI
│   ├── day28.html          # RL & Self-Improving Agents
│   ├── day29.html          # AI Copilot Architecture
│   └── day30.html          # 30-Day Final Capstone
│
├── exercises/              # Hands-on coding exercises
│   ├── index.html          # Exercises dashboard with progress tracking
│   ├── day1_exercises.html
│   ├── ...                 # day2_exercises.html through day30_exercises.html
│   ├── 01_basic_completion.py      # Python: basic API completion
│   ├── 02_conversation_memory.py   # Python: conversation memory
│   ├── 03_streaming.py             # Python: SSE streaming
│   ├── 04_structured_output.py     # Python: structured output / tool use
│   └── 05_tool_use_intro.py        # Python: multi-tool agent intro
│
├── day1/                   # Day 1 standalone Python workspace
│   ├── 01_basic_completion.py
│   ├── 02_conversation_memory.py
│   ├── 03_streaming.py
│   ├── 04_structured_output.py
│   ├── 05_tool_use_intro.py
│   └── activities/
│       ├── index.html                      # Activity launcher
│       ├── activity1_token_tracker.py
│       ├── activity2_system_prompt_engineer.py
│       ├── activity3_cv_extractor.py
│       └── activity4_multi_tool_agent.py
│
├── weeks/                  # Week overview pages
│   ├── week1.html          # Week 1 overview: Core Foundations
│   ├── week2.html          # Week 2 overview: Frameworks & Models
│   ├── week3.html          # Week 3 overview: ML & Fine-Tuning
│   ├── week4.html          # Week 4 overview: Prompt Eng & Cloud
│   └── week5.html          # Week 5 overview: Final Capstone
│
└── next/                   # "What's Next" deep-dive topic pages (beyond Day 30)
    ├── a2a.html            # Agent-to-Agent (A2A) Protocol
    ├── mcp.html            # Model Context Protocol (MCP)
    ├── computer_use.html   # Computer Use API
    ├── evals.html          # Agent Evaluation at Scale
    ├── finetune.html       # Fine-tuning & Model Customization
    ├── constitutional.html # Constitutional AI & Alignment
    ├── agent_frameworks.html # Agentic AI Frameworks Landscape
    ├── llmstack.html       # Building Your Own LLM Stack
    ├── mixture.html        # Mixture-of-Agents
    ├── multimodal.html     # Multimodal Agents
    └── neural_networks.html # Neural Networks Deep Dive
```

## Features

- **30 interactive day lessons** with live widgets, code samples, and concept checks
- **11 "What's Next" deep-dive pages** (`next/`) covering A2A, MCP, Computer Use, Evals, Fine-tuning, Constitutional AI, and more — for going beyond Day 30
- **203-term glossary** with category filtering (models, frameworks, concepts, tools)
- **116 Q&A + 20 comparisons + 40 flashcards** for interview preparation
- **Python exercises** — runnable scripts alongside each HTML lesson (basic completion, memory, streaming, structured output, tool use)
- **Day 1 activity workspace** (`day1/activities/`) — 4 guided hands-on activities with an activity launcher
- **AI Coach** powered by FastAPI + OpenRouter with SSE streaming — free Gemma 3 fallback chain, no paid key required to get started
- **Neural network canvas animation** (`bg-animation.js`) shared across all pages
- **localStorage progress tracking** — your completion state persists across sessions
- **Week-grouped collapsible dashboard** — collapse/expand weeks on the main index

## JD Alignment Map

Maps common Senior/Staff AI Engineer job description requirements to specific curriculum days:

| JD Requirement | Day |
|---|---|
| LangChain / LCEL | Day 8 |
| LLM providers (OpenAI / Claude / Gemini) | Day 9 |
| Vector databases | Day 10 |
| Embeddings | Day 11 |
| Advanced RAG | Day 12 |
| Multi-agent systems (CrewAI) | Day 13 |
| PyTorch inference | Day 15 |
| HuggingFace / Ollama | Day 16 |
| Fine-tuning / LoRA / PEFT | Day 17 |
| MLflow / experiment tracking | Day 18 |
| MLOps pipelines | Day 19 |
| AI observability / RAGAS | Day 20 |
| Prompt engineering | Day 22 |
| Guardrails / safety | Day 23 |
| Structured outputs | Day 24 |
| Auth / multi-tenancy | Day 25 |
| Cloud AI (Bedrock / Vertex / Azure OpenAI) | Day 26 |
| Docker / Kubernetes | Day 27 |
| RLHF / DPO | Day 28 |
| Copilot architecture | Day 29 |

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS (no frameworks, no build tools) |
| Backend | FastAPI + Python |
| AI | OpenRouter API (multi-provider: Claude, GPT-4, Gemini, etc.) |
| Animation | Canvas 2D API (`bg-animation.js`) |
| Progress | `localStorage` (no server, no auth required) |

## Running the AI Coach

1. Get a free API key from [openrouter.ai](https://openrouter.ai)
2. Set the key — either export it or drop it in a `.env` file (auto-loaded by `python-dotenv`):
   ```bash
   # Option A: shell export
   export OPENROUTER_API_KEY=sk-or-your-key-here
   # Option B: .env file in the project root
   echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
   ```
3. Install dependencies and start the server:
   ```bash
   pip install fastapi uvicorn httpx python-dotenv
   uvicorn server:app --reload --port 8000
   ```
4. Open `ai.html` in your browser — the coach connects automatically to `localhost:8000`

The server tries free models in order: `google/gemma-3-27b-it:free` → `gemma-3-12b-it:free` → `gemma-3-4b-it:free` → `gemma-3n-e4b-it:free` → `arcee-ai/trinity-mini:free`. The first available model wins. You can override the model per request via the `model` field in the JSON body.

**Troubleshooting:**

- **Port conflict**: Change `--port 8000` to another port and update the `BASE_URL` constant in `ai.html`
- **CORS errors**: The server includes `CORSMiddleware` allowing all origins; check that `server.py` is running
- **API key errors**: Verify `OPENROUTER_API_KEY` is set — check both the shell environment and the `.env` file
- **Model not available**: The fallback chain handles this automatically; or edit `FALLBACK_MODELS` in `server.py` to add/swap models

## Beyond Day 30 — "What's Next" Deep Dives

The `next/` folder contains 11 standalone topic pages for engineers who want to go deeper after completing the 30-day curriculum:

| Page | Topic |
|------|-------|
| `next/a2a.html` | Agent-to-Agent (A2A) Protocol |
| `next/mcp.html` | Model Context Protocol (MCP) |
| `next/computer_use.html` | Computer Use API |
| `next/evals.html` | Agent Evaluation at Scale |
| `next/finetune.html` | Fine-tuning & Model Customization |
| `next/constitutional.html` | Constitutional AI & Alignment |
| `next/agent_frameworks.html` | Agentic AI Frameworks Landscape |
| `next/llmstack.html` | Building Your Own LLM Stack |
| `next/mixture.html` | Mixture-of-Agents |
| `next/multimodal.html` | Multimodal Agents |
| `next/neural_networks.html` | Neural Networks Deep Dive |

These pages follow the same no-build, open-in-browser format as the day lessons.

## Progress System

Progress is stored in `localStorage` with no backend dependency.

**Day completion state:**

```
localStorage key: aai_progress
Format: { "1": "complete", "2": "active", "3": "locked", ... }
```

**Week collapse state:**

```
localStorage key: aai_week_collapsed
Format: { "1": false, "2": true, ... }   // true = collapsed
```

**Exercise completion:**

```
localStorage key: aai_exercises
Format: { "day1_ex1": true, "day1_ex2": true, ... }
```

To reset all progress, use the "Reset All Progress" button on the exercises dashboard, or run in the browser console:

```js
Object.keys(localStorage).filter(k => k.startsWith('aai_')).forEach(k => localStorage.removeItem(k));
location.reload();
```

## Contributing / Extending

**Adding a new day:**

1. Copy `days/day7.html` as a template
2. Update the `<title>`, hero section, and content sections
3. Update the `← Prev` / `Next →` nav links in the new file and in the adjacent day files
4. Add an entry to the `DAYS` array in `index.html`

**Adding glossary terms:**

In `glossary.html`, add a term card with the correct `data-category` attribute:

```html
<div class="term-card" data-category="frameworks">
  <h3 class="term-name">Your Term</h3>
  <p class="term-def">Definition here.</p>
</div>
```

**Adding Q&A:**

In `interview.html`, add to the appropriate section:

```html
<div class="qa-card">
  <div class="qa-question">Q: Your question?</div>
  <div class="qa-answer">A: Your answer.</div>
</div>
```

## License

MIT — use freely for personal study, team onboarding, or as a template for your own curriculum.
