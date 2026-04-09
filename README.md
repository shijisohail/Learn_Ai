# 30-Day Agentic AI Mastery

![30 days](https://img.shields.io/badge/curriculum-30%20days-7c3aed) ![203 terms](https://img.shields.io/badge/glossary-203%20terms-0d9488) ![116 Q&A](https://img.shields.io/badge/interview-116%20Q%26A-f59e0b) ![No build tools](https://img.shields.io/badge/build-none-10b981)

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
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
uvicorn server:app --reload --port 8000
# Then open index.html or ai.html in your browser
```

The AI Coach provides contextual guidance, concept explanations, and coding help — powered by OpenRouter with SSE streaming.

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
├── interview.html          # 116 Q&A + 20 comparisons + 40 flashcards
├── ai.html                 # AI Coach interface (connects to FastAPI backend)
├── server.py               # FastAPI backend — OpenRouter API, SSE streaming
├── bg-animation.js         # Neural network canvas animation (shared across pages)
│
├── days/                   # 30 individual lesson pages
│   ├── day1.html           # How LLMs Work + The API
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
│   └── ...                 # day2_exercises.html through day30_exercises.html
│
└── weeks/                  # Week overview pages
    ├── week1.html          # Week 1 overview: Core Foundations
    ├── week2.html          # Week 2 overview: Frameworks & Models
    ├── week3.html          # Week 3 overview: ML & Fine-Tuning
    ├── week4.html          # Week 4 overview: Prompt Eng & Cloud
    └── week5.html          # Week 5 overview: Final Capstone
```

## Features

- **30 interactive day lessons** with live widgets, code samples, and concept checks
- **203-term glossary** with category filtering (models, frameworks, concepts, tools)
- **116 Q&A + 20 comparisons + 40 flashcards** for interview preparation
- **AI Coach** powered by FastAPI + OpenRouter with SSE streaming responses
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

1. Get an API key from [openrouter.ai](https://openrouter.ai)
2. Set the environment variable:
   ```bash
   export OPENROUTER_API_KEY=sk-or-your-key-here
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn server:app --reload --port 8000
   ```
4. Open `ai.html` in your browser — the coach connects automatically to `localhost:8000`

**Troubleshooting:**

- **Port conflict**: Change `--port 8000` to another port and update the `BASE_URL` constant in `ai.html`
- **CORS errors**: The server includes `CORSMiddleware` allowing all origins; check that `server.py` is running
- **API key errors**: Verify `OPENROUTER_API_KEY` is set in the same shell session as uvicorn
- **Model not available**: Edit `server.py` to change the `model` parameter to any OpenRouter-supported model

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
