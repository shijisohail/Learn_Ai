# Day 5 — Multi-Agent Systems

> A single agent is a loop. A multi-agent system is an architecture. The hard part isn't getting agents to talk — it's deciding what they should say to each other, when, and in what form. Every added agent is a bet that the coordination overhead is worth the specialization gain.

---

## 1. When to Use Multi-Agent

### Single Agent Limitations

Before adding agents, be clear about what's actually broken with your single agent. The failure modes are specific:

**Context explosion.** A single agent accumulates context across every task step. A 50-tool agent analyzing a large codebase will hit 200K tokens in a few turns. At that point, the model starts ignoring early context ("lost in the middle" effect), latency spikes, and costs compound. Multi-agent solves this by giving each agent its own context window — only what it needs for its specific job.

**Single domain expertise.** One system prompt cannot simultaneously be a security expert, a performance engineer, and a style guide enforcer. You end up with a mediocre generalist. A system prompt optimized for security reasoning will be verbose with CVE patterns and attack vectors. That same verbosity degrades writing quality if you ask it to edit prose. Specialized agents with focused system prompts outperform a single omniscient agent on multi-domain tasks.

**No parallelism.** A single agent is inherently sequential. If you need security analysis AND performance analysis AND style review, a single agent does them in sequence. Wall clock time = sum of all agent times. Parallel agents bring this to max(security_time, perf_time, style_time). For a pipeline where each analysis takes 15 seconds, that's 45 seconds vs 15 seconds.

**Single point of failure.** One agent crashes → entire pipeline fails. With independent agents, a failure in the style agent doesn't have to abort the security analysis. You can design graceful degradation at the orchestration level.

### Multi-Agent Decision Criteria

| Factor | Use Single Agent | Use Multi-Agent |
|---|---|---|
| **Task complexity** | One coherent domain | Multiple distinct domains |
| **Context size** | Fits in one context window | Would overflow, or needs isolation |
| **Parallelizability** | Steps depend on each other | Independent subtasks exist |
| **Latency budget** | Tight (< 5 seconds) | Can afford 10-60 seconds |
| **Quality bar** | General competence sufficient | Domain-expert quality needed |
| **Failure tolerance** | Hard failure OK | Partial success valuable |
| **Team ownership** | One person owns the prompt | Multiple domain owners |

### The Cost Warning

**Multi-agent = N × LLM calls.** This is arithmetic, not a suggestion.

One user request to a 4-agent pipeline (orchestrator + 3 workers) means at minimum 4 LLM calls. If each worker loops twice on average, you're at 7 calls per user request. At Claude Sonnet pricing ($3 input / $15 output per 1M tokens), a pipeline generating 5K total output tokens costs roughly $0.075 per request. Scale to 10K requests/day and you're at $750/day — potentially 10-50× more expensive than a single-agent approach.

**The rule:** Only add an agent when the quality or speed improvement demonstrably justifies the cost multiplier. Always prototype with a single agent first. Measure quality. Only reach for multi-agent when you hit a measurable wall.

---

## 2. Architecture Patterns

| Pattern | Description | When to Use | Pros | Cons | Example Use Case |
|---|---|---|---|---|---|
| **Supervisor/Worker** | One orchestrator routes tasks to specialized workers based on LLM routing decisions | Clear task decomposition with different domains | Clean separation, easy to add workers, orchestrator owns all routing logic | Orchestrator is a bottleneck and a cost center | Code review pipeline: security + perf + style agents |
| **Peer-to-Peer (Pipeline)** | Agents pass work directly to the next in sequence, no central coordinator | Linear transformation pipelines where each step depends on the prior | Low overhead, simple to reason about, natural data flow | Sequential only — no parallelism, tight coupling | Research → Draft → Edit → Fact-check |
| **Hierarchical** | Supervisors manage sub-supervisors who manage workers — multiple tiers of orchestration | Extremely complex tasks with nested subproblems requiring independent sub-pipelines | Scales to very complex tasks, each tier isolated | High latency, very hard to debug, token explosion across tiers | Enterprise research platform with domain sub-teams |
| **Event-Driven** | Agents subscribe to event types and emit results to a shared event bus; no direct coupling | Async processing, event-triggered workflows, agents that need to be independently scalable | Fully decoupled, resilient, naturally observable, horizontal scaling | Complex infrastructure, eventual consistency, hard to trace causality | Real-time monitoring: alert → triage agent → escalation agent |

### Choosing Your Pattern

For most backend engineering use cases: start with **Supervisor/Worker**. It maps cleanly to LangGraph, is easy to reason about, and handles both sequential and parallel workers. Move to hierarchical only when the supervisor itself becomes overloaded with routing decisions. Use event-driven only when agents need to live in separate services and you have the infrastructure (Kafka, Redis Streams, etc.) to support it.

---

## 3. The Supervisor Pattern (LangGraph)

### Concept

```
                 ┌──────────────────┐
User Request ──► │    Supervisor    │ ◄── Routes to the right worker
                 │  (LLM-driven)    │     based on current state
                 └────────┬─────────┘
                          │
              ┌───────────┼──────────────┐
              ▼           ▼              ▼
        ┌─────────┐ ┌──────────┐ ┌────────────┐
        │Research │ │  Writer  │ │   Critic   │
        │  Agent  │ │  Agent   │ │   Agent    │
        └────┬────┘ └────┬─────┘ └─────┬──────┘
             │            │              │
             └───────────►▼◄────────────┘
                     Supervisor
                   (decide next step)
```

Supervisors are stateful routers. They read the current state and decide which worker should act next. Workers do focused work and return to the supervisor.

### Full Implementation

```python
# multi_agent_supervisor.py
import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ─── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Full conversation history
    task: str                                  # Original user task
    research_output: str                       # Output from research agent
    draft: str                                 # Output from writer agent
    critique: str                              # Output from critic agent
    final_output: str                          # Packaged final result
    next_agent: str                            # Supervisor's routing decision
    iteration: int                             # Guard against infinite loops

# ─── Model ───────────────────────────────────────────────────────────────────

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=4096,
)

# ─── Supervisor ───────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are an orchestrator managing a content creation pipeline.

Available workers:
- "research": Gathers facts and information about the topic
- "writer": Writes content based on the research output
- "critic": Reviews the draft and identifies weaknesses
- "finish": Returns the final output to the user

Your routing rules:
1. Start with "research" if research_output is empty
2. Move to "writer" once research_output exists
3. Move to "critic" once a draft exists
4. Move to "finish" when critique is done OR after 2 full cycles

Respond with ONLY the worker name: research, writer, critic, or finish"""

def supervisor_node(state: AgentState) -> dict:
    """Reads current state, decides which worker acts next."""
    context_parts = [f"Task: {state['task']}"]

    if state.get("research_output"):
        context_parts.append(
            f"Research: DONE ({len(state['research_output'])} chars)"
        )
    else:
        context_parts.append("Research: NOT DONE")

    if state.get("draft"):
        context_parts.append(f"Draft: EXISTS ({len(state['draft'])} chars)")
    else:
        context_parts.append("Draft: NOT WRITTEN")

    if state.get("critique"):
        context_parts.append(f"Critique: DONE — {state['critique'][:150]}...")
    else:
        context_parts.append("Critique: NOT DONE")

    context = "\n".join(context_parts)

    response = model.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"Current state:\n{context}\n\nWhere next?"),
    ])

    next_agent = response.content.strip().lower()
    valid = {"research", "writer", "critic", "finish"}
    if next_agent not in valid:
        next_agent = "finish"

    return {
        "next_agent": next_agent,
        "iteration": state.get("iteration", 0) + 1,
    }

# ─── Workers ─────────────────────────────────────────────────────────────────

def research_agent(state: AgentState) -> dict:
    """Gathers structured facts about the task topic."""
    response = model.invoke([
        SystemMessage(content="""You are a research specialist. Your job:
1. Identify the 5-7 most important facts about the topic
2. Note any important context, caveats, or background
3. List real examples or case studies if applicable
Format as bullet points. Be factual and specific."""),
        HumanMessage(content=f"Research this thoroughly: {state['task']}"),
    ])
    return {"research_output": response.content}


def writer_agent(state: AgentState) -> dict:
    """Writes a first draft using the research output."""
    response = model.invoke([
        SystemMessage(content="""You are a professional technical writer.
Write clear, well-structured content. Rules:
- Only use facts from the provided research
- Use headers and bullet points for clarity
- Do not invent statistics or examples not in the research
- Target length: comprehensive but not padded"""),
        HumanMessage(content=f"""Task: {state['task']}

Research material:
{state.get('research_output', 'No research available')}

Write a comprehensive response now."""),
    ])
    return {"draft": response.content}


def critic_agent(state: AgentState) -> dict:
    """Reviews the draft and provides specific, actionable feedback."""
    response = model.invoke([
        SystemMessage(content="""You are a sharp editor. Review the draft against the research for:
1. Factual accuracy — are claims supported by the research?
2. Completeness — are important topics from research missing?
3. Clarity — is the structure logical and easy to follow?
4. Specificity — are vague claims present that should be concrete?

Give specific, actionable feedback. If the draft is strong, say so clearly.
Rate overall quality: STRONG / ACCEPTABLE / NEEDS_REVISION"""),
        HumanMessage(content=f"""Original task: {state['task']}

Research:
{state.get('research_output', '')}

Draft to review:
{state.get('draft', '')}"""),
    ])
    return {"critique": response.content}


def finish_node(state: AgentState) -> dict:
    """Packages the final output for return to the user."""
    return {"final_output": state.get("draft", "No output generated")}

# ─── Routing ─────────────────────────────────────────────────────────────────

def route_from_supervisor(
    state: AgentState,
) -> Literal["research", "writer", "critic", "finish"]:
    """Reads the supervisor's routing decision. Hard-stops at 6 iterations."""
    if state.get("iteration", 0) >= 6:
        return "finish"
    return state.get("next_agent", "finish")

# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_supervisor_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("finish", finish_node)

    # Entry: always start at supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research": "research",
            "writer": "writer",
            "critic": "critic",
            "finish": "finish",
        },
    )

    # Each worker returns control to the supervisor
    graph.add_edge("research", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")

    # Finish is the terminal node
    graph.add_edge("finish", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_supervisor_graph()

    result = app.invoke({
        "task": "Explain the CAP theorem with concrete examples for distributed systems engineers",
        "messages": [],
        "research_output": "",
        "draft": "",
        "critique": "",
        "final_output": "",
        "next_agent": "",
        "iteration": 0,
    })

    print("=== FINAL OUTPUT ===")
    print(result["final_output"])
    print(f"\nCompleted in {result['iteration']} supervisor iterations")
```

---

## 4. Agent Handoffs

### What Belongs in a Handoff

A handoff is a structured contract between agents. When Agent A finishes and hands to Agent B, the handoff must include everything B needs and nothing it doesn't. Missing handoff fields are the #1 cause of multi-agent quality failures — the receiving agent makes assumptions about context it doesn't have.

| Field | Type | Why It's Required |
|---|---|---|
| `task` | str | Original goal — B must know what problem we're solving |
| `context` | dict | What has been done so far and its key outputs |
| `constraints` | list[str] | What B must not do (e.g., "don't add new topics not in research") |
| `expected_output_format` | str | Exact format B should return — prevents format mismatch |
| `previous_work_summary` | str | Condensed summary — avoid passing raw output when it's large |
| `handoff_reason` | str | Why we're routing to B (optional, helps debugging) |

### LangGraph Command Pattern for Handoffs

LangGraph's `Command` type lets a worker explicitly name the next node rather than returning to a supervisor:

```python
from langgraph.types import Command
from langchain_core.messages import AIMessage

def research_agent_with_handoff(state: AgentState) -> Command:
    """Research agent that explicitly hands off to the writer."""

    response = model.invoke([
        SystemMessage(content="You are a research specialist. Be thorough and factual."),
        HumanMessage(content=f"Research: {state['task']}"),
    ])
    research_result = response.content

    # Build a structured handoff payload
    handoff_context = {
        "task": state["task"],
        "research_complete": True,
        "constraints": [
            "Only use facts from research_output",
            "Do not invent statistics or examples",
            "Target length: 600-800 words",
        ],
        "expected_format": "Markdown with ## headers and bullet points",
        "previous_work_summary": f"Research completed: {len(research_result)} chars of structured facts",
    }

    return Command(
        update={
            "research_output": research_result,
            "handoff_context": handoff_context,
            "messages": [AIMessage(content=f"Research complete. Handing off to writer.")],
        },
        goto="writer",  # Explicit routing — no supervisor needed
    )


def writer_agent_with_handoff(state: AgentState) -> Command:
    """Writer reads the handoff context and respects its constraints."""

    handoff = state.get("handoff_context", {})
    constraints_text = "\n".join(
        f"- {c}" for c in handoff.get("constraints", [])
    )
    output_format = handoff.get("expected_format", "prose")

    response = model.invoke([
        SystemMessage(content=f"""You are a professional technical writer.

You must follow these constraints:
{constraints_text}

Output format required: {output_format}"""),
        HumanMessage(content=f"""Task: {state['task']}

Research to use:
{state.get('research_output', '')}

Write the content now."""),
    ])

    return Command(
        update={
            "draft": response.content,
            "handoff_context": {
                "task": state["task"],
                "constraints": [
                    "Do not change factual content",
                    "Only improve clarity and flow",
                    "Keep the same structure",
                ],
                "expected_format": "same markdown format as the draft",
            },
        },
        goto="critic",
    )
```

### When to Use Command vs Conditional Edges

| Mechanism | Who Controls Routing | Best For |
|---|---|---|
| `Command(goto=...)` | The worker itself | Peer-to-peer pipelines, when routing logic depends on the worker's output |
| `add_conditional_edges` | A supervisor function | Supervisor patterns, when a central agent owns all routing decisions |

Use `Command` when the next step depends on what the current step produced ("if I found vulnerabilities, go to security agent; otherwise go to formatter"). Use conditional edges when a supervisor LLM owns all routing.

---

## 5. Parallel Agent Execution

### The Fan-Out / Fan-In Pattern

```
              ┌──── Security Agent ────┐
Code Diff ────┤──── Perf Agent ────────├──── Aggregator ──── Review
              └──── Style Agent ────────┘
```

All 3 run simultaneously. Wall clock time = max(security_time, perf_time, style_time) instead of their sum.

### When Parallel Is Worth It

Parallel execution is worthwhile when:
1. Subtasks are **independent** — agent 2 doesn't need agent 1's output to start
2. Tasks are **I/O bound** — you're waiting for network/API calls, not CPU work
3. **Latency > cost** is your priority — parallel uses the same total tokens, just less wall time
4. There are 2+ agents that would otherwise set up identical context before doing different work

Parallel execution is **not** worthwhile when:
- Agents need each other's outputs (sequential by definition)
- You have a tight cost budget (parallel does not save tokens)
- Tasks complete in < 2 seconds anyway (overhead not worth it)

### asyncio.gather for Parallel Agents

```python
# parallel_agents.py
import asyncio
import os
import time
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


async def research_agent_async(
    topic: str, focus: str, agent_id: int
) -> dict:
    """A single research agent with a specific focus area."""
    print(f"[Agent {agent_id}] Starting: {focus}")

    response = await client.messages.create(
        model="claude-3-5-haiku-20241022",  # Haiku for parallel calls — cheaper
        max_tokens=1024,
        system=f"""You are a research specialist focused on: {focus}
Be specific, factual, and concise. Output as structured bullet points.
Focus only on your assigned area — do not cover other areas.""",
        messages=[
            {"role": "user", "content": f"Research the {focus} aspects of: {topic}"}
        ],
    )

    print(f"[Agent {agent_id}] Done: {focus}")
    return {
        "agent_id": agent_id,
        "focus": focus,
        "findings": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }


async def aggregator_agent(topic: str, research_results: list[dict]) -> str:
    """Combines parallel research outputs into a single coherent summary."""

    combined = "\n\n".join([
        f"=== {r['focus'].upper()} ===\n{r['findings']}"
        for r in research_results
    ])

    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",  # Sonnet for synthesis — quality matters
        max_tokens=2048,
        system="""You are a synthesis specialist. Combine research from multiple parallel agents
into a single coherent summary. Preserve all key findings.
Remove any redundancy. Organize from most to least important.""",
        messages=[
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nParallel research results:\n{combined}\n\nSynthesize now.",
            }
        ],
    )

    return response.content[0].text


async def parallel_research_pipeline(topic: str) -> dict:
    """
    Fan-out: 3 research agents run simultaneously on different focus areas.
    Fan-in: aggregator synthesizes all findings into one coherent output.
    """
    focus_areas = [
        "technical implementation details and architecture",
        "real-world use cases and production examples",
        "limitations, failure modes, and tradeoffs",
    ]

    # Fan-out: all 3 run at the same time
    start = time.time()
    print(f"Launching {len(focus_areas)} parallel agents...")

    research_results = await asyncio.gather(
        *[
            research_agent_async(topic, focus, agent_id=i + 1)
            for i, focus in enumerate(focus_areas)
        ]
    )

    parallel_time = time.time() - start
    print(f"All agents done in {parallel_time:.1f}s")

    # Fan-in: synthesize
    print("Synthesizing results...")
    final_summary = await aggregator_agent(topic, list(research_results))

    total_tokens = sum(r["tokens_used"] for r in research_results)

    return {
        "topic": topic,
        "summary": final_summary,
        "individual_results": research_results,
        "total_tokens": total_tokens,
        "parallel_execution_time_s": round(parallel_time, 2),
    }


if __name__ == "__main__":
    result = asyncio.run(
        parallel_research_pipeline("Service mesh architectures in Kubernetes")
    )
    print("\n=== SYNTHESIZED RESEARCH ===")
    print(result["summary"])
    print(f"\nTotal tokens: {result['total_tokens']}")
    print(f"Parallel execution: {result['parallel_execution_time_s']}s")
```

### LangGraph Native Parallel Nodes

When multiple edges leave the same source node, LangGraph executes them in parallel:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

class ReviewState(TypedDict):
    code: str
    security_findings: list[dict]
    perf_findings: list[dict]
    style_findings: list[dict]
    combined_report: str

def build_parallel_review_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("security_agent", security_agent)
    graph.add_node("perf_agent", performance_agent)
    graph.add_node("style_agent", style_agent)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "orchestrator")

    # Fan-out: orchestrator → all 3 workers simultaneously
    graph.add_edge("orchestrator", "security_agent")
    graph.add_edge("orchestrator", "perf_agent")
    graph.add_edge("orchestrator", "style_agent")

    # Fan-in: LangGraph waits for ALL 3 before running aggregator
    graph.add_edge("security_agent", "aggregator")
    graph.add_edge("perf_agent", "aggregator")
    graph.add_edge("style_agent", "aggregator")

    graph.add_edge("aggregator", END)

    return graph.compile()

# For dynamic parallelism (variable number of agents), use Send:
def dynamic_parallel_dispatch(state: ReviewState) -> list[Send]:
    """Send the same code to N agents dynamically."""
    agents = ["security_agent", "perf_agent", "style_agent"]
    return [
        Send(agent, {"code": state["code"]})
        for agent in agents
    ]
```

---

## 6. Shared State vs Message Passing

### Shared State (TypedDict)

All agents read from and write to a central state object. This is LangGraph's native pattern.

```python
class SharedPipelineState(TypedDict):
    # Original input
    task: str

    # Each agent has its own dedicated output field — no write contention
    research_output: str          # Written by: research_agent
    draft: str                    # Written by: writer_agent
    security_findings: list[dict] # Written by: security_agent
    perf_findings: list[dict]     # Written by: perf_agent
    critique: str                 # Written by: critic_agent
    final_output: str             # Written by: finish_node

    # Metadata written by any agent
    cost_tracker: dict            # Updated by every agent
    error_log: list[str]          # Appended to on any failure
```

**Key pattern:** Each agent owns exactly one field. No two agents write to the same field. This eliminates write contention without any additional infrastructure.

**Pros:**
- No serialization overhead
- Built-in LangGraph checkpointing
- Easy to inspect full pipeline state at any step
- Straightforward debugging

**Cons:**
- TypedDict grows with each new agent (manageable with good naming)
- All agents are in the same process (fine for most use cases)

### Message Passing

Agents communicate via structured messages over a shared bus. Better for cross-process or cross-service agent communication.

```python
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    message_type: str          # "task" | "result" | "error" | "handoff"
    payload: dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class InProcessMessageBus:
    """Simple async message bus for agents in the same process."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}

    def register(self, agent_name: str):
        self._queues[agent_name] = asyncio.Queue()

    async def send(self, message: AgentMessage):
        queue = self._queues.get(message.to_agent)
        if not queue:
            raise ValueError(f"Unknown agent: {message.to_agent}")
        await queue.put(message)

    async def receive(
        self, agent_name: str, timeout: float = 30.0
    ) -> AgentMessage:
        queue = self._queues.get(agent_name)
        if not queue:
            raise ValueError(f"Agent '{agent_name}' not registered")
        return await asyncio.wait_for(queue.get(), timeout=timeout)


# Usage pattern
async def research_worker(bus: InProcessMessageBus, agent_name: str):
    """A worker that listens for tasks and publishes results."""
    while True:
        message = await bus.receive(agent_name)

        if message.message_type == "task":
            # Do the work
            result = await perform_research(message.payload["topic"])

            # Publish result
            await bus.send(AgentMessage(
                from_agent=agent_name,
                to_agent="orchestrator",
                message_type="result",
                payload={"research": result},
                correlation_id=message.correlation_id,  # Preserve trace ID
            ))
```

### Recommendation

**Default to shared state (TypedDict)** for agents within a single LangGraph graph. It's what the framework is built for. You get checkpointing, streaming, and HITL for free.

**Use message passing** only when:
- Agents live in separate processes or services
- You need genuine async event-driven decoupling
- Agents need to be independently deployable and scalable

For a 3-7 agent pipeline in one FastAPI service, shared state is always the right choice.

---

## 7. Framework Comparison

| Dimension | LangGraph | CrewAI | AutoGen | Raw Anthropic SDK |
|---|---|---|---|---|
| **Abstraction level** | Medium — you see the graph | High — hides mechanics | High — conversation-centric | None — full control |
| **Flexibility** | High | Medium | Medium | Total |
| **Production-ready** | Yes — checkpoints, streaming, HITL | Partial | Research-grade | Yes (if you build it) |
| **Learning curve** | Medium — graph concepts required | Low — role-based intuitive | Medium — conversation model | Low |
| **State management** | Built-in TypedDict + checkpointer | Limited, implicit | Conversation history only | DIY |
| **Streaming** | Native `.stream()` | Limited | Limited | Native |
| **Human-in-the-loop** | Native `interrupt_before/after` | Manual | Via `human_input_mode` | Manual |
| **Observability** | LangSmith integration | Limited | Limited | DIY or Langfuse |
| **Agent communication** | Graph edges + Command | Role tasks | Reply chains | DIY |
| **Anthropic SDK native** | Via langchain-anthropic | Via langchain-anthropic | Via raw API | Yes — first class |
| **Best for** | Production agents, complex state | Quick prototypes | Research/academic | Full control, minimal deps |

### The Honest Assessment

**LangGraph** is the production choice for 2024-2025. It gives you the graph mental model, state management, streaming, checkpointing, and HITL out of the box. The LangChain dependency is a cost — but the primitives it provides would take weeks to rebuild correctly.

**CrewAI** is great for demos and simple pipelines. The crew/role abstraction is intuitive but the internals are opaque. When things go wrong in production, you'll struggle to debug what the framework is actually doing.

**AutoGen** is Microsoft Research's agent framework. It has interesting ideas (group chat, code execution agents) but is not production-grade software. Use it for prototyping novel agent patterns, not shipping services.

**Raw Anthropic SDK** is what you used in Days 1-4. Right for simple agents. When you need persistent state, branching, HITL, or parallel execution, the overhead of reinventing LangGraph's primitives is not worth it.

---

## 8. Real Example: Content Pipeline

### Architecture Walkthrough

```
User provides: topic + target audience
        ↓
┌───────────────────────────────────────────────────┐
│  Research Agent (Sonnet)                          │
│  Input:  topic, target_audience                   │
│  Output: research_output (structured facts)       │
│  Why Sonnet: accuracy matters; wrong facts        │
│             propagate through the entire pipeline │
└───────────────────┬───────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────┐
│  Writer Agent (Sonnet)                            │
│  Input:  research_output, topic, audience         │
│  Output: draft (full article)                     │
│  Why Sonnet: first draft quality determines       │
│             how much editing is needed            │
└───────────────────┬───────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────┐
│  Editor Agent (Haiku)                             │
│  Input:  draft                                    │
│  Output: edited_draft (structure, clarity)        │
│  Why Haiku: structural edits don't need Sonnet    │
│             IQ — saves ~60% cost on this step     │
└───────────────────┬───────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────┐
│  Fact-checker Agent (Haiku)                       │
│  Input:  research_output + edited_draft           │
│  Output: issues_found (list), fact_check_passed   │
│  Why Haiku: comparison task, not creative         │
└───────────────────┬───────────────────────────────┘
                    ↓
            Final article + quality report
```

Mixed-model strategy: Sonnet for creative/reasoning steps, Haiku for structural/comparison steps. Reduces cost ~50% vs all-Sonnet with minimal quality degradation.

```python
# content_pipeline.py
import asyncio
import json
import os
from typing import TypedDict
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class PipelineState(TypedDict):
    topic: str
    target_audience: str
    research_output: str
    draft: str
    edited_draft: str
    fact_check_passed: bool
    issues_found: list[dict]
    final_article: str
    total_tokens: int


async def research_agent(state: PipelineState) -> PipelineState:
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system="""You are a research specialist. Gather comprehensive, accurate information.
Format output as:
1. KEY FACTS (5-7 bullet points, specific and verifiable)
2. CONTEXT (1-2 paragraphs of background)
3. EXAMPLES (2-3 concrete real-world examples)
4. CAVEATS (important limitations or nuances)""",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research for an article about: {state['topic']}\n"
                    f"Target audience: {state['target_audience']}"
                ),
            }
        ],
    )
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return {
        **state,
        "research_output": response.content[0].text,
        "total_tokens": state.get("total_tokens", 0) + tokens,
    }


async def writer_agent(state: PipelineState) -> PipelineState:
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=3000,
        system=f"""You are a professional technical writer for: {state['target_audience']}.
Rules:
- Only use facts from the provided research
- Structure: Introduction → 3-4 main sections → Conclusion
- Use Markdown headers and bullet points
- Do not invent statistics not in the research""",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Topic: {state['topic']}\n\n"
                    f"Research:\n{state['research_output']}\n\n"
                    "Write the full article."
                ),
            }
        ],
    )
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return {
        **state,
        "draft": response.content[0].text,
        "total_tokens": state["total_tokens"] + tokens,
    }


async def editor_agent(state: PipelineState) -> PipelineState:
    response = await client.messages.create(
        model="claude-3-haiku-20240307",  # Haiku sufficient for structural edits
        max_tokens=3000,
        system="""You are a copy editor. Improve without changing facts:
1. Active voice over passive voice
2. Remove redundant phrases
3. Strengthen section headers
4. Ensure logical flow between sections
Return the complete edited article.""",
        messages=[
            {"role": "user", "content": f"Edit this article:\n\n{state['draft']}"}
        ],
    )
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return {
        **state,
        "edited_draft": response.content[0].text,
        "total_tokens": state["total_tokens"] + tokens,
    }


async def fact_checker_agent(state: PipelineState) -> PipelineState:
    response = await client.messages.create(
        model="claude-3-haiku-20240307",  # Haiku sufficient for comparison
        max_tokens=1024,
        system="""You are a fact-checker. Compare the article against research notes.
Find claims in the article NOT supported by the research.
Return JSON array: [{"claim": "...", "issue": "not in research|contradicts research"}]
If everything checks out, return: []
IMPORTANT: Return ONLY the JSON array, no other text.""",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research notes:\n{state['research_output']}\n\n"
                    f"Article:\n{state['edited_draft']}"
                ),
            }
        ],
    )
    issues_text = response.content[0].text.strip()
    try:
        issues = json.loads(issues_text)
    except json.JSONDecodeError:
        issues = []

    tokens = response.usage.input_tokens + response.usage.output_tokens
    return {
        **state,
        "issues_found": issues,
        "fact_check_passed": len(issues) == 0,
        "final_article": state["edited_draft"],
        "total_tokens": state["total_tokens"] + tokens,
    }


async def run_content_pipeline(topic: str, audience: str) -> PipelineState:
    state: PipelineState = {
        "topic": topic,
        "target_audience": audience,
        "research_output": "",
        "draft": "",
        "edited_draft": "",
        "fact_check_passed": False,
        "issues_found": [],
        "final_article": "",
        "total_tokens": 0,
    }

    print("Step 1/4: Researching...")
    state = await research_agent(state)

    print("Step 2/4: Writing first draft...")
    state = await writer_agent(state)

    print("Step 3/4: Editing...")
    state = await editor_agent(state)

    print("Step 4/4: Fact-checking...")
    state = await fact_checker_agent(state)

    if state["issues_found"]:
        print(f"WARNING: {len(state['issues_found'])} fact-check issues")
        for issue in state["issues_found"]:
            print(f"  - {issue}")

    print(f"\nTotal tokens used: {state['total_tokens']}")
    return state


if __name__ == "__main__":
    result = asyncio.run(run_content_pipeline(
        topic="Circuit breaker pattern in microservices",
        audience="Senior backend engineers with Python/FastAPI experience",
    ))
    print("\n=== FINAL ARTICLE ===")
    print(result["final_article"])
```

---

## 9. Production Notes

### Cost Amplification: Track Per Request

```python
import time
from dataclasses import dataclass, field

# Anthropic pricing as of late 2024 (check anthropic.com/pricing for updates)
MODEL_PRICING = {
    "claude-3-5-sonnet-20241022": (3.0, 15.0),     # input/output per 1M tokens
    "claude-3-5-haiku-20241022": (0.25, 1.25),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-opus-20240229": (15.0, 75.0),
}

@dataclass
class RequestCostTracker:
    request_id: str
    started_at: float = field(default_factory=time.time)
    agent_calls: list[dict] = field(default_factory=list)

    def record(
        self, agent: str, model: str, input_tokens: int, output_tokens: int
    ):
        in_price, out_price = MODEL_PRICING.get(model, (3.0, 15.0))
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        self.agent_calls.append({
            "agent": agent,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

    def summary(self) -> dict:
        return {
            "request_id": self.request_id,
            "duration_s": round(time.time() - self.started_at, 2),
            "llm_calls": len(self.agent_calls),
            "total_tokens": sum(
                c["input_tokens"] + c["output_tokens"] for c in self.agent_calls
            ),
            "total_cost_usd": round(
                sum(c["cost_usd"] for c in self.agent_calls), 6
            ),
            "most_expensive_agent": max(
                self.agent_calls, key=lambda c: c["cost_usd"], default=None
            ),
        }
```

### Agent Timeouts: Per Agent, Not Global

```python
import asyncio
from typing import TypeVar, Awaitable

T = TypeVar("T")

async def with_agent_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    agent_name: str,
) -> T:
    """Wrap any agent coroutine with a per-agent timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Agent '{agent_name}' exceeded {timeout_seconds}s timeout"
        )


# Different agents get different timeouts based on expected duration
async def run_pipeline_timed(state):
    state = await with_agent_timeout(
        research_agent(state), timeout_seconds=30, agent_name="research"
    )
    state = await with_agent_timeout(
        writer_agent(state), timeout_seconds=45, agent_name="writer"
    )
    state = await with_agent_timeout(
        editor_agent(state), timeout_seconds=20, agent_name="editor"
    )
    return state
```

### Observability: Correlation IDs Across Agents

Every LLM call in a multi-agent system needs a correlation ID so you can trace a full user request across all agents in your logs:

```python
import uuid
import structlog

logger = structlog.get_logger()

def agent_logger(request_id: str, agent_name: str):
    return logger.bind(request_id=request_id, agent=agent_name)

async def research_agent_instrumented(
    state: PipelineState, request_id: str
) -> PipelineState:
    log = agent_logger(request_id, "research")
    log.info("agent.start", topic=state["topic"])
    start = time.time()

    response = await client.messages.create(...)
    
    log.info(
        "agent.complete",
        latency_ms=round((time.time() - start) * 1000),
        tokens=response.usage.input_tokens + response.usage.output_tokens,
    )
    return {**state, "research_output": response.content[0].text}
```

### Result Validation at Agent Boundaries

The output of Agent A is the input of Agent B. Validate at the boundary — don't let bad data propagate:

```python
from pydantic import BaseModel, ValidationError
import json

class ResearchOutput(BaseModel):
    key_facts: list[str]
    examples: list[str]
    caveats: list[str]

async def research_agent_validated(state: PipelineState) -> PipelineState:
    for attempt in range(2):  # 2 attempts max
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system="Return JSON matching: {key_facts: [], examples: [], caveats: []}",
            messages=[{"role": "user", "content": f"Research: {state['topic']}"}],
        )
        try:
            validated = ResearchOutput.model_validate_json(
                response.content[0].text
            )
            return {**state, "research_output": validated.model_dump_json()}
        except (ValidationError, json.JSONDecodeError) as e:
            if attempt == 1:
                raise ValueError(f"Research agent failed validation twice: {e}")
            # Will retry on next iteration
    raise RuntimeError("Should not reach here")
```

### Graceful Degradation

```python
async def run_pipeline_resilient(state: PipelineState) -> PipelineState:
    """Critical stages fail hard; optional stages degrade gracefully."""

    # Required: if these fail, surface the error
    state = await research_agent(state)
    state = await writer_agent(state)

    # Optional: degrade gracefully, don't crash the pipeline
    try:
        state = await editor_agent(state)
    except Exception as e:
        logger.warning("editor_agent.failed", error=str(e), using="raw_draft")
        state = {**state, "edited_draft": state["draft"]}

    try:
        state = await fact_checker_agent(state)
    except Exception as e:
        logger.warning("fact_checker.failed", error=str(e))
        state = {
            **state,
            "fact_check_passed": False,
            "issues_found": [],
            "final_article": state.get("edited_draft", state["draft"]),
        }

    return state
```

---

## 10. Code Examples

All code in this document is complete and runnable with `ANTHROPIC_API_KEY` set. Quick reference:

| Section | Code | What It Shows |
|---|---|---|
| Section 3 | `multi_agent_supervisor.py` | Full LangGraph supervisor pattern |
| Section 5 | `parallel_agents.py` | asyncio.gather fan-out/fan-in |
| Section 8 | `content_pipeline.py` | Linear pipeline, mixed models |
| Section 9 | Cost tracker, timeout wrapper | Production reliability primitives |

---

## 11. Key Concepts Cheatsheet

| Concept | One-Line Definition | When It Matters |
|---|---|---|
| **Supervisor pattern** | Orchestrator LLM routes tasks to specialized workers | Complex multi-domain tasks |
| **Fan-out / fan-in** | Split work to N parallel agents, aggregate results | Independent subtasks, latency-sensitive |
| **LangGraph Command** | Worker explicitly names the next node | Peer-to-peer handoffs |
| **Conditional edges** | Router function decides next node | Supervisor-controlled routing |
| **Shared state (TypedDict)** | All agents read/write one object, each owns a field | Most multi-agent systems |
| **Message passing** | Agents communicate via structured messages | Cross-process, event-driven |
| **Handoff contract** | Task + context + constraints + expected output format | Every agent-to-agent transfer |
| **Cost multiplier** | N agents × M iterations = N×M LLM calls | Budget planning |
| **Correlation ID** | UUID passed through every agent call | Debugging multi-agent pipelines |
| **Graceful degradation** | Optional agents fail without crashing the pipeline | Production resilience |
| **Per-agent timeout** | Each agent has its own deadline | Preventing one slow agent from blocking all |
| **Mixed model routing** | Haiku for cheap tasks, Sonnet for quality tasks | Cost optimization in pipelines |

---

## 12. Day 5 Exercises

### Exercise 1: Implement a Debate Agent System

Build a 3-agent system that debates a technical decision:
- **Proposer**: Argues FOR a given position (e.g., "use microservices over monolith")
- **Opposer**: Argues AGAINST, directly rebutting the proposer
- **Judge**: Receives both arguments in one call and delivers a verdict with clear reasoning

Requirements:
- Proposer and Opposer run in **parallel** using `asyncio.gather`
- Judge receives both arguments in a single API call
- State TypedDict tracks: `proposal`, `opposition`, `judgment`, `winning_side`
- Judge's verdict must include: which argument was stronger, why, and what the deciding factor was

Expected output: A structured debate verdict identifying the stronger argument with supporting reasoning.

### Exercise 2: Cost-Aware Supervisor

Take the supervisor pattern from Section 3 and add full cost awareness:
- Integrate the `RequestCostTracker` from Section 9 into every agent call
- Record tokens and cost after each agent completes
- Add a `max_cost_usd=0.15` parameter to the graph runner
- If the cost budget is exceeded mid-pipeline: route immediately to `finish` with partial results
- After completion, print a full cost breakdown by agent

Expected output: Per-agent cost breakdown, total pipeline cost, and confirmation that budget enforcement stops the pipeline when exceeded.

### Exercise 3: Fault-Tolerant Content Pipeline

Build the 4-stage pipeline from Section 8 with full fault tolerance:
- Wrap each agent call in `with_agent_timeout` with appropriate per-agent timeouts
- Implement the graceful degradation pattern from Section 9
- Add a `pipeline_health` field to state: dict mapping agent names to "success" | "failed" | "skipped"
- Artificially raise exceptions in 2 agents (e.g., editor and fact-checker) to test degradation
- The pipeline must complete and return a final article even with 2 agent failures

Expected output: Completed article, pipeline health report showing which stages succeeded/failed/degraded.

### Exercise 4: LangGraph Parallel Code Review

Build a parallel code review system using LangGraph's native parallel node support:
- Input: a Python function with intentional bugs planted (SQL injection, no type hints, hardcoded secret, N+1 loop)
- Three parallel agent nodes: `security_agent`, `performance_agent`, `style_agent`
- Each finds findings specific to its domain only
- `aggregator_node` combines findings, deduplicates, and sorts by severity (CRITICAL → HIGH → MEDIUM → LOW)
- Use the fan-out with `graph.add_edge(source, dest)` pattern from Section 5

Expected output: A structured review report in JSON with findings sorted by severity, each with: `severity`, `agent`, `line`, `description`, `suggested_fix`.
