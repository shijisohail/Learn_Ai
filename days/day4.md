# Day 4 — LangGraph: Stateful Agents

> LangGraph is what happens when you take the ReAct loop and make it production-grade. Graph-based orchestration gives you branching, persistence, human-in-the-loop, and streaming — all of which are painful to retrofit onto a while loop.

---

## 1. Why LangGraph?

### Limitations of the Raw While Loop

The `react_loop` from Day 2 breaks down quickly when you need production features:

| Feature | Raw While Loop | LangGraph |
|---|---|---|
| Branching logic | Spaghetti if/else | Declarative conditional edges |
| State persistence | Roll your own | Built-in checkpointers |
| Human-in-the-loop | Hard to interrupt | `interrupt_before`/`interrupt_after` |
| Resumable execution | Not possible | Thread IDs + checkpointing |
| Parallel execution | Manual asyncio | Native parallel nodes |
| Debugging | Print statements | LangSmith tracing |
| Streaming | Manual SSE | `stream()` with event types |
| Error recovery | Try/except + restart | Checkpoint + replay |

> LangGraph is not magic — it's your while loop with state management, persistence, and control flow built in. Everything it does could be done by hand, but you'd rebuild these same abstractions every time.

### When NOT to Use LangGraph

- Simple single-step transformations → raw SDK
- Stateless request/response APIs → raw SDK
- Extremely latency-sensitive paths → raw SDK (LangGraph has overhead)
- Experimental/prototype agents → raw SDK first, migrate when complexity grows

---

## 2. Core Concepts

### The Building Blocks

```
StateGraph
├── State (TypedDict)         — The shared data that flows through the graph
├── Nodes (Python functions)  — Units of work; read/write state
├── Edges                     — Connections between nodes
│   ├── Direct edges          — Always go from A to B
│   └── Conditional edges     — Routing function decides next node
├── Checkpointer              — Persists state after each node execution
└── Compiled Graph            — Executable version; call .invoke() or .stream()
```

### State Flow

```
START
  ↓
[Node A] — reads state, does work, writes to state
  ↓
[Node B] — reads state, calls tools, writes results to state
  ↓
[Router] — conditional edge function; returns node name or END
  ↓
[Node C] or END
```

Every node is just a Python function with signature:
```python
def my_node(state: MyState) -> dict:
    # Read from state: state["messages"], state["some_field"]
    # Do work
    # Return dict of state updates (partial updates, not full replacement)
    return {"some_field": new_value}
```

---

## 3. State Design

### TypedDict for State

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Messages accumulate via add_messages reducer
    # Without the reducer, each node would overwrite the list
    messages: Annotated[list, add_messages]

    # Intermediate results stored explicitly
    search_results: list[dict]
    analysis: Optional[str]

    # Control flow flags
    requires_human_review: bool
    iteration_count: int

    # Metadata
    task_id: str
    error: Optional[str]
```

### State Reducers

Reducers define how state fields accumulate. The `add_messages` reducer is the most important:

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Without reducer (default): each node REPLACES the field
class SimpleState(TypedDict):
    counter: int  # node returning {"counter": 5} replaces current value

# With add_messages reducer: messages ACCUMULATE
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # node returning {"messages": [new_msg]} APPENDS to existing list
```

**Custom reducer example:**

```python
from typing import Annotated

def merge_lists(existing: list, new: list) -> list:
    """Deduplicate while preserving order."""
    seen = set()
    result = []
    for item in existing + new:
        key = item.get("id", str(item))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    sources: Annotated[list[dict], merge_lists]  # Custom reducer
    facts: list[str]   # Overwritten each time (no reducer)
```

### What Belongs in State vs. External Storage

| Data | Where to Store |
|---|---|
| Current conversation | State: `messages` |
| Task inputs/config | State: dedicated fields |
| Intermediate LLM outputs | State: dedicated fields |
| Large files/blobs | External storage; put URL/ID in state |
| Vector DB query results | State (per-run); don't persist |
| Historical data | External DB; retrieve into state when needed |

---

## 4. Building Your First Graph

### Step-by-Step: Research Agent

```python
import anthropic
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

# ---- Step 1: Define State ----

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: list[dict]
    final_report: Optional[str]
    iterations: int

# ---- Step 2: Create Model and Tools ----

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

model = ChatAnthropic(model="claude-sonnet-4-5").bind_tools(tools)

SYSTEM = SystemMessage(content=(
    "You are a research assistant. Use the search tool to find information. "
    "When you have enough information, write a comprehensive report."
))

# ---- Step 3: Define Nodes ----

def call_model(state: ResearchState) -> dict:
    """Main reasoning node: calls the LLM."""
    messages = [SYSTEM] + state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }

def finalize_report(state: ResearchState) -> dict:
    """Extract the final text response as the report."""
    last_message = state["messages"][-1]
    return {"final_report": last_message.content}

# ---- Step 4: Define Routing ----

def should_continue(state: ResearchState) -> str:
    """Route: if last message has tool calls, execute tools; else we're done."""
    last_message = state["messages"][-1]
    iteration_limit = 8

    if state.get("iterations", 0) >= iteration_limit:
        return "finalize"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "finalize"

# ---- Step 5: Build Graph ----

graph_builder = StateGraph(ResearchState)

# Add nodes
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("finalize", finalize_report)

# Add edges
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges(
    "model",
    should_continue,
    {
        "tools": "tools",
        "finalize": "finalize",
    }
)
graph_builder.add_edge("tools", "model")  # After tools, always back to model
graph_builder.add_edge("finalize", END)

# ---- Step 6: Compile ----

graph = graph_builder.compile()

# ---- Step 7: Invoke ----

result = graph.invoke({
    "messages": [HumanMessage(content="What are the main Python web frameworks in 2025?")],
    "search_results": [],
    "final_report": None,
    "iterations": 0,
})

print(result["final_report"])
```

---

## 5. Conditional Routing

### The `should_continue` Pattern

The routing function is the heart of conditional graphs. It reads state and returns a string matching one of your edge labels:

```python
def route_after_model(state: AgentState) -> str:
    last_msg = state["messages"][-1]

    # Check for tool calls first
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        # Could route to different tool executors based on tool type
        tool_names = [tc["name"] for tc in last_msg.tool_calls]
        if any(t.startswith("dangerous_") for t in tool_names):
            return "require_approval"
        return "execute_tools"

    # Check content-based routing
    content = last_msg.content.lower() if isinstance(last_msg.content, str) else ""
    if "insufficient information" in content:
        return "gather_more_info"
    if "error" in content or state.get("error"):
        return "handle_error"

    return "done"

# Register the routing with explicit node mapping
graph.add_conditional_edges(
    "model_node",
    route_after_model,
    {
        "execute_tools": "tool_executor",
        "require_approval": "human_approval",
        "gather_more_info": "research_node",
        "handle_error": "error_handler",
        "done": END,
    }
)
```

### Multiple Branches Example

```python
class TriageState(TypedDict):
    messages: Annotated[list, add_messages]
    ticket_type: Optional[str]  # "bug" | "feature" | "question"
    priority: Optional[str]

def triage_router(state: TriageState) -> str:
    ticket_type = state.get("ticket_type", "")
    priority = state.get("priority", "")

    if ticket_type == "bug" and priority == "P1":
        return "escalate"
    elif ticket_type == "bug":
        return "bug_handler"
    elif ticket_type == "feature":
        return "feature_handler"
    else:
        return "general_support"

graph.add_conditional_edges(
    "triage",
    triage_router,
    {
        "escalate": "incident_manager",
        "bug_handler": "eng_queue",
        "feature_handler": "product_queue",
        "general_support": "support_queue",
    }
)
```

---

## 6. Persistence & Checkpointing

### SQLite Checkpointer (Development)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)

    # Each invocation with a thread_id is persistent
    config = {"configurable": {"thread_id": "user-123-session-456"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="Start a research task")]},
        config=config
    )

    # Later — continue the same thread (agent resumes from last checkpoint)
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Now write the final report")]},
        config=config  # Same thread_id → continues from checkpoint
    )
```

### PostgreSQL Checkpointer (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

db_url = "postgresql://user:pass@localhost:5432/agents"

with PostgresSaver.from_conn_string(db_url) as checkpointer:
    checkpointer.setup()  # Creates checkpoint tables
    graph = graph_builder.compile(checkpointer=checkpointer)
```

### Thread Management

```python
import uuid

class AgentSession:
    """Manages agent threads with lifecycle tracking."""

    def __init__(self, graph, checkpointer_url: str):
        from langgraph.checkpoint.postgres import PostgresSaver
        self._checkpointer = PostgresSaver.from_conn_string(checkpointer_url)
        self._checkpointer.setup()
        self.graph = graph_builder.compile(checkpointer=self._checkpointer)

    def create_thread(self, user_id: str) -> str:
        thread_id = f"{user_id}-{uuid.uuid4().hex[:8]}"
        return thread_id

    def get_thread_state(self, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)

    def list_threads(self, user_id: str) -> list[str]:
        # Implementation depends on your metadata storage
        pass

    def invoke(self, thread_id: str, message: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
```

### What Gets Checkpointed

After every node execution, LangGraph saves:
- The full state at that point
- The node that just ran
- The edge traversal history

This means you can replay, resume, fork, and time-travel through agent execution.

---

## 7. Human-in-the-Loop

### interrupt_before Pattern

```python
from langgraph.types import interrupt

def dangerous_operation_node(state: AgentState) -> dict:
    """Node that requires human approval before executing."""
    proposed_action = state["proposed_action"]

    # This pauses execution and returns control to the caller
    # The caller can inspect state and decide to continue or abort
    human_decision = interrupt({
        "question": "Approve this operation?",
        "proposed_action": proposed_action,
        "impact": "This will delete 5,000 records"
    })

    if human_decision == "approved":
        result = execute_dangerous_operation(proposed_action)
        return {"operation_result": result}
    else:
        return {"operation_result": "cancelled", "error": "User rejected the operation"}


# Compile with interrupt capability
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_operation"],  # Pause BEFORE this node
)

# Usage flow:
config = {"configurable": {"thread_id": "task-001"}}

# First call: runs until interrupt
result = graph.invoke(initial_state, config=config)
# result["__interrupt__"] contains the interrupt payload

# Review state, present to human
print(result)

# Resume with human decision
graph.invoke(
    Command(resume="approved"),  # or "rejected"
    config=config
)
```

### Approval Workflow Pattern

```python
from langgraph.types import Command
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()
active_sessions: dict[str, dict] = {}  # thread_id -> session state

@app.post("/agent/start")
async def start_agent(task: str, user_id: str):
    thread_id = f"{user_id}-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    # Run until interrupt or completion
    result = await asyncio.to_thread(
        graph.invoke,
        {"messages": [HumanMessage(content=task)]},
        config
    )

    if "__interrupt__" in result:
        active_sessions[thread_id] = {
            "status": "pending_approval",
            "interrupt_data": result["__interrupt__"]
        }
        return {"thread_id": thread_id, "status": "pending_approval",
                "approval_required": result["__interrupt__"]}

    return {"thread_id": thread_id, "status": "complete", "result": result}

@app.post("/agent/{thread_id}/approve")
async def approve_action(thread_id: str, approved: bool):
    if thread_id not in active_sessions:
        raise HTTPException(404, "Thread not found")

    config = {"configurable": {"thread_id": thread_id}}
    decision = "approved" if approved else "rejected"

    result = await asyncio.to_thread(
        graph.invoke,
        Command(resume=decision),
        config
    )

    del active_sessions[thread_id]
    return {"status": "complete", "result": result}
```

---

## 8. Streaming

### stream() vs invoke()

```python
# invoke(): wait for complete result
result = graph.invoke(initial_state, config)
# Returns final state dict when done

# stream(): get updates as they happen
for chunk in graph.stream(initial_state, config):
    # chunk is a dict: {node_name: state_update}
    node_name = list(chunk.keys())[0]
    state_update = chunk[node_name]
    print(f"Node '{node_name}' updated: {state_update}")
```

### Streaming Modes

```python
# stream_mode="updates" (default): emit after each node
for chunk in graph.stream(state, config, stream_mode="updates"):
    print(chunk)  # {"node_name": {"field": "new_value"}}

# stream_mode="values": emit full state after each node
for chunk in graph.stream(state, config, stream_mode="values"):
    print(chunk["messages"][-1].content)  # Full state

# stream_mode="messages": emit individual LLM tokens
for chunk in graph.stream(state, config, stream_mode="messages"):
    # chunk is (message_chunk, metadata)
    msg_chunk, metadata = chunk
    if hasattr(msg_chunk, "content"):
        print(msg_chunk.content, end="", flush=True)
```

### FastAPI SSE Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/agent/stream")
async def stream_agent(request: dict):
    async def event_generator():
        thread_id = request.get("thread_id", str(uuid.uuid4()))
        config = {"configurable": {"thread_id": thread_id}}

        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=request["message"])]},
            config,
            stream_mode="messages"
        ):
            msg_chunk, metadata = chunk
            if hasattr(msg_chunk, "content") and msg_chunk.content:
                event = {
                    "type": "token",
                    "content": msg_chunk.content,
                    "node": metadata.get("langgraph_node", ""),
                }
                yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )
```

---

## 9. Production Notes

### Compiled Graph is Reusable

```python
# Module-level: compile ONCE at startup
graph = build_graph().compile(checkpointer=checkpointer)

# Request-level: invoke with thread-specific config
def handle_request(thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": [HumanMessage(message)]}, config)
```

Compilation processes the graph structure and validates edges. It's O(nodes) — do it once, not per request.

### Thread ID Strategy

```python
# User-session threads (most common)
thread_id = f"user:{user_id}:session:{session_id}"

# Task threads (ephemeral, no persistence needed)
thread_id = f"task:{task_type}:{uuid.uuid4()}"

# Named persistent threads (e.g., per-document review)
thread_id = f"doc:{document_hash}"
```

### Memory Management

Checkpoints accumulate. In production, implement a cleanup strategy:

```python
from datetime import datetime, timedelta

def cleanup_old_threads(db_url: str, older_than_days: int = 30):
    """Remove checkpoint data for threads older than N days."""
    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM checkpoints
                WHERE created_at < %s
                AND thread_id NOT IN (SELECT thread_id FROM active_threads)
            """, (cutoff,))
        conn.commit()
```

### Error Recovery

```python
def with_error_recovery(state: AgentState) -> dict:
    """Wrapper node for error handling."""
    error = state.get("error")
    if error:
        # Log, alert, clean state
        logger.error(f"Agent error in thread {state.get('task_id')}: {error}")
        return {
            "messages": [AIMessage(content=f"I encountered an error: {error}. Let me try a different approach.")],
            "error": None,  # Clear error
            "iteration_count": 0,  # Reset
        }
    return {}
```

---

## 10. Code Examples

### Complete Research Agent with LangGraph

```python
import uuid
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# ---- Tools ----

@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    # Replace with real implementation
    return f"Search results for '{query}': [mock results]"

@tool
def fetch_page(url: str) -> str:
    """Fetch content from a specific URL."""
    return f"Content from {url}: [mock content]"

@tool
def write_to_file(filename: str, content: str) -> str:
    """Save content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Written to {filename}"

tools = [search_web, fetch_page, write_to_file]

# ---- State ----

class ResearchAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    iterations: int
    error: Optional[str]

# ---- Nodes ----

model = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4096).bind_tools(tools)

SYSTEM = SystemMessage(content="""You are a thorough research assistant.

Process:
1. Use search_web to find relevant information
2. Use fetch_page to get details from promising sources
3. Synthesize information into a comprehensive report
4. Use write_to_file to save the final report

Be systematic: search broadly first, then dive deep on the best sources.""")

def researcher(state: ResearchAgentState) -> dict:
    messages = [SYSTEM] + state["messages"]
    try:
        response = model.invoke(messages)
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }
    except Exception as e:
        return {"error": str(e)}

def handle_error(state: ResearchAgentState) -> dict:
    error = state.get("error", "Unknown error")
    return {
        "messages": [AIMessage(content=f"Encountered error: {error}. Retrying...")],
        "error": None,
    }

# ---- Routing ----

def route(state: ResearchAgentState) -> str:
    if state.get("error"):
        return "error"
    if state.get("iterations", 0) >= 10:
        return END
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ---- Graph Assembly ----

def build_research_graph():
    g = StateGraph(ResearchAgentState)

    g.add_node("researcher", researcher)
    g.add_node("tools", ToolNode(tools))
    g.add_node("error_handler", handle_error)

    g.add_edge(START, "researcher")
    g.add_conditional_edges("researcher", route, {
        "tools": "tools",
        "error": "error_handler",
        END: END,
    })
    g.add_edge("tools", "researcher")
    g.add_edge("error_handler", "researcher")

    return g

# ---- Usage with Persistence ----

with SqliteSaver.from_conn_string("research_agent.db") as mem:
    graph = build_research_graph().compile(checkpointer=mem)

    thread_id = f"research-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in graph.stream(
        {
            "messages": [HumanMessage(content="Research the state of Python async frameworks in 2025")],
            "task": "framework_research",
            "iterations": 0,
            "error": None,
        },
        config,
        stream_mode="updates"
    ):
        node_name = list(chunk.keys())[0]
        print(f"[{node_name}] step complete")
```

---

## 11. Key Concepts Cheatsheet

| Concept | One-liner |
|---|---|
| StateGraph | The graph builder; add nodes and edges, then compile |
| State (TypedDict) | Shared data dict flowing through all nodes |
| add_messages reducer | Appends messages instead of replacing; essential for chat |
| Node | Python function: (state) → dict of state updates |
| Conditional edge | Routing function returns a string key; graph picks next node |
| Checkpointer | Persists state after each node; enables resume/replay |
| thread_id | Identifies a specific agent run; maps to a checkpoint chain |
| interrupt() | Pause execution, return to caller; await human input |
| Command(resume=...) | Resume an interrupted graph with a value |
| stream_mode="messages" | Get individual LLM tokens as they generate |
| ToolNode | Pre-built node that executes LangChain tools |

### Graph Compilation Checklist

- Every conditional edge has all return values mapped to nodes (or END)
- START has an outgoing edge
- All paths eventually reach END
- Checkpointer passed if persistence needed
- interrupt_before/after set if human review needed

---

## 12. Day 4 Exercises

**Exercise 1: Simple Calculator Graph**
Build a LangGraph agent with three nodes: `parse_problem` (extracts numbers and operation from text), `calculate` (performs the operation), `format_response` (formats the answer). Use direct edges (no conditionals). Invoke it with "What is 2,847 multiplied by 93?"
_Expected: Correct numerical answer, clear state transitions in stream output_

**Exercise 2: Loop with Circuit Breaker**
Extend the research graph to add a `circuit_breaker` node that activates if `iterations >= 5` OR if the same tool is called 3 times with identical arguments. The circuit breaker should produce a partial report with what was gathered so far.
_Expected: Agent stops gracefully at circuit breaker, returns partial results_

**Exercise 3: Checkpointed Conversation**
Build a simple chat agent with SQLite checkpointing. Run a 5-turn conversation, then kill the process. Start it again with the same thread_id and verify the conversation history is restored. Add a "summarize our conversation" capability that uses the full checkpointed history.
_Expected: History survives process restart; summary references all 5 original turns_

**Exercise 4: Human Approval Gate**
Build a file management agent that can read, write, and delete files. Implement an approval gate using `interrupt()` that triggers before any write or delete operation. Test the full flow: agent proposes action → human sees proposal → human approves → action executes.
_Expected: No destructive operation executes without explicit approval_
