# Day 2 — Tool Use & The ReAct Loop

> Tool use is the interface between reasoning and action. The model reasons; your code acts. Keep this separation clean and your agent architecture becomes much easier to reason about.

---

## 1. Function Calling Deep Dive

### The Tool Use Protocol

Tool use is a structured dialogue protocol baked into Claude's training. The model has learned to output well-formed tool call requests when it determines a tool is needed. This is not prompt engineering — it's a first-class API feature.

**What the model outputs when calling a tool:**

```python
# response.content when stop_reason == "tool_use"
[
    TextBlock(type="text", text="I'll search for that information."),  # Optional reasoning
    ToolUseBlock(
        type="tool_use",
        id="toolu_01A09q90qw90lq917835lq9",   # Unique ID for this call
        name="search_database",                 # Tool name you defined
        input={                                 # Arguments matching your schema
            "query": "FastAPI connection pooling",
            "limit": 5
        }
    )
]
```

The `id` is critical — you must reference it when returning the tool result.

**What you send back:**

```python
# Append the assistant's response to history
messages.append({"role": "assistant", "content": response.content})

# Then append your tool result
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01A09q90qw90lq917835lq9",  # Must match
            "content": "Found 3 results: [...]",              # String or content blocks
        }
    ]
})
```

> The `tool_use_id` linkage is how the model tracks which tool call this result corresponds to. Multiple tools can be called in parallel, so this ID is essential for correlation.

### Parallel Tool Calls

Claude can request multiple tools simultaneously:

```python
# response.content when parallel calls requested
[
    ToolUseBlock(id="toolu_aaa", name="get_cpu_metrics", input={"host": "web-01"}),
    ToolUseBlock(id="toolu_bbb", name="get_memory_metrics", input={"host": "web-01"}),
    ToolUseBlock(id="toolu_ccc", name="get_disk_metrics", input={"host": "web-01"}),
]
```

You execute all of them, then return all results in a single `user` message:

```python
import asyncio
from typing import Any

async def execute_parallel_tools(tool_calls: list, dispatcher) -> list[dict]:
    """Execute multiple tool calls concurrently."""
    async def run_one(tc):
        try:
            result = await asyncio.to_thread(dispatcher.execute, tc.name, tc.input)
            return {"type": "tool_result", "tool_use_id": tc.id, "content": result}
        except Exception as e:
            return {"type": "tool_result", "tool_use_id": tc.id,
                    "content": f"Error: {e}", "is_error": True}

    return await asyncio.gather(*[run_one(tc) for tc in tool_calls])
```

### Tool Result Format

Tool results can be plain strings, structured text, or content blocks:

```python
# Simple string (most common)
{"type": "tool_result", "tool_use_id": id, "content": "42 records found"}

# Error result
{"type": "tool_result", "tool_use_id": id, "content": "Database timeout", "is_error": True}

# Rich content (for images, structured data)
{"type": "tool_result", "tool_use_id": id, "content": [
    {"type": "text", "text": "Analysis complete:"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
]}
```

---

## 2. Writing Great Tool Schemas

The quality of your tool descriptions is directly proportional to how well the model uses them. This is one of the highest-leverage optimizations available.

### The Anatomy of a Tool Schema

```python
tool = {
    "name": "search_codebase",           # snake_case, verb_noun format
    "description": (
        "Search the codebase for files matching a pattern.\n\n"
        "Use this when you need to find which files contain a specific function, class, or string. "
        "Returns file paths and line numbers where matches are found.\n"
        "Do NOT use for reading file contents — use read_file for that.\n"
        "Supports regex patterns."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for. Example: 'def process_payment' or 'class.*Repository'"
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in, relative to project root. Default: '.' (entire project)",
                "default": "."
            },
            "file_extension": {
                "type": "string",
                "description": "Filter by extension: 'py', 'ts', 'sql'. Omit to search all files.",
                "enum": ["py", "ts", "js", "sql", "yaml", "json"]
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Default 20, max 100.",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            }
        },
        "required": ["pattern"]
    }
}
```

### Description Quality Rules

**DO:**
- Explain WHEN to use this tool vs. alternatives ("use X for Y, not Z")
- Provide concrete examples in the description
- Explain what the return value looks like
- Document side effects ("this will write to the database")
- Specify units ("timeout in seconds, not milliseconds")

**DON'T:**
- Use vague verbs: "handles", "processes", "manages" — be specific
- Skip the description on "obvious" tools — the model sees thousands of similar names
- Mark things required when they have sensible defaults
- Make one giant tool when 3 focused tools would be better

### Common Schema Mistakes

```python
# BAD: Vague, no examples, no constraints
{
    "name": "database_query",
    "description": "Run a database query",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

# GOOD: Specific, constrained, safe
{
    "name": "query_user_records",
    "description": (
        "Execute a read-only SQL query against the users database.\n\n"
        "IMPORTANT: Only SELECT statements are allowed. INSERT/UPDATE/DELETE/DROP will be rejected.\n"
        "Always include a LIMIT clause to avoid returning too many rows.\n"
        "Returns results as a JSON array of objects.\n\n"
        "Example: SELECT id, email, created_at FROM users WHERE created_at > NOW() - INTERVAL '7 days' LIMIT 100"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SELECT SQL query. Must include LIMIT <= 1000."
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Query timeout in seconds. Default 10, max 60.",
                "default": 10,
                "minimum": 1,
                "maximum": 60
            }
        },
        "required": ["sql"]
    }
}
```

### Tool Count vs. Performance

| Tool Count | Effect |
|---|---|
| 1-5 | Minimal overhead, reliable tool selection |
| 5-20 | Slight increase in wrong tool selection; descriptions critical |
| 20+ | Significant degradation; consider dynamic tool injection |
| 50+ | Use sub-agents with specialized tool sets |

---

## 3. The ReAct Pattern

**ReAct = Reasoning + Acting.** The model reasons about what to do, acts by calling tools, observes results, reasons again. This loop is the foundation of all agentic behavior.

```
Prompt → [Reason] → [Act (tool call)] → [Observe (tool result)] → [Reason] → [Act] → ... → [Answer]
```

### Core Loop Implementation

```python
import anthropic
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

ToolFunction = Callable[..., Any]


def react_loop(
    client: anthropic.Anthropic,
    system: str,
    initial_message: str,
    tools: list[dict],
    tool_registry: dict[str, ToolFunction],
    model: str = "claude-sonnet-4-5",
    max_iterations: int = 10,
    max_tokens: int = 4096,
) -> str:
    """
    Run the ReAct loop until end_turn or max_iterations reached.
    Returns the final text response.
    """
    messages = [{"role": "user", "content": initial_message}]
    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(max_iterations):
        logger.info(f"ReAct iteration {iteration + 1}/{max_iterations}")

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Always append the full assistant response (including tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text_blocks = [b for b in response.content if b.type == "text"]
            logger.info(f"ReAct complete. Tokens: {total_input_tokens} in / {total_output_tokens} out")
            return text_blocks[-1].text if text_blocks else ""

        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Hit max_tokens at iteration {iteration + 1}. Increase max_tokens.")

        if response.stop_reason != "tool_use":
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

        # Execute all tool calls in this response
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for tool_call in tool_calls:
            logger.info(f"Tool call: {tool_call.name}({tool_call.input})")
            result = _execute_tool_safe(tool_call, tool_registry)
            tool_results.append(result)
            logger.info(f"Tool result: {result['content'][:200]}")

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"Max iterations ({max_iterations}) reached. "
        "Increase max_iterations or reduce task complexity."
    )


def _execute_tool_safe(tool_call, registry: dict[str, ToolFunction]) -> dict:
    """Execute a tool call and always return a valid result dict."""
    if tool_call.name not in registry:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": f"Error: Tool '{tool_call.name}' is not available. Available tools: {list(registry.keys())}",
            "is_error": True,
        }

    try:
        # Unpack dict as kwargs for clean function signatures
        result = registry[tool_call.name](**tool_call.input)
        content = result if isinstance(result, str) else str(result)
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": content,
        }
    except TypeError as e:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": f"Argument error calling {tool_call.name}: {e}",
            "is_error": True,
        }
    except Exception as e:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": f"{type(e).__name__}: {e}",
            "is_error": True,
        }
```

### The Message History After 3 Tool Calls

Understanding what the messages array looks like during a loop is essential for debugging:

```python
messages = [
    # Turn 0: Initial prompt
    {"role": "user", "content": "Investigate why our payment service has high latency"},

    # Turn 1: Model reasons and calls two tools in parallel
    {"role": "assistant", "content": [
        {"type": "text", "text": "I'll check metrics and recent logs simultaneously."},
        {"type": "tool_use", "id": "toolu_001", "name": "get_metrics",
         "input": {"service": "payment", "metric": "p99_latency", "window": "1h"}},
        {"type": "tool_use", "id": "toolu_002", "name": "get_pod_logs",
         "input": {"service": "payment", "lines": 100}},
    ]},

    # Turn 2: Tool results (both returned together)
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_001",
         "content": "p99: 2400ms (normal: 180ms). Spike started 14:32 UTC"},
        {"type": "tool_result", "tool_use_id": "toolu_002",
         "content": "14:32:01 ERROR Connection pool exhausted (pool_size=10, waiting=47)"},
    ]},

    # Turn 3: Model narrows down, calls another tool
    {"role": "assistant", "content": [
        {"type": "text", "text": "Pool exhaustion found. Let me check DB connection counts."},
        {"type": "tool_use", "id": "toolu_003", "name": "query_database",
         "input": {"sql": "SELECT count(*), state FROM pg_stat_activity GROUP BY state"}},
    ]},

    # Turn 4: Final tool result
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_003",
         "content": '[{"count": 98, "state": "idle in transaction"}, {"count": 2, "state": "active"}]'},
    ]},

    # Turn 5: Final diagnosis (stop_reason: end_turn)
    {"role": "assistant", "content": [
        {"type": "text", "text": "**Root Cause**: Connection pool exhaustion due to transactions left open..."}
    ]},
]
```

---

## 4. Agent Dispatcher Pattern

```python
import json
import logging
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class ToolDispatcher:
    """Central registry for agent tools with schema management."""

    def __init__(self):
        self._registry: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def tool(self, schema: dict):
        """Decorator: register a function as a tool with its schema."""
        def decorator(func: Callable):
            self._registry[schema["name"]] = func
            self._schemas.append(schema)
            logger.debug(f"Registered tool: {schema['name']}")
            return func
        return decorator

    @property
    def schemas(self) -> list[dict]:
        return self._schemas

    def execute(self, name: str, args: dict) -> str:
        """Execute a named tool and return string result."""
        if name not in self._registry:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._registry.keys())}")

        result = self._registry[name](**args)
        return result if isinstance(result, str) else json.dumps(result, default=str, indent=2)

    def process_response(self, response) -> list[dict]:
        """Extract and execute all tool calls from an API response."""
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        results = []

        for call in tool_calls:
            try:
                content = self.execute(call.name, call.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": content,
                })
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": f"Error: {type(e).__name__}: {e}",
                    "is_error": True,
                })

        return results


# --- Example usage ---

import os
import subprocess

ops_dispatcher = ToolDispatcher()

@ops_dispatcher.tool({
    "name": "run_shell_command",
    "description": (
        "Run a read-only shell command. Only whitelisted commands are allowed.\n"
        "Allowed: ps, df, free, uptime, netstat, ss, ping (1 packet only)\n"
        "Returns stdout and stderr as a string."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run. Must start with an allowed command."
            }
        },
        "required": ["command"]
    }
})
def run_shell_command(command: str) -> str:
    ALLOWED = {"ps", "df", "free", "uptime", "netstat", "ss"}
    cmd_name = command.strip().split()[0]
    if cmd_name not in ALLOWED:
        raise PermissionError(f"Command '{cmd_name}' not in whitelist: {ALLOWED}")

    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=10
    )
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
```

---

## 5. Prompt Engineering for Agents

### Agent System Prompt Structure

```python
DEVOPS_AGENT_SYSTEM = """You are a senior site reliability engineer AI with access to production monitoring tools.

## Capabilities
You can investigate incidents, analyze metrics, query logs, and create tickets.

## Available Tools
- `get_metrics(service, metric, window)`: Query Prometheus for service metrics
- `get_pod_logs(service, lines, since)`: Retrieve Kubernetes pod logs  
- `query_database(sql, timeout_seconds)`: Read-only SQL queries
- `get_distributed_trace(trace_id)`: Fetch a Jaeger/Tempo trace
- `create_incident(severity, title, description)`: Create a PagerDuty incident

## Investigation Protocol
1. Gather broad metrics first (latency, error rate, saturation)
2. Narrow to the specific component showing anomalies
3. Check logs around the time anomalies started
4. Form a hypothesis, then verify with targeted queries
5. State root cause with supporting evidence before recommending action

## Decision Rules
- P1 (full outage) or P2 (partial outage): ALWAYS create an incident ticket
- Never suggest destructive actions (pod restart, rollback) without explicit user approval
- If after 6 tool calls you cannot determine root cause, say so and summarize your findings

## Output Format
Final response must include:
- **Root Cause**: One clear sentence
- **Evidence**: Bullet points with actual data (timestamps, numbers)
- **Recommended Action**: Specific, actionable steps
- **Incident Ticket**: Created/Not needed (include ID if created)"""
```

### Chain-of-Thought Prompting for Agents

Prepend a thinking instruction to get better reasoning before tool calls:

```python
THINKING_PREFIX = """Before calling any tools, briefly plan your approach in 2-3 sentences.
State: (1) what you know, (2) what you need to find out, (3) which tools you'll use first."""

# Inject into the user message, not the system prompt
def add_thinking_prompt(user_message: str) -> str:
    return f"{THINKING_PREFIX}\n\nTask: {user_message}"
```

### Few-Shot Examples in System Prompts

Few-shot examples dramatically improve tool use quality for complex scenarios:

Few-shot example (inline, not as code block):

**User:** Why is the checkout service slow?

**Agent thought:** I need to check latency metrics and recent error logs simultaneously.
Calls get_metrics("checkout", "p99_latency", "30m") and get_pod_logs("checkout", 50) in parallel.

**Observation:** p99=3200ms (normal: 200ms). Logs show "upstream connect timeout to inventory-svc"

**Agent thought:** Bottleneck is the inventory service. Check its error rate.
Calls get_metrics("inventory", "error_rate", "30m").

**Observation:** inventory error_rate=67%, memory usage=95%

**Final:** Root cause is inventory OOM causing connection timeouts cascading to checkout latency.

This pattern demonstrates: parallel initial calls, hypothesis from data, targeted follow-up.

---

## 6. Common Agent Failure Modes

### Infinite Loops

**Cause:** Agent keeps calling tools with the same inputs, never reaching end_turn.

**Detection:** Track (tool_name, frozenset(args.items())) tuples. If the same call repeats 3+ times, abort.

```python
from collections import Counter
import json

class LoopDetector:
    def __init__(self, max_repeats: int = 3):
        self.call_counts: Counter = Counter()
        self.max_repeats = max_repeats

    def check(self, tool_name: str, tool_input: dict) -> None:
        key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        self.call_counts[key] += 1
        if self.call_counts[key] >= self.max_repeats:
            raise RuntimeError(
                f"Infinite loop detected: {tool_name} called {self.max_repeats}x with same args"
            )
```

### Hallucinated Tool Calls

**Cause:** Model invents tool names not in your schema. Rare with Claude, more common with weaker models.

**Defense:** Your dispatcher raises `ValueError` for unknown tools. Return that as `is_error: True`. The model will usually recover. If it hallucinates the same tool repeatedly, your tool descriptions likely have a gap — the model is trying to do something your tools don't support.

### Wrong Argument Types

**Cause:** Model passes string "5" when integer is expected, or omits required fields.

**Defense:** Validate input against your Pydantic model before execution:

```python
from pydantic import BaseModel, ValidationError

class SearchArgs(BaseModel):
    query: str
    limit: int = 20
    offset: int = 0

def search_database_validated(raw_args: dict) -> str:
    try:
        args = SearchArgs.model_validate(raw_args)
    except ValidationError as e:
        return f"Invalid arguments: {e}"
    return search_database(args.query, args.limit, args.offset)
```

### Token Explosion in Multi-Step Agents

Each iteration adds tokens to the history (input + output). After 10 iterations:
- Tokens per call: ~5,000 (initial context + accumulated history)
- 10 calls × 5,000 = 50,000 input tokens just for history
- At Sonnet pricing: 50,000 × $3/MTok = $0.15 per agent run

**Mitigation:**
1. Compress tool results — don't dump 10KB JSON, extract the relevant fields
2. Summarize intermediate results periodically
3. Set hard `max_iterations` with clear error messages
4. Use `claude-haiku-4-5` for simple tool calls, Sonnet for final reasoning

### Tool Result Ignored

**Cause:** Model acknowledges a tool result but doesn't actually use the data.

**Detection:** Check if the final answer references numbers/facts from tool results.

**Fix:** Add explicit instruction: "Base your final answer ONLY on the tool results you received. Do not use prior knowledge about this system."

---

## 7. Production Notes

### Max Iterations Guard

Always set and enforce a hard limit. Without it, a buggy agent can run indefinitely, burning tokens and money:

```python
MAX_AGENT_ITERATIONS = 15  # Tune per use case
MAX_AGENT_COST_USD = 0.50  # Kill if cost exceeds this

def check_cost(usage, pricing={"input": 3.0, "output": 15.0}) -> float:
    return (usage["input_tokens"] * pricing["input"] +
            usage["output_tokens"] * pricing["output"]) / 1_000_000
```

### Tool Execution Timeouts

Wrap every tool execution in a timeout. A hung tool blocks the entire agent:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def execute_with_timeout(func, args: dict, timeout_seconds: float = 30) -> str:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **args)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            return f"Error: Tool timed out after {timeout_seconds}s"
```

### Structured Logging for Agent Traces

Every tool call and its result should be logged for debugging and cost tracking:

```python
import structlog

log = structlog.get_logger()

def logged_tool_call(tool_name: str, tool_input: dict, trace_id: str) -> dict:
    log.info("tool_call_start", tool=tool_name, args=tool_input, trace_id=trace_id)
    result = execute_tool(tool_name, tool_input)
    log.info("tool_call_end", tool=tool_name,
             result_len=len(result.get("content", "")),
             is_error=result.get("is_error", False),
             trace_id=trace_id)
    return result
```

### Cost of Multi-Step Agents

Real numbers for a 5-iteration agent (Sonnet pricing):

| Component | Tokens | Cost |
|---|---|---|
| System prompt (static) | 500 | $0.0015 per call |
| Tool schemas (5 tools) | 800 | $0.0024 per call |
| Conversation history (grows) | 500→3000 avg | $0.001→0.009 per call |
| Tool results (per call) | 300 avg | $0.0009 per call |
| Output per call | 400 avg | $0.006 per call |
| **Total 5-iteration run** | ~25,000 | **~$0.075** |

At 1,000 agent runs/day: $75/day. Plan accordingly.

---

## 8. Code Examples

### Full Multi-Tool Agent

```python
import anthropic
from typing import Any
import json
import httpx
import os

client = anthropic.Anthropic()

# Tool implementations
def web_search(query: str, num_results: int = 5) -> str:
    """Stub: replace with real search API."""
    return json.dumps([
        {"title": f"Result for {query}", "url": "https://example.com", "snippet": "..."}
    ])

def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch webpage content."""
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        return response.text[:5000]  # Limit to 5KB
    except Exception as e:
        return f"Fetch error: {e}"

def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use for facts, current events, documentation, or anything requiring up-to-date data. "
            "Returns a list of results with titles, URLs, and snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query. Be specific."},
                "num_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the contents of a webpage. "
            "Use after web_search when you need the full content of a specific page. "
            "Returns up to 5KB of the page text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL including https://"},
                "timeout": {"type": "integer", "default": 10}
            },
            "required": ["url"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression. "
            "Use for arithmetic, percentages, conversions. "
            "Supports: +, -, *, /, (), decimal numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(1024 * 8) / 1000'"}
            },
            "required": ["expression"]
        }
    }
]

TOOL_REGISTRY = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "calculate": calculate,
}

SYSTEM = """You are a research assistant with web access and calculation capability.

When given a research question:
1. Start with a broad search to understand the landscape
2. Follow up with specific searches or URL fetches to get details
3. Use calculate for any numerical analysis
4. Synthesize findings into a clear, structured answer with sources

Be thorough but efficient — use parallel tool calls when gathering independent information."""


def run_research_agent(question: str) -> str:
    return react_loop(
        client=client,
        system=SYSTEM,
        initial_message=question,
        tools=TOOLS,
        tool_registry=TOOL_REGISTRY,
        max_iterations=10,
    )


# Use the react_loop function from section 3 above
# result = run_research_agent("What are the main differences between HNSW and IVF vector indexes?")
```

### Error Handling Wrapper with Retry

```python
import time
import anthropic
from anthropic import RateLimitError, APIStatusError, APIConnectionError

def resilient_react_loop(
    client: anthropic.Anthropic,
    max_api_retries: int = 3,
    **loop_kwargs
) -> str:
    """react_loop with API-level retry for transient errors."""
    last_error = None

    for attempt in range(max_api_retries):
        try:
            return react_loop(client=client, **loop_kwargs)
        except RateLimitError as e:
            wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
            time.sleep(wait)
            last_error = e
        except APIConnectionError as e:
            time.sleep(1)
            last_error = e
        except APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(2 ** attempt)
                last_error = e
            else:
                raise  # 4xx errors: don't retry

    raise RuntimeError(f"Failed after {max_api_retries} retries") from last_error
```

---

## 9. Key Concepts Cheatsheet

| Concept | One-liner |
|---|---|
| tool_use block | Model's structured request to call a function |
| tool_result | Your execution result, linked by tool_use_id |
| ReAct loop | while stop_reason == "tool_use": execute and feed back |
| Parallel calls | Multiple tool_use blocks in one response; execute all before replying |
| max_iterations | Hard guard against infinite loops; always set this |
| is_error: true | Signal to model that tool failed; it will usually try to recover |
| Tool description | The most impactful thing you can tune for tool selection quality |
| LoopDetector | Track (tool, args) pairs; abort on repeated identical calls |

### Stop Reason Decision Tree

```
response.stop_reason
├── "end_turn"      → Extract text, done
├── "tool_use"      → Execute tools, append results, loop
├── "max_tokens"    → Increase max_tokens or truncate context
└── "stop_sequence" → Done (you defined this stop)
```

---

## 10. Day 2 Exercises

**Exercise 1: Multi-Tool Calculator Agent**
Build an agent with 3 tools: `add(a, b)`, `multiply(a, b)`, `divide(a, b)`. Give it this task: "Calculate the compound interest on $10,000 at 7% annual rate for 5 years, compounded monthly." Verify it uses multiple tool calls correctly.
_Expected: Correct answer (~$4,176.25 interest), showing 3+ tool calls_

**Exercise 2: Tool Schema Optimization**
Take this bad schema: `{"name": "data", "description": "Gets data", "input_schema": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}}`. Rewrite it as a proper database record fetcher. Test that the model uses it correctly vs. the original.
_Expected: Improved tool call quality, fewer wrong-argument errors_

**Exercise 3: Loop Detector Integration**
Extend the `react_loop` function to use the `LoopDetector` class. Create a deliberately broken tool that always returns an error, and verify the loop detector catches the infinite loop after 3 identical calls.
_Expected: RuntimeError raised after 3 identical calls, not an infinite loop_

**Exercise 4: Async Parallel Agent**
Rewrite `react_loop` as `async_react_loop` using `asyncio`. Make tool execution genuinely async (use `asyncio.to_thread` for sync tools). Verify that when the model requests 3 tools in parallel, they execute concurrently (time the difference vs. sequential).
_Expected: 3 concurrent tool calls completing in ~1× their max latency, not 3× their sum_
