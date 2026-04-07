# Day 1 — LLMs & The API

> This document is a reference, not a tutorial. Read it once end-to-end, then use it as a lookup when you're building. If you know distributed systems, you already have 80% of the mental models you need — you're just learning the domain-specific vocabulary.

---

## 1. How LLMs Work

### The Transformer Architecture (Systems Engineer Framing)

An LLM is a function: `f(token_sequence) → probability_distribution_over_next_token`. Everything else is engineering around that core.

**Tokens → Embeddings**

Before any computation, tokens (integer IDs) are mapped to dense vectors via an embedding matrix `E ∈ R^{vocab_size × d_model}`. For Claude, `d_model` is in the thousands. This is just a lookup table — token ID 5423 → row 5423 of `E`. The resulting vector is a point in high-dimensional space where semantic similarity corresponds to geometric proximity. "cat" and "kitten" land near each other. "cat" and "database" do not.

**Positional Encoding**

Transformers process all tokens in parallel (unlike RNNs). To inject sequence order, positional encodings are added to embeddings — either sinusoidal (original paper) or learned (most modern models). Without this, the model sees a bag of tokens with no concept of "this came before that."

**Attention Mechanism (The Real Engine)**

Attention is a differentiable, learned weighted lookup. For each token position, the mechanism asks: "which other tokens in this sequence are most relevant to computing my next representation?"

Concretely for each attention head:
```
Q = X @ W_Q   # Queries: "what am I looking for?"
K = X @ W_K   # Keys:    "what do I advertise?"
V = X @ W_V   # Values:  "what do I actually contain?"

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

The `softmax(Q @ K.T)` produces an attention weight matrix — row `i`, column `j` is "how much does token `i` attend to token `j`." This is bounded by the full context window. Every token can attend to every prior token. This is O(n²) in memory and compute, which is why context windows have limits.

Multi-head attention runs this in parallel across H heads, each learning different relationship types (syntactic, semantic, coreference, etc.), then concatenates:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) @ W_O
```

**Feed-Forward Layers**

After attention, each token position passes independently through a two-layer MLP:
```
FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
```

These layers are where most of the model's "knowledge" is stored — they act as key-value memories encoding factual associations. The FFN is actually larger than the attention block (typically 4× the model dimension).

**Layer Stack**

Both attention + FFN blocks stack N times (e.g., 32–96 layers for large models), with residual connections and layer normalization. Output of the final layer → linear projection → softmax → probability distribution over the vocabulary.

**Autoregressive Generation**

Generation is a for-loop:
```python
tokens = encode(prompt)
while not done:
    logits = model(tokens)           # forward pass
    next_token = sample(logits[-1])  # from last position's distribution
    tokens.append(next_token)
    if next_token == EOS: break
```

`temperature` scales the logits before softmax — higher = flatter distribution = more random. `top_p` (nucleus sampling) truncates to the smallest set of tokens whose cumulative probability exceeds p. The model sees its own output as input on each iteration — this is why errors compound and why you can't parallelize generation.

**Why Stateless**

The model has no persistent memory between API calls. The weights are fixed after training. A "conversation" is an illusion — on every call, you're feeding the complete history and getting one more completion. The model is a pure function of its input.

**Context Window as Working Memory**

The context window (128K for Claude 3.5, 200K for Claude 3 Opus) is literally the maximum sequence length the attention mechanism can process. It's not a soft limit — tokens beyond the window are invisible to the model. Think of it as RAM: fast, limited, expensive. Everything you want the model to "know" during a session must fit here.

---

## 2. Tokenization

### What Tokens Are

Tokens are the atomic units of text from the model's perspective. Not characters, not words — something in between. "tokenization" might be 2-3 tokens. "cat" is 1. A typical English word averages ~1.3 tokens. Code tends to be denser.

**Why Not Characters?**
- Sequences would be 4-5× longer → O(n²) attention becomes unmanageable
- No semantic pre-grouping

**Why Not Words?**
- Vocabulary explosion (every morphological variant = new token)
- Can't handle unknown words, numbers, code identifiers, URLs

### Byte-Pair Encoding (BPE)

BPE starts with a character-level vocabulary, then iteratively merges the most frequent adjacent pair into a single token, repeating until the vocabulary hits the target size (~100K tokens). This produces:
- Common subwords as single tokens ("ing", "tion", "API")
- Rare words split into sub-tokens ("tokenization" → "token" + "ization")
- Numbers often split digit-by-digit ("12345" → "1" + "23" + "45")

**Practical Implications:**

| Content Type | Tokens per word (approx) |
|---|---|
| Common English prose | ~1.3 |
| Technical/domain terms | ~1.8 |
| Python code | ~1.5 |
| JSON | ~2.0 |
| Minified code | ~3.0+ |
| Chinese/Japanese | ~2-3 per character |

**Why Token Count Matters:**
1. **Cost**: You pay per token (input and output separately)
2. **Context limits**: Every byte of your conversation history, system prompt, and tool schemas consumes tokens
3. **Latency**: More input tokens = slower time-to-first-token; more output tokens = longer generation

```python
import anthropic

client = anthropic.Anthropic()

# Count tokens before calling (saves money in development)
token_count = client.messages.count_tokens(
    model="claude-sonnet-4-5",
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Explain BPE tokenization"}]
)
print(f"Input tokens: {token_count.input_tokens}")
```

---

## 3. The Anthropic Messages API

### Full Request Schema

```python
response = client.messages.create(
    # Required
    model="claude-sonnet-4-5",          # Model ID string
    max_tokens=1024,                     # Hard cap on output tokens
    messages=[...],                      # Conversation history array

    # Optional but important
    system="You are...",                 # System prompt (string or content blocks)
    temperature=1.0,                     # 0.0-1.0, default 1.0
    top_p=0.999,                         # Nucleus sampling threshold
    top_k=40,                            # Top-k sampling (less common)
    stop_sequences=["</answer>"],        # Additional stop strings

    # Tools
    tools=[...],                         # Tool definitions
    tool_choice={"type": "auto"},        # auto | any | tool | none

    # Advanced
    stream=False,                        # Enable streaming
    metadata={"user_id": "usr_123"},     # Request metadata (not sent to model)
)
```

### Parameter Reference

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | str | required | Use `claude-sonnet-4-5` for most work |
| `max_tokens` | int | required | Output token budget. 4096 is safe default |
| `messages` | list | required | Alternating user/assistant turns |
| `system` | str/list | None | Injected before conversation. Not a turn. |
| `temperature` | float | 1.0 | Lower = more deterministic. 0.0 = greedy |
| `top_p` | float | 0.999 | Reduce for more focused outputs |
| `top_k` | int | None | Rarely needed; prefer top_p |
| `stop_sequences` | list[str] | [] | Model stops when it generates any of these |
| `tools` | list | [] | Available tools (see Day 2) |
| `tool_choice` | dict | auto | Controls if/which tools the model uses |
| `stream` | bool | False | Enable streaming response |

**Temperature in Practice:**
- `0.0`: Deterministic (greedy), best for structured extraction, code generation with correct answers
- `0.3-0.5`: Low creativity, good for factual Q&A, data extraction
- `0.7-1.0`: Standard for general use, creative writing
- Never go above 1.0 — the API will error

### The Messages Array

```python
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Why?"},
]
```

Rules:
- Must alternate user/assistant (with caveats for tool use — see Day 2)
- First message must be `role: user`
- `content` can be a string or a list of content blocks (for multimodal, tool results, etc.)
- **You own this array** — the API has no session state

### The Response Object

```python
# response: anthropic.types.Message
{
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Here is the answer..."
        }
    ],
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",          # See below
    "stop_sequence": None,
    "usage": {
        "input_tokens": 25,
        "output_tokens": 117,
        "cache_creation_input_tokens": 0,   # Prompt cache writes
        "cache_read_input_tokens": 0,       # Prompt cache hits
    }
}
```

### Stop Reasons

| `stop_reason` | Meaning | Action |
|---|---|---|
| `end_turn` | Model finished naturally | Normal completion |
| `max_tokens` | Hit your `max_tokens` limit | Increase limit or truncate output |
| `tool_use` | Model wants to call a tool | Execute tool, append result, call API again |
| `stop_sequence` | Hit one of your `stop_sequences` | Response complete (you defined the stop) |

> `stop_reason == "tool_use"` is the heartbeat of agentic systems. When you see this, your code runs, not the model.

### The Usage Object

```python
usage = response.usage
cost_estimate = (
    usage.input_tokens * 3.0 / 1_000_000  +   # $3/MTok for Sonnet 3.5
    usage.output_tokens * 15.0 / 1_000_000     # $15/MTok for Sonnet 3.5
)
```

Always log `usage` in production. Token counts compound in multi-step agents.

---

## 4. Memory = Context

### The Stateless Reality

Every API call is independent. The model has no memory of previous calls. "Memory" in LLM applications is the engineering problem of what to put in the `messages` array.

```python
# WRONG mental model
response1 = client.messages.create(messages=[{"role": "user", "content": "My name is Sharjeel"}])
response2 = client.messages.create(messages=[{"role": "user", "content": "What's my name?"}])
# response2 has no idea

# CORRECT: you maintain history
history = []
history.append({"role": "user", "content": "My name is Sharjeel"})
response1 = client.messages.create(model="claude-sonnet-4-5", max_tokens=100, messages=history)
history.append({"role": "assistant", "content": response1.content[0].text})
history.append({"role": "user", "content": "What's my name?"})
response2 = client.messages.create(model="claude-sonnet-4-5", max_tokens=100, messages=history)
# Works correctly
```

### Context Window Management Strategies

**1. Sliding Window (Simple)**
Keep the last N turns. Loses early context but simple.
```python
MAX_TURNS = 20
messages = messages[-MAX_TURNS:]
```

**2. Token-Based Truncation (Better)**
Keep as many recent messages as fit within a token budget.
```python
def truncate_to_budget(messages: list, max_tokens: int, model: str) -> list:
    client = anthropic.Anthropic()
    while len(messages) > 1:
        count = client.messages.count_tokens(model=model, messages=messages)
        if count.input_tokens <= max_tokens:
            return messages
        messages = messages[2:]  # Drop oldest user+assistant pair
    return messages
```

**3. Summarization (Robust)**
When history exceeds threshold, compress old turns into a summary.
```python
def summarize_history(old_messages: list) -> str:
    summary_prompt = f"""Summarize this conversation in 200 words, preserving key facts, decisions, and context:
    
{format_messages(old_messages)}"""
    
    response = client.messages.create(
        model="claude-haiku-4-5",  # Cheap model for this task
        max_tokens=300,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    return response.content[0].text
```

**4. Semantic Retrieval (Day 3 topic)**
Store turns as vectors, retrieve relevant ones per query. Best for long-running agents.

---

## 5. Structured Output Pattern

Agents need machine-readable output. The LLM returns text — you need to get structured data from it reliably.

### Approach 1: Prompt Engineering + JSON Parsing

```python
from pydantic import BaseModel
import json

class CodeReview(BaseModel):
    severity: str  # "low" | "medium" | "high" | "critical"
    issues: list[str]
    suggestions: list[str]
    score: int  # 1-10

SYSTEM_PROMPT = """You are a code reviewer. Always respond with valid JSON matching this schema:
{
  "severity": "low|medium|high|critical",
  "issues": ["list of specific issues found"],
  "suggestions": ["list of improvement suggestions"],
  "score": <integer 1-10>
}

No markdown, no explanation — pure JSON only."""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0.0,  # Deterministic for structured output
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": f"Review this code:\n```python\n{code}\n```"}]
)

# Parse and validate
raw = response.content[0].text
review = CodeReview.model_validate_json(raw)
```

### Approach 2: Tool Use for Structured Output (Preferred)

Use a tool schema as your output schema — the model is better trained to produce valid tool arguments than JSON in text:

```python
output_schema = {
    "name": "submit_review",
    "description": "Submit the code review result",
    "input_schema": {
        "type": "object",
        "properties": {
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"]
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "score": {"type": "integer", "minimum": 1, "maximum": 10}
        },
        "required": ["severity", "issues", "suggestions", "score"]
    }
}

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0.0,
    tools=[output_schema],
    tool_choice={"type": "tool", "name": "submit_review"},  # Force this specific tool
    messages=[{"role": "user", "content": f"Review:\n{code}"}]
)

# Extract structured data from tool use block
tool_use = next(b for b in response.content if b.type == "tool_use")
review = CodeReview.model_validate(tool_use.input)
```

> Using `tool_choice: {"type": "tool", "name": "..."}` forces the model to always call that specific tool — effectively giving you a guaranteed structured output. This is the most reliable extraction pattern.

---

## 6. Tool Use Introduction

Tool use (function calling) is how LLMs take actions in the world. The model can't execute code — it can only output text that says "please call this function with these arguments." Your code does the actual work.

The full flow:
1. You define tools (name, description, input schema)
2. You send a request with tools defined
3. Model responds with `stop_reason: "tool_use"` and a `tool_use` content block
4. Your code extracts the tool name and arguments
5. Your code executes the tool
6. You append the result as a `tool_result` message
7. You call the API again with the updated history
8. Repeat until `stop_reason: "end_turn"`

This is covered in depth in Day 2, but remember: **the model never executes anything.** It's always text in, text out. Tool "execution" is just a structured way for the model to request external function calls.

---

## 7. Production Notes

### Streaming vs. Synchronous

```python
# Synchronous: use for internal pipeline steps, tool calls, batch processing
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=messages
)
text = response.content[0].text

# Streaming: use for user-facing interfaces
with client.messages.stream(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=messages
) as stream:
    for text_chunk in stream.text_stream:
        print(text_chunk, end="", flush=True)
    final = stream.get_final_message()
```

**Decision rule:** If a human is waiting for output, stream. If your code is consuming the output programmatically, sync is simpler and you avoid partial-state handling.

### Token Budgeting in Production

```python
# Budget allocation for a 100K context window
CONTEXT_BUDGET = 100_000
SYSTEM_PROMPT_TOKENS = 2_000      # Measure your actual system prompt
TOOL_SCHEMAS_TOKENS = 1_500       # Grows with tool count
RESPONSE_RESERVE = 4_096          # max_tokens you set
HISTORY_BUDGET = CONTEXT_BUDGET - SYSTEM_PROMPT_TOKENS - TOOL_SCHEMAS_TOKENS - RESPONSE_RESERVE
# 92_404 tokens for conversation history
```

### Rate Limits

Anthropic uses a tiered rate limit system. As of 2025:

| Tier | Requests/min | Tokens/min (input) | Tokens/min (output) |
|---|---|---|---|
| Free | 5 | 25K | 5K |
| Tier 1 ($5 spend) | 50 | 50K | 10K |
| Tier 2 ($40 spend) | 1000 | 100K | 32K |
| Tier 3 ($200 spend) | 2000 | 200K | 64K |
| Tier 4 ($400 spend) | 4000 | 400K | 128K |

Rate limit errors return HTTP 429. Always implement exponential backoff:

```python
import time
import anthropic
from anthropic import RateLimitError, APIStatusError

def call_with_retry(client, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

### Error Codes Reference

| HTTP Code | Error Type | Action |
|---|---|---|
| 400 | `invalid_request_error` | Fix your request (bad params, invalid messages) |
| 401 | `authentication_error` | Check API key |
| 403 | `permission_error` | Check permissions for this model/feature |
| 404 | `not_found_error` | Wrong endpoint or model name |
| 429 | `rate_limit_error` | Backoff and retry |
| 500 | `api_error` | Anthropic server error, retry |
| 529 | `overloaded_error` | Anthropic overloaded, retry with backoff |

---

## 8. Code Examples

### Basic Completion

```python
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain the CAP theorem in 3 sentences."}
    ]
)

print(response.content[0].text)
print(f"\nTokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
```

### Conversation with History

```python
import anthropic
from typing import TypeAlias

Message: TypeAlias = dict[str, str]

class ConversationSession:
    def __init__(self, system_prompt: str, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = system_prompt
        self.history: list[Message] = []
        self.total_tokens = 0

    def chat(self, user_message: str, max_tokens: int = 1024) -> str:
        self.history.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            messages=self.history,
        )

        assistant_text = response.content[0].text
        self.history.append({"role": "assistant", "content": assistant_text})
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        return assistant_text

    def reset(self):
        self.history = []

# Usage
session = ConversationSession(
    system_prompt="You are a senior distributed systems engineer. Be concise and technical."
)

print(session.chat("What's the difference between Paxos and Raft?"))
print(session.chat("Which one would you use for a new project and why?"))
print(f"Total tokens: {session.total_tokens}")
```

### Streaming Response

```python
import anthropic
import sys

client = anthropic.Anthropic()

def stream_response(messages: list, system: str = "") -> str:
    """Stream response and return full text when complete."""
    full_text = ""

    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            full_text += chunk

        # Get final message for usage stats
        final = stream.get_final_message()
        print(f"\n\n[{final.usage.input_tokens} in / {final.usage.output_tokens} out]")

    return full_text

# SSE-compatible async version for FastAPI
import asyncio

async def stream_response_async(messages: list, system: str = ""):
    """Async generator for streaming in FastAPI SSE endpoints."""
    async with anthropic.AsyncAnthropic().messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        async for chunk in stream.text_stream:
            yield chunk
```

### Structured Output with Pydantic

```python
import anthropic
import json
from pydantic import BaseModel, ValidationError
from typing import Literal

class ServiceAnalysis(BaseModel):
    service_name: str
    service_type: Literal["synchronous", "asynchronous", "event-driven", "batch"]
    estimated_rps: int
    bottlenecks: list[str]
    recommended_scaling: str
    cache_candidate: bool

client = anthropic.Anthropic()

def analyze_service(description: str, retries: int = 3) -> ServiceAnalysis:
    """Extract structured service analysis from free-text description."""

    tool_def = {
        "name": "record_analysis",
        "description": "Record the structured analysis of a service",
        "input_schema": ServiceAnalysis.model_json_schema()
    }

    for attempt in range(retries):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            temperature=0.0,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "record_analysis"},
            messages=[{
                "role": "user",
                "content": f"Analyze this service:\n\n{description}"
            }]
        )

        try:
            tool_block = next(b for b in response.content if b.type == "tool_use")
            return ServiceAnalysis.model_validate(tool_block.input)
        except (StopIteration, ValidationError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to extract structured output after {retries} attempts: {e}")

# Usage
analysis = analyze_service("""
Our payment service handles checkout requests. It calls the inventory API synchronously,
then writes to PostgreSQL, then publishes to Kafka. Currently getting 500 req/s but
seeing DB connection pool exhaustion at peak.
""")

print(f"Type: {analysis.service_type}")
print(f"Bottlenecks: {analysis.bottlenecks}")
print(f"Cache candidate: {analysis.cache_candidate}")
```

---

## 9. Key Concepts Cheatsheet

| Concept | One-liner |
|---|---|
| Token | Atomic text unit, ~0.75 words on average |
| Context window | The model's RAM — everything must fit here |
| Temperature | Controls randomness. 0=deterministic, 1=default |
| stop_reason | Why generation stopped — your main control signal |
| Messages array | You own the conversation history; API is stateless |
| Tool use | Model says "call function X with args Y"; you execute |
| Structured output | Force JSON via tool_choice or prompt + parse |
| Streaming | Tokens arrive as generated, not buffered |
| Prompt caching | Pay once to store a prompt prefix, reuse cheaply |

### Model Selection Quick Reference

| Model | Best For | Cost (input/output per MTok) |
|---|---|---|
| `claude-haiku-4-5` | Classification, extraction, fast tasks | $0.80 / $4.00 |
| `claude-sonnet-4-5` | General purpose, coding, agents | $3.00 / $15.00 |
| `claude-opus-4-5` | Complex reasoning, highest quality | $15.00 / $75.00 |

---

## 10. Day 1 Exercises

**Exercise 1: Token Profiling**
Write a function that takes a system prompt and a list of conversation messages and prints: total input tokens, percentage used by system vs. conversation, and estimated cost at Sonnet pricing. Test it with a 5-turn conversation about database design.
_Expected: Token breakdown table with cost estimate_

**Exercise 2: Context Manager**
Build a `ManagedConversation` class that automatically truncates history when it exceeds 80% of a configurable token budget, using token-counting API to measure accurately before each call. Log when truncation occurs.
_Expected: Class that never exceeds token budget, with truncation logging_

**Exercise 3: Structured Extraction**
Given a free-text incident report (e.g., "At 14:32 UTC the payment service returned 503s for 8 minutes, affecting 2300 users, root cause was connection pool exhaustion"), extract a Pydantic model with: `timestamp`, `service`, `duration_minutes`, `affected_users`, `root_cause`, `severity` (P1-P4). Use both the prompt approach and the tool approach, compare reliability.
_Expected: Populated Pydantic model with correct field types_

**Exercise 4: Streaming API Server**
Build a FastAPI endpoint `POST /chat` that accepts `{message: str, session_id: str}` and streams the response as Server-Sent Events. Maintain conversation history per session_id in memory. Use async Anthropic client.
_Expected: Working SSE endpoint where `curl` shows tokens arriving in real time_
