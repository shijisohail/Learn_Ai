# Day 7 — Capstone: Autonomous Code Review Agent

> You've spent six days building toward this. Today you wire it all together: multi-agent orchestration, parallel fan-out, structured output, streaming, production hardening, and a real FastAPI deployment. The result is a working autonomous code review system that you can actually run on pull requests.

---

## 1. System Design

### Requirements

- Accept a GitHub PR URL as input
- Fetch the diff and changed file contents from GitHub API
- Run three specialist analysis agents in parallel (Security, Performance, Style)
- Deduplicate and rank findings via a Critic agent
- Produce a structured JSON review report via a Formatter agent
- Stream progress to the caller via Server-Sent Events (SSE)
- Budget: max $0.50 per review, max 60s wall-clock time
- Cache: identical (repo, pr_number, commit_sha) returns the cached result without re-running LLMs

### Architecture Diagram

```
GitHub PR URL
      │
      ▼
┌─────────────────────┐
│   FastAPI Endpoint  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Orchestrator     │ ← fetches PR diff, decides which agents to run
└──┬──────┬──────┬────┘
   │      │      │  fan-out (parallel via LangGraph Send)
   ▼      ▼      ▼
Sec    Perf   Style   ← specialist agents
Agent  Agent  Agent
   │      │      │  fan-in (merge findings into state via operator.add reducers)
   └──────┴──────┘
          │
          ▼
┌─────────────────────┐
│    Critic Agent     │ ← deduplicates, adds context, ranks by severity
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Formatter Agent   │ ← structured JSON output: ReviewReport
└─────────┬───────────┘
          │
          ▼
    ReviewReport JSON
```

### Design Principles

**Single responsibility.** Each agent has exactly one job. Security knows OWASP. Performance knows N+1. The Formatter knows JSON. No agent does two jobs.

**Stateless agents.** Agents receive their inputs as arguments and return outputs. They never read from or write to shared state directly — the orchestrator (LangGraph graph) owns the state.

**Orchestrator owns state.** The LangGraph `AgentState` TypedDict is the single source of truth. Nodes read from it and return dicts that update it. No agent holds a reference to state between calls.

**Parallel execution for speed.** Security, Performance, and Style run simultaneously via LangGraph's `Send()` fan-out pattern. Total wall-clock time is bounded by the slowest specialist, not the sum of all three.

**Fail partial, not total.** If one specialist fails (network hiccup, timeout), the others continue. The Critic works with whatever findings it receives. Graceful degradation beats hard failures.

---

## 2. State Design

### Pydantic Models

```python
from __future__ import annotations
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


class Category(str, Enum):
    SECURITY    = "security"
    PERFORMANCE = "performance"
    STYLE       = "style"


class Finding(BaseModel):
    id: str = Field(description="Unique finding ID, e.g. SEC-001")
    category: Category
    severity: Severity
    title: str = Field(description="One-line summary, e.g. 'SQL injection via f-string'")
    description: str = Field(description="Full explanation of the issue")
    file: str = Field(description="Filename where the issue was found")
    line_range: Optional[tuple[int, int]] = Field(
        default=None, description="(start_line, end_line) of the problematic code"
    )
    code_snippet: Optional[str] = Field(
        default=None, description="Offending code excerpt, 10 lines max"
    )
    recommendation: str = Field(description="Concrete fix recommendation")
    cwe_id: Optional[str] = Field(
        default=None, description="CWE identifier for security findings, e.g. CWE-89"
    )
    is_duplicate: bool = Field(default=False, description="Marked by Critic if duplicate")


class QuickWin(BaseModel):
    title: str
    description: str
    effort: str  # "1-line fix", "10-min refactor", etc.


class Praise(BaseModel):
    title: str
    description: str


class ReviewMetadata(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    duration_seconds: float
    agents_run: list[str]
    agents_failed: list[str]
    cache_hit: bool = False


class ReviewReport(BaseModel):
    pr_url: str
    repo: str
    pr_number: int
    commit_sha: str
    reviewed_at: datetime
    executive_summary: str = Field(
        description="2-3 sentence plain-English summary of the overall review"
    )
    findings_by_severity: dict[Severity, list[Finding]] = Field(
        description="Findings grouped by severity level"
    )
    all_findings: list[Finding] = Field(
        description="All findings in ranked order (CRITICAL first)"
    )
    quick_wins: list[QuickWin] = Field(
        description="Low-effort, high-value fixes the author can do immediately"
    )
    praise: list[Praise] = Field(
        description="Things done well — keeps the review balanced"
    )
    total_findings: int
    findings_by_category: dict[Category, int]
    metadata: ReviewMetadata
```

### LangGraph State TypedDict

```python
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    pr_url: str                          # Full GitHub PR URL
    repo: str                            # "owner/repo"
    pr_number: int                       # PR number as integer
    commit_sha: str                      # Head commit SHA for cache key

    # ── Fetched data ───────────────────────────────────────────────────────
    pr_diff: str                         # Raw unified diff text
    code_files: dict[str, str]           # {filename: full_file_content}

    # ── Specialist outputs (accumulated via reducer) ───────────────────────
    # Annotated[list, operator.add] means each node's list gets APPENDED,
    # not replaced. LangGraph calls the reducer when merging parallel branches.
    security_findings:    Annotated[list[Finding], operator.add]
    performance_findings: Annotated[list[Finding], operator.add]
    style_findings:       Annotated[list[Finding], operator.add]

    # ── Post-critic ────────────────────────────────────────────────────────
    all_findings: list[Finding]          # Merged, deduplicated, ranked by Critic

    # ── Final output ───────────────────────────────────────────────────────
    final_report: ReviewReport | None

    # ── Bookkeeping (also accumulated via operator.add) ────────────────────
    agents_run:           Annotated[list[str], operator.add]
    agents_failed:        Annotated[list[str], operator.add]
    total_input_tokens:   Annotated[int, operator.add]
    total_output_tokens:  Annotated[int, operator.add]
    errors:               Annotated[list[str], operator.add]
    start_time:           float          # time.monotonic() set at graph start
```

The `Annotated[list[Finding], operator.add]` pattern is the critical LangGraph concept for parallel branches. When security_node, performance_node, and style_node all return dicts containing their respective `_findings` lists, LangGraph's state reducer calls `operator.add` (list concatenation) to merge them rather than overwriting. Without this annotation, the last branch to complete would clobber the others.

---

## 3. Tool Implementations

These functions are called by the orchestrator node (fetch) and by specialist nodes before the LLM analysis step. They gather raw data and produce structured hints.

```python
import re
import ast
import asyncio
import httpx
import os
from typing import Optional


GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


async def fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch the unified diff for a PR from the GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {**GITHUB_HEADERS, "Accept": "application/vnd.github.v3.diff"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


async def fetch_pr_metadata(repo: str, pr_number: int) -> dict:
    """Fetch PR metadata: title, head commit SHA, author, branch names."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=GITHUB_HEADERS)
        resp.raise_for_status()
        d = resp.json()
        return {
            "title":       d["title"],
            "commit_sha":  d["head"]["sha"],
            "base_branch": d["base"]["ref"],
            "head_branch": d["head"]["ref"],
            "author":      d["user"]["login"],
        }


async def get_changed_files(repo: str, pr_number: int) -> dict[str, str]:
    """
    List all files changed in the PR and fetch their full contents.
    Returns {filename: file_content}. Skips deleted and binary files.
    """
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=GITHUB_HEADERS)
        resp.raise_for_status()
        files = resp.json()

    async def _fetch_one(filename: str, raw_url: str) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(raw_url, headers=GITHUB_HEADERS)
            if r.status_code != 200:
                return filename, "[fetch failed]"
            try:
                return filename, r.text
            except UnicodeDecodeError:
                return filename, "[binary file skipped]"

    tasks = [
        _fetch_one(f["filename"], f["raw_url"])
        for f in files
        if f.get("status") != "removed" and f.get("raw_url")
    ]
    # Cap at 10 files to control context size
    results = await asyncio.gather(*tasks[:10], return_exceptions=True)
    return {
        fname: content
        for item in results
        if isinstance(item, tuple)
        for fname, content in [item]
        if not isinstance(fname, Exception)
    }


def parse_diff_hunks(diff: str) -> dict[str, list[str]]:
    """
    Parse unified diff into {filename: [added_lines]}.
    Only returns lines prefixed with '+' (additions) — what we're reviewing.
    """
    files: dict[str, list[str]] = {}
    current_file = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            files[current_file] = []
        elif line.startswith("+") and current_file and not line.startswith("+++"):
            files[current_file].append(line[1:])  # strip leading '+'
    return files


def analyze_imports(code: str) -> list[str]:
    """
    Extract all imported module names from Python source code.
    Falls back to regex when AST parsing fails (partial diffs, syntax errors).
    """
    imports: list[str] = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
    except SyntaxError:
        for line in code.splitlines():
            m = re.match(r"^\s*(?:import|from)\s+([\w.]+)", line)
            if m:
                imports.append(m.group(1).split(".")[0])
    return list(set(imports))


def count_complexity_indicators(code: str) -> dict:
    """
    Rough static complexity analysis — no execution required.
    Returns counts that help the Performance agent prioritize its attention.
    """
    lines = code.splitlines()
    return {
        "total_lines": len(lines),
        "nested_loops": sum(
            1 for l in lines if re.match(r"\s{8,}(?:for|while)\s", l)
        ),
        "string_queries": sum(
            1 for l in lines
            if re.search(r"(SELECT|INSERT|UPDATE|DELETE)", l, re.I)
            and any(marker in l for marker in ('f"', "f'", ".format(", "% "))
        ),
        "sleep_calls":  sum(1 for l in lines if "time.sleep(" in l),
        "global_vars":  sum(1 for l in lines if l.strip().startswith("global ")),
        "long_function_candidate": sum(1 for l in lines if re.match(r"\s*(?:async )?def ", l)),
    }


def check_dangerous_patterns(code: str) -> list[dict]:
    """
    Regex quick-scan for high-confidence security red flags.
    Returns list of {pattern, line_number, line_content}.
    These are hints for the Security agent — not definitive findings.
    """
    PATTERNS = [
        (r"eval\s*\(",                          "eval() usage"),
        (r"exec\s*\(",                          "exec() usage"),
        (r"subprocess.*shell\s*=\s*True",       "shell=True in subprocess"),
        (r"os\.system\s*\(",                    "os.system() call"),
        (r"pickle\.loads?\s*\(",                "pickle deserialization"),
        (r"password\s*=\s*[\"'][^\"']+[\"']",   "hardcoded password string"),
        (r"secret\s*=\s*[\"'][^\"']+[\"']",     "hardcoded secret string"),
        (r"api_key\s*=\s*[\"'][^\"']+[\"']",    "hardcoded API key"),
        (r"verify\s*=\s*False",                 "SSL verification disabled"),
        (r"yaml\.load\s*\([^,)]+\)",            "yaml.load without Loader (unsafe)"),
        (r"MD5\b|md5\s*\(",                     "MD5 usage (weak hash)"),
        (r"random\.random\s*\(",                "random.random() for security use"),
    ]
    matches = []
    for i, line in enumerate(code.splitlines(), start=1):
        for pattern, description in PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                matches.append({
                    "pattern": description,
                    "line_number": i,
                    "line_content": line.strip(),
                })
    return matches
```

---

## 4. Agent System Prompts and Implementations

### Shared LLM Call Helper

```python
import anthropic
import json
import time
from typing import Any

anthropic_client = anthropic.Anthropic()

SPECIALIST_MODEL = "claude-sonnet-4-5"
CRITIC_MODEL     = "claude-sonnet-4-5"
FORMATTER_MODEL  = "claude-sonnet-4-5"

PRICING = {
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
}


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    p = PRICING.get(model, PRICING[SPECIALIST_MODEL])
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _call_llm(
    system: str,
    user_content: str,
    model: str = SPECIALIST_MODEL,
    max_tokens: int = 4096,
) -> tuple[Any, int, int]:
    """
    Synchronous LLM call (wrapped in asyncio.to_thread by callers).
    Returns (parsed_json, input_tokens, output_tokens).
    """
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text
    in_tok  = response.usage.input_tokens
    out_tok = response.usage.output_tokens

    # Strip markdown fences if the model wrapped its JSON
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Agent returned invalid JSON: {e}\nRaw:\n{raw[:400]}")

    return parsed, in_tok, out_tok
```

### Security Agent

System prompt and implementation. The prompt is long because security requires precision — vague instructions produce vague findings.

```python
SECURITY_SYSTEM = """\
You are a security-focused code reviewer with deep expertise in the OWASP Top 10 and Python/FastAPI security patterns.

Your job: analyze the provided pull request diff and return a JSON list of security findings.

Focus areas:
1. SQL Injection — f-string or %-format queries, missing parameterization
2. Hardcoded Secrets — passwords, API keys, tokens in source code
3. Insecure Deserialization — pickle.loads on untrusted data, yaml.load without Loader
4. Path Traversal — user-controlled paths passed to open(), os.path.join()
5. Command Injection — shell=True with user input, os.system() with interpolation
6. Broken Authentication — missing auth decorators, JWT algorithms=["none"]
7. Sensitive Data Exposure — logging passwords/tokens, raw exceptions in API responses
8. SSRF — user-controlled URLs passed to httpx/requests without allowlist validation
9. Template Injection — untrusted data rendered in Jinja2 templates
10. IDOR — user-supplied IDs used without authorization check

Return a JSON array. Each item must have these exact keys:
{
  "id": "SEC-001",
  "category": "security",
  "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
  "title": "Short description",
  "description": "Full explanation of the vulnerability",
  "file": "path/to/file.py",
  "line_range": [start_line, end_line],
  "code_snippet": "offending code (max 8 lines)",
  "recommendation": "Concrete fix",
  "cwe_id": "CWE-89"
}

If you find nothing, return []. Do not invent findings. Quote the offending line.
Only return the JSON array — no markdown, no explanation."""


async def run_security_agent(
    code_files: dict[str, str],
    diff: str,
    dangerous_patterns: list[dict],
) -> tuple[list[Finding], int, int]:
    hints = ""
    if dangerous_patterns:
        hints = "\n\nPre-scan flagged these patterns (verify before reporting):\n"
        for p in dangerous_patterns[:15]:
            hints += f"  Line {p['line_number']}: {p['pattern']} — {p['line_content']}\n"

    file_ctx = "".join(
        f"\n\n### File: {fname}\n```python\n{content[:8000]}\n```"
        for fname, content in list(code_files.items())[:10]
    )

    user_content = (
        f"Review this pull request for security issues.\n\n"
        f"## Unified Diff\n```diff\n{diff[:15000]}\n```"
        f"{file_ctx}{hints}"
    )

    async with asyncio.timeout(15.0):
        raw, in_tok, out_tok = await asyncio.to_thread(
            _call_llm, SECURITY_SYSTEM, user_content
        )

    findings = []
    for i, item in enumerate(raw if isinstance(raw, list) else [], start=1):
        try:
            item.setdefault("id", f"SEC-{i:03d}")
            item["category"] = "security"
            findings.append(Finding(**item))
        except Exception:
            pass
    return findings, in_tok, out_tok
```

### Performance Agent

```python
PERFORMANCE_SYSTEM = """\
You are a performance-focused code reviewer specializing in Python async patterns,
database query optimization, and distributed systems efficiency.

Focus areas:
1. N+1 Queries — loops that call the DB once per item instead of once total (use select_related / joinedload)
2. Missing async/await — blocking sync calls inside async functions (requests.get, time.sleep, sync DB drivers)
3. Blocking I/O — synchronous file reads, CPU-bound work blocking the event loop
4. Memory leaks — unbounded lists growing in loops, global caches without size limits or TTL
5. Unnecessary recomputation — regex.compile(), model initialization, or expensive ops inside loops
6. Missing DB indices — ORM filter calls on columns that are almost certainly unindexed
7. Inefficient data structures — O(n) linear search in large lists where set() would be O(1)
8. Unbounded result sets — DB queries missing .limit() that could return millions of rows
9. Synchronous heavy work in request handlers — should be offloaded to background tasks
10. Redundant serialization — converting the same data to/from JSON multiple times per request

Return a JSON array. Each item:
{
  "id": "PERF-001",
  "category": "performance",
  "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
  "title": "Short description",
  "description": "Why this is a problem and its impact",
  "file": "path/to/file.py",
  "line_range": [start_line, end_line],
  "code_snippet": "problematic code",
  "recommendation": "Concrete fix with example"
}

If you find nothing, return []. Only return the JSON array."""


async def run_performance_agent(
    code_files: dict[str, str],
    diff: str,
    complexity: dict[str, dict],
) -> tuple[list[Finding], int, int]:
    complexity_hints = "".join(
        f"\n{fname}: {indicators}"
        for fname, indicators in complexity.items()
        if any(v > 0 for k, v in indicators.items() if k != "total_lines")
    ) or "\nNo obvious complexity issues from pre-scan."

    file_ctx = "".join(
        f"\n\n### File: {fname}\n```python\n{content[:8000]}\n```"
        for fname, content in list(code_files.items())[:10]
    )

    user_content = (
        f"Review this pull request for performance issues.\n\n"
        f"## Unified Diff\n```diff\n{diff[:15000]}\n```"
        f"{file_ctx}\n\n## Complexity Pre-scan\n{complexity_hints}"
    )

    async with asyncio.timeout(15.0):
        raw, in_tok, out_tok = await asyncio.to_thread(
            _call_llm, PERFORMANCE_SYSTEM, user_content
        )

    findings = []
    for i, item in enumerate(raw if isinstance(raw, list) else [], start=1):
        try:
            item.setdefault("id", f"PERF-{i:03d}")
            item["category"] = "performance"
            findings.append(Finding(**item))
        except Exception:
            pass
    return findings, in_tok, out_tok
```

### Style Agent

```python
STYLE_SYSTEM = """\
You are a code quality reviewer focused on Python best practices, maintainability, and long-term health.

Focus areas:
1. Missing type hints — function parameters and return values without annotations
2. Missing docstrings — public functions/classes/modules without docstrings
3. Dead code — commented-out code blocks, variables assigned but never read
4. Inconsistent naming — mixing camelCase/snake_case, non-descriptive names (x, tmp, data2)
5. Functions too long — functions over 40 lines that should be decomposed
6. Magic numbers — literal numbers in logic without named constants
7. Mutable default arguments — def foo(items=[]) is a classic Python bug
8. Bare except clauses — except: without specifying the exception type
9. God classes — classes doing too many unrelated things (>10 public methods, >5 concerns)
10. Missing __all__ — public modules without explicit export control

Severity guidance:
- MEDIUM: missing type hints/docstrings on public APIs, mutable defaults, bare except
- LOW: naming inconsistencies, magic numbers, minor style issues
- INFO: purely subjective preferences

Return a JSON array. Each item:
{
  "id": "STYLE-001",
  "category": "style",
  "severity": "MEDIUM|LOW|INFO",
  "title": "Short description",
  "description": "Why this matters",
  "file": "path/to/file.py",
  "line_range": [start_line, end_line],
  "code_snippet": "problematic code",
  "recommendation": "How to fix it"
}

If you find nothing, return []. Only return the JSON array."""


async def run_style_agent(
    code_files: dict[str, str],
    diff: str,
) -> tuple[list[Finding], int, int]:
    file_ctx = "".join(
        f"\n\n### File: {fname}\n```python\n{content[:8000]}\n```"
        for fname, content in list(code_files.items())[:10]
    )

    user_content = (
        f"Review this pull request for code style and quality issues.\n\n"
        f"## Unified Diff\n```diff\n{diff[:15000]}\n```{file_ctx}"
    )

    async with asyncio.timeout(15.0):
        raw, in_tok, out_tok = await asyncio.to_thread(
            _call_llm, STYLE_SYSTEM, user_content
        )

    findings = []
    for i, item in enumerate(raw if isinstance(raw, list) else [], start=1):
        try:
            item.setdefault("id", f"STYLE-{i:03d}")
            item["category"] = "style"
            findings.append(Finding(**item))
        except Exception:
            pass
    return findings, in_tok, out_tok
```

### Critic Agent

The Critic receives all specialist findings, deduplicates, verifies severity, and re-ranks. It is the quality gate that prevents the final report from being a raw dump of every potential issue from three independent LLM calls.

```python
CRITIC_SYSTEM = """\
You are a senior engineering lead reviewing AI-generated code review findings.

Your job:
1. Remove exact duplicates — same issue, same file, same approximate line
2. Merge near-duplicates — same issue reported by two agents at different severities (keep the higher severity, combine descriptions)
3. Verify severity levels:
   - CRITICAL: exploitable vulnerability, data loss risk, immediate production impact
   - HIGH: serious issue that should block merge
   - MEDIUM: should be fixed before merge but not blocking
   - LOW: nice to fix, not blocking
   - INFO: purely informational
4. Add missing context or clarification to vague descriptions
5. Sort result: CRITICAL → HIGH → MEDIUM → LOW → INFO

Input: JSON object with keys "security_findings", "performance_findings", "style_findings".
Output: {"ranked_findings": [...]} with deduplicated, ranked findings.
Mark kept findings with "is_duplicate": false.
Only return the JSON object — no markdown, no explanation."""


async def run_critic_agent(
    security_findings: list[Finding],
    performance_findings: list[Finding],
    style_findings: list[Finding],
) -> tuple[list[Finding], int, int]:
    payload = {
        "security_findings":    [f.model_dump(exclude_none=True) for f in security_findings],
        "performance_findings": [f.model_dump(exclude_none=True) for f in performance_findings],
        "style_findings":       [f.model_dump(exclude_none=True) for f in style_findings],
    }
    user_content = f"Review and deduplicate these findings:\n\n{json.dumps(payload, indent=2)}"

    async with asyncio.timeout(20.0):
        raw, in_tok, out_tok = await asyncio.to_thread(
            _call_llm, CRITIC_SYSTEM, user_content, CRITIC_MODEL, 6000
        )

    ranked: list[Finding] = []
    for item in raw.get("ranked_findings", []):
        try:
            ranked.append(Finding(**item))
        except Exception:
            pass
    return ranked, in_tok, out_tok
```

### Formatter Agent

The Formatter takes the ranked findings list and produces the human-facing portions of the report: the executive summary, quick wins, and praise. It does not invent findings — it only shapes the presentation of what the Critic already validated.

```python
FORMATTER_SYSTEM = """\
You are a technical writer producing a developer-facing code review report.

Given ranked findings and PR metadata, produce a concise report with these components:

1. executive_summary: 2-3 sentences summarizing the overall review. Be direct.
   Good: "This PR introduces a SQL injection vulnerability (CRITICAL) via an f-string query in user_repo.py.
   Three missing type hints and one blocking DB call were also found. Approve after fixing SEC-001."
   Bad: "There are some issues in this PR that should be addressed."

2. quick_wins: The 3-5 findings that are lowest-effort, highest-impact to fix.
   Format: [{"title": "...", "description": "...", "effort": "1-line fix | 5-min refactor"}]

3. praise: 2-4 specific things done well. Name the file or pattern. Do not write generic praise.
   Good: "Parameterized queries used correctly in user_repo.py line 45"
   Bad: "Good code quality overall"
   If nothing stands out, mention structural positives (tests added, docstrings present, etc.)

Output exactly this JSON object:
{
  "executive_summary": "...",
  "quick_wins": [...],
  "praise": [...]
}

Only return this JSON object — no markdown, no explanation."""


async def run_formatter_agent(
    ranked_findings: list[Finding],
    pr_metadata: dict,
    code_files: dict[str, str],
) -> tuple[dict, int, int]:
    findings_json = json.dumps(
        [f.model_dump(exclude_none=True) for f in ranked_findings[:30]], indent=2
    )
    user_content = (
        f"PR Metadata: {json.dumps(pr_metadata)}\n"
        f"Files changed: {list(code_files.keys())}\n\n"
        f"Ranked findings ({len(ranked_findings)} total):\n{findings_json}\n\n"
        f"Produce the review report JSON."
    )

    async with asyncio.timeout(15.0):
        raw, in_tok, out_tok = await asyncio.to_thread(
            _call_llm, FORMATTER_SYSTEM, user_content, FORMATTER_MODEL, 3000
        )
    return raw, in_tok, out_tok
```

---

## 5. LangGraph Assembly

```python
from langgraph.graph import StateGraph, END
from langgraph.types import Send


# ── Node: Fetch PR data ────────────────────────────────────────────────────

async def fetch_node(state: AgentState) -> dict:
    """Parse PR URL, fetch diff + metadata + file contents in parallel."""
    pr_url = state["pr_url"]
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url)
    if not m:
        raise ValueError(f"Cannot parse PR URL: {pr_url}")
    repo      = m.group(1)
    pr_number = int(m.group(2))

    diff, metadata, code_files = await asyncio.gather(
        fetch_pr_diff(repo, pr_number),
        fetch_pr_metadata(repo, pr_number),
        get_changed_files(repo, pr_number),
    )

    return {
        "repo":       repo,
        "pr_number":  pr_number,
        "commit_sha": metadata["commit_sha"],
        "pr_diff":    diff,
        "code_files": code_files,
        "start_time": time.monotonic(),
    }


# ── Fan-out router ─────────────────────────────────────────────────────────

def fan_out(state: AgentState) -> list[Send]:
    """
    Send the same state to all three specialists simultaneously.
    LangGraph executes these branches in parallel and waits for all to complete
    before moving to the next node ("critic"). The Annotated[list, operator.add]
    reducers on the findings fields merge results from all branches.
    """
    return [
        Send("security_node",    state),
        Send("performance_node", state),
        Send("style_node",       state),
    ]


# ── Specialist nodes ───────────────────────────────────────────────────────

async def security_node(state: AgentState) -> dict:
    try:
        patterns = check_dangerous_patterns(state["pr_diff"])
        findings, in_tok, out_tok = await run_security_agent(
            state["code_files"], state["pr_diff"], patterns
        )
        return {
            "security_findings":   findings,
            "agents_run":          ["security"],
            "total_input_tokens":  in_tok,
            "total_output_tokens": out_tok,
        }
    except Exception as e:
        return {
            "security_findings": [],
            "agents_failed": ["security"],
            "errors": [f"Security agent failed: {e}"],
        }


async def performance_node(state: AgentState) -> dict:
    try:
        complexity = {
            fname: count_complexity_indicators(content)
            for fname, content in list(state["code_files"].items())[:5]
        }
        findings, in_tok, out_tok = await run_performance_agent(
            state["code_files"], state["pr_diff"], complexity
        )
        return {
            "performance_findings": findings,
            "agents_run":           ["performance"],
            "total_input_tokens":   in_tok,
            "total_output_tokens":  out_tok,
        }
    except Exception as e:
        return {
            "performance_findings": [],
            "agents_failed": ["performance"],
            "errors": [f"Performance agent failed: {e}"],
        }


async def style_node(state: AgentState) -> dict:
    try:
        findings, in_tok, out_tok = await run_style_agent(
            state["code_files"], state["pr_diff"]
        )
        return {
            "style_findings":      findings,
            "agents_run":          ["style"],
            "total_input_tokens":  in_tok,
            "total_output_tokens": out_tok,
        }
    except Exception as e:
        return {
            "style_findings": [],
            "agents_failed": ["style"],
            "errors": [f"Style agent failed: {e}"],
        }


# ── Critic node ────────────────────────────────────────────────────────────

async def critic_node(state: AgentState) -> dict:
    try:
        ranked, in_tok, out_tok = await run_critic_agent(
            state.get("security_findings", []),
            state.get("performance_findings", []),
            state.get("style_findings", []),
        )
        return {
            "all_findings":        ranked,
            "agents_run":          ["critic"],
            "total_input_tokens":  in_tok,
            "total_output_tokens": out_tok,
        }
    except Exception as e:
        # Degrade: concatenate all findings without deduplication
        all_f = (
            state.get("security_findings", []) +
            state.get("performance_findings", []) +
            state.get("style_findings", [])
        )
        return {
            "all_findings":  all_f,
            "agents_failed": ["critic"],
            "errors": [f"Critic degraded to raw merge: {e}"],
        }


# ── Formatter node ─────────────────────────────────────────────────────────

async def formatter_node(state: AgentState) -> dict:
    all_findings = state.get("all_findings", [])
    pr_metadata  = {
        "pr_url":     state["pr_url"],
        "repo":       state["repo"],
        "pr_number":  state["pr_number"],
    }

    try:
        report_parts, in_tok, out_tok = await run_formatter_agent(
            all_findings, pr_metadata, state["code_files"]
        )
    except Exception as e:
        report_parts = {
            "executive_summary": f"Review found {len(all_findings)} issues. Formatter error: {e}",
            "quick_wins": [],
            "praise": [],
        }
        in_tok, out_tok = 0, 0

    total_in  = state.get("total_input_tokens", 0) + in_tok
    total_out = state.get("total_output_tokens", 0) + out_tok
    duration  = time.monotonic() - state.get("start_time", time.monotonic())

    active_findings = [f for f in all_findings if not f.is_duplicate]

    by_sev: dict[Severity, list[Finding]] = {s: [] for s in Severity}
    by_cat: dict[Category, int] = {c: 0 for c in Category}
    for f in active_findings:
        by_sev[f.severity].append(f)
        by_cat[f.category] = by_cat.get(f.category, 0) + 1

    report = ReviewReport(
        pr_url=state["pr_url"],
        repo=state["repo"],
        pr_number=state["pr_number"],
        commit_sha=state.get("commit_sha", "unknown"),
        reviewed_at=datetime.utcnow(),
        executive_summary=report_parts.get("executive_summary", ""),
        findings_by_severity=by_sev,
        all_findings=active_findings,
        quick_wins=[QuickWin(**q) for q in report_parts.get("quick_wins", [])],
        praise=[Praise(**p) for p in report_parts.get("praise", [])],
        total_findings=len(active_findings),
        findings_by_category=by_cat,
        metadata=ReviewMetadata(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            estimated_cost_usd=_estimate_cost(total_in, total_out, FORMATTER_MODEL),
            duration_seconds=duration,
            agents_run=state.get("agents_run", []),
            agents_failed=state.get("agents_failed", []),
        ),
    )

    return {
        "final_report":        report,
        "agents_run":          ["formatter"],
        "total_input_tokens":  in_tok,
        "total_output_tokens": out_tok,
    }


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    # Register each node: name → async function
    g.add_node("fetch",            fetch_node)
    g.add_node("security_node",    security_node)
    g.add_node("performance_node", performance_node)
    g.add_node("style_node",       style_node)
    g.add_node("critic",           critic_node)
    g.add_node("formatter",        formatter_node)

    # Entry point: graph starts at "fetch"
    g.set_entry_point("fetch")

    # After fetch: fan-out to all three specialists in parallel.
    # fan_out() returns a list of Send() objects. LangGraph executes them
    # simultaneously and waits for all to complete before proceeding.
    g.add_conditional_edges(
        "fetch", fan_out,
        ["security_node", "performance_node", "style_node"]
    )

    # All three specialists converge at critic.
    # LangGraph automatically waits for all incoming branches before running critic.
    g.add_edge("security_node",    "critic")
    g.add_edge("performance_node", "critic")
    g.add_edge("style_node",       "critic")

    g.add_edge("critic",    "formatter")
    g.add_edge("formatter", END)

    return g.compile()


REVIEW_GRAPH = build_graph()
```

---

## 6. FastAPI Application

```python
import uuid
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as PydanticBase
from collections import defaultdict

app = FastAPI(title="Code Review Agent", version="1.0.0")

# In-memory store — replace with Redis for multi-replica deployments
jobs:       dict[str, dict]           = {}
job_queues: dict[str, asyncio.Queue]  = defaultdict(asyncio.Queue)


class ReviewRequest(PydanticBase):
    pr_url: str


@app.post("/review")
async def create_review(req: ReviewRequest, background_tasks: BackgroundTasks):
    """Enqueue a review job. Returns job_id immediately; processing happens in background."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "pr_url": req.pr_url, "result": None, "error": None}
    background_tasks.add_task(_run_review_job, job_id, req.pr_url)
    return {"job_id": job_id, "status": "queued"}


@app.get("/review/{job_id}/stream")
async def stream_review(job_id: str):
    """SSE stream of agent progress events. Connect before or after job starts."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = job_queues[job_id]

    async def event_generator():
        while True:
            item = await queue.get()
            if item is None:   # sentinel: job is done
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/review/{job_id}")
async def get_review(job_id: str):
    """Poll for review result. Returns partial status while running."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error", "Unknown error"))
    return {"job_id": job_id, "status": job["status"], "result": job.get("result")}


@app.get("/health")
async def health():
    running = sum(1 for j in jobs.values() if j["status"] == "running")
    return {"status": "ok", "running_jobs": running, "total_jobs": len(jobs)}


async def _run_review_job(job_id: str, pr_url: str) -> None:
    """Background coroutine: executes the full LangGraph review pipeline."""
    queue = job_queues[job_id]
    jobs[job_id]["status"] = "running"

    async def emit(event: str, **kwargs):
        payload = json.dumps({"event": event, "timestamp": datetime.utcnow().isoformat(), **kwargs})
        await queue.put(payload)

    try:
        await emit("review_start", pr_url=pr_url)

        initial_state: AgentState = {
            "pr_url": pr_url, "repo": "", "pr_number": 0, "commit_sha": "",
            "pr_diff": "", "code_files": {}, "security_findings": [],
            "performance_findings": [], "style_findings": [], "all_findings": [],
            "final_report": None, "agents_run": [], "agents_failed": [],
            "total_input_tokens": 0, "total_output_tokens": 0,
            "start_time": time.monotonic(), "errors": [],
        }

        async with asyncio.timeout(60.0):
            result = await REVIEW_GRAPH.ainvoke(initial_state)

        report: ReviewReport = result["final_report"]

        # Emit per-agent completion events
        for agent, findings_key in [
            ("security",    "security_findings"),
            ("performance", "performance_findings"),
            ("style",       "style_findings"),
        ]:
            count = len(result.get(findings_key, []))
            await emit("agent_done", agent=agent, findings_count=count)

        await emit(
            "review_complete",
            total_findings=report.total_findings,
            critical=len(report.findings_by_severity.get(Severity.CRITICAL, [])),
            high=len(report.findings_by_severity.get(Severity.HIGH, [])),
            cost_usd=report.metadata.estimated_cost_usd,
            duration_seconds=report.metadata.duration_seconds,
        )

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = report.model_dump(mode="json")

    except asyncio.TimeoutError:
        jobs[job_id].update({"status": "failed", "error": "Review timed out after 60s"})
        await emit("review_failed", reason="timeout")
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})
        await emit("review_failed", reason=str(e))
    finally:
        await queue.put(None)  # signal SSE stream to close
```

**SSE event format — each line sent to connected clients:**

```json
{"event": "review_start",    "timestamp": "2026-04-06T12:00:00", "pr_url": "https://github.com/owner/repo/pull/42"}
{"event": "agent_done",      "timestamp": "2026-04-06T12:00:08", "agent": "security",     "findings_count": 3}
{"event": "agent_done",      "timestamp": "2026-04-06T12:00:08", "agent": "performance",  "findings_count": 2}
{"event": "agent_done",      "timestamp": "2026-04-06T12:00:09", "agent": "style",        "findings_count": 5}
{"event": "review_complete", "timestamp": "2026-04-06T12:00:12", "total_findings": 8, "critical": 1, "high": 2, "cost_usd": 0.038, "duration_seconds": 12.4}
```

---

## 7. Complete main.py

```python
# main.py — Autonomous Code Review Agent
# Run: uvicorn main:app --reload
# Install: pip install anthropic langgraph fastapi uvicorn httpx pydantic redis

from __future__ import annotations

import ast
import asyncio
import json
import operator
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional

import anthropic
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ── Config ──────────────────────────────────────────────────────────────────

GITHUB_TOKEN   = os.environ["GITHUB_TOKEN"]
GITHUB_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

SPECIALIST_MODEL    = "claude-sonnet-4-5"
CRITIC_MODEL        = "claude-sonnet-4-5"
FORMATTER_MODEL     = "claude-sonnet-4-5"
MAX_COST_PER_REVIEW = float(os.environ.get("MAX_COST_USD", "0.50"))
MAX_FILES           = int(os.environ.get("MAX_FILES", "10"))

PRICING = {
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-5":   {"input": 15.00, "output": 75.00},
}

anthropic_client = anthropic.Anthropic()

# ── Pydantic Models ──────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"

class Category(str, Enum):
    SECURITY    = "security"
    PERFORMANCE = "performance"
    STYLE       = "style"

class Finding(BaseModel):
    id: str = Field(default="UNKNOWN-001")
    category: Category
    severity: Severity
    title: str
    description: str
    file: str
    line_range: Optional[tuple[int, int]] = None
    code_snippet: Optional[str] = None
    recommendation: str
    cwe_id: Optional[str] = None
    is_duplicate: bool = False

class QuickWin(BaseModel):
    title: str
    description: str
    effort: str

class Praise(BaseModel):
    title: str
    description: str

class ReviewMetadata(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    duration_seconds: float
    agents_run: list[str]
    agents_failed: list[str]
    cache_hit: bool = False

class ReviewReport(BaseModel):
    pr_url: str
    repo: str
    pr_number: int
    commit_sha: str
    reviewed_at: datetime
    executive_summary: str
    findings_by_severity: dict[Severity, list[Finding]]
    all_findings: list[Finding]
    quick_wins: list[QuickWin]
    praise: list[Praise]
    total_findings: int
    findings_by_category: dict[Category, int]
    metadata: ReviewMetadata

# ── LangGraph State ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    pr_url: str
    repo: str
    pr_number: int
    commit_sha: str
    pr_diff: str
    code_files: dict[str, str]
    security_findings:    Annotated[list[Finding], operator.add]
    performance_findings: Annotated[list[Finding], operator.add]
    style_findings:       Annotated[list[Finding], operator.add]
    all_findings: list[Finding]
    final_report: Optional[ReviewReport]
    agents_run:           Annotated[list[str], operator.add]
    agents_failed:        Annotated[list[str], operator.add]
    total_input_tokens:   Annotated[int, operator.add]
    total_output_tokens:  Annotated[int, operator.add]
    start_time: float
    errors:               Annotated[list[str], operator.add]

# ── Utilities ────────────────────────────────────────────────────────────────

def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    p = PRICING.get(model, PRICING[SPECIALIST_MODEL])
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

def _call_llm(
    system: str,
    user_content: str,
    model: str = SPECIALIST_MODEL,
    max_tokens: int = 4096,
) -> tuple[Any, int, int]:
    response = anthropic_client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    return json.loads(raw), response.usage.input_tokens, response.usage.output_tokens

# ── GitHub Tools ─────────────────────────────────────────────────────────────

async def fetch_pr_diff(repo: str, pr_number: int) -> str:
    headers = {**GITHUB_HEADERS, "Accept": "application/vnd.github.v3.diff"}
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}", headers=headers
        )
        r.raise_for_status()
        return r.text

async def fetch_pr_metadata(repo: str, pr_number: int) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}", headers=GITHUB_HEADERS
        )
        r.raise_for_status()
        d = r.json()
        return {
            "commit_sha": d["head"]["sha"],
            "title":      d["title"],
            "author":     d["user"]["login"],
        }

async def get_changed_files(repo: str, pr_number: int) -> dict[str, str]:
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files",
            headers=GITHUB_HEADERS,
        )
        r.raise_for_status()
        files = r.json()

    async def _fetch(fname: str, raw_url: str) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(raw_url, headers=GITHUB_HEADERS)
            try:
                return fname, r.text if r.status_code == 200 else "[fetch failed]"
            except Exception:
                return fname, "[binary]"

    tasks = [
        _fetch(f["filename"], f["raw_url"])
        for f in files
        if f.get("status") != "removed" and f.get("raw_url")
    ]
    results = await asyncio.gather(*tasks[:MAX_FILES], return_exceptions=True)
    return {
        fname: content
        for item in results
        if isinstance(item, tuple)
        for fname, content in [item]
    }

def check_dangerous_patterns(code: str) -> list[dict]:
    PATTERNS = [
        (r"eval\s*\(",                        "eval() usage"),
        (r"exec\s*\(",                        "exec() usage"),
        (r"subprocess.*shell\s*=\s*True",     "shell=True"),
        (r"os\.system\s*\(",                  "os.system()"),
        (r"pickle\.loads?\s*\(",              "pickle deserialization"),
        (r"password\s*=\s*[\"'][^\"']+[\"']", "hardcoded password"),
        (r"api_key\s*=\s*[\"'][^\"']+[\"']",  "hardcoded API key"),
        (r"verify\s*=\s*False",               "SSL verification disabled"),
        (r"yaml\.load\s*\([^,)]+\)",          "yaml.load without Loader"),
    ]
    matches = []
    for i, line in enumerate(code.splitlines(), 1):
        for pattern, desc in PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                matches.append({
                    "pattern": desc,
                    "line_number": i,
                    "line_content": line.strip(),
                })
    return matches

def count_complexity_indicators(code: str) -> dict:
    lines = code.splitlines()
    return {
        "total_lines":    len(lines),
        "nested_loops":   sum(1 for l in lines if re.match(r"\s{8,}(?:for|while)\s", l)),
        "string_queries": sum(
            1 for l in lines
            if re.search(r"(SELECT|INSERT|UPDATE|DELETE)", l, re.I)
            and any(m in l for m in ('f"', "f'", ".format(", "% "))
        ),
        "sleep_calls":    sum(1 for l in lines if "time.sleep(" in l),
        "global_vars":    sum(1 for l in lines if l.strip().startswith("global ")),
    }

# ── Agent System Prompts ──────────────────────────────────────────────────────

SECURITY_SYSTEM = """\
You are a security code reviewer (OWASP Top 10 expert).
Analyze for: SQL injection, hardcoded secrets, insecure deserialization, path traversal,
command injection, broken auth, sensitive data exposure, SSRF, template injection, IDOR.
Return JSON array: [{id, category:"security", severity, title, description, file,
line_range, code_snippet, recommendation, cwe_id}].
Return [] if nothing found. Only the JSON array."""

PERFORMANCE_SYSTEM = """\
You are a performance code reviewer specializing in Python async and database patterns.
Analyze for: N+1 queries, blocking I/O in async functions, missing pagination, memory leaks,
expensive ops in loops, unindexed DB filters, O(n) list search, unbounded result sets.
Return JSON array: [{id, category:"performance", severity, title, description, file,
line_range, code_snippet, recommendation}].
Return [] if nothing found. Only the JSON array."""

STYLE_SYSTEM = """\
You are a code quality reviewer focused on Python best practices and maintainability.
Analyze for: missing type hints, missing docstrings, dead code, inconsistent naming,
functions >40 lines, magic numbers, mutable default args, bare except clauses.
Severity: MEDIUM for public API issues, LOW for minor style, INFO for opinions.
Return JSON array: [{id, category:"style", severity, title, description, file,
line_range, code_snippet, recommendation}].
Return [] if nothing found. Only the JSON array."""

CRITIC_SYSTEM = """\
You are a senior engineer deduplicating AI code review findings.
1. Remove exact duplicates. 2. Merge near-duplicates (keep higher severity).
3. Verify severity: CRITICAL=exploitable, HIGH=blocks merge, MEDIUM=fix before merge,
   LOW=nice-to-have, INFO=informational. 4. Sort: CRITICAL→HIGH→MEDIUM→LOW→INFO.
Input: JSON with security_findings, performance_findings, style_findings.
Output: {"ranked_findings": [...]}. Only this JSON object."""

FORMATTER_SYSTEM = """\
You are writing a developer-facing code review report.
Given ranked findings, produce exactly:
{"executive_summary": "2-3 direct sentences naming the most critical issue",
 "quick_wins": [{"title","description","effort"}],
 "praise": [{"title","description"}]}.
Be specific in praise — name the file and line. Only this JSON object."""

# ── Agent Runners ─────────────────────────────────────────────────────────────

def _build_context(diff: str, code_files: dict[str, str], extra: str = "") -> str:
    file_ctx = "".join(
        f"\n\n### {fname}\n```python\n{content[:8000]}\n```"
        for fname, content in list(code_files.items())[:MAX_FILES]
    )
    return f"## Diff\n```diff\n{diff[:15000]}\n```{file_ctx}{extra}"

async def _run_specialist(
    system: str,
    user_content: str,
    prefix: str,
    category: str,
) -> tuple[list[Finding], int, int]:
    async with asyncio.timeout(15.0):
        raw, in_tok, out_tok = await asyncio.to_thread(_call_llm, system, user_content)
    findings = []
    for i, item in enumerate(raw if isinstance(raw, list) else [], 1):
        try:
            item.setdefault("id", f"{prefix}-{i:03d}")
            item["category"] = category
            findings.append(Finding(**item))
        except Exception:
            pass
    return findings, in_tok, out_tok

# ── Graph Nodes ───────────────────────────────────────────────────────────────

async def fetch_node(state: AgentState) -> dict:
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)", state["pr_url"])
    if not m:
        raise ValueError(f"Bad PR URL: {state['pr_url']}")
    repo, pr_number = m.group(1), int(m.group(2))
    diff, meta, files = await asyncio.gather(
        fetch_pr_diff(repo, pr_number),
        fetch_pr_metadata(repo, pr_number),
        get_changed_files(repo, pr_number),
    )
    return {
        "repo":       repo,
        "pr_number":  pr_number,
        "commit_sha": meta["commit_sha"],
        "pr_diff":    diff,
        "code_files": files,
        "start_time": time.monotonic(),
    }

def fan_out(state: AgentState) -> list[Send]:
    return [
        Send("security_node",    state),
        Send("performance_node", state),
        Send("style_node",       state),
    ]

async def security_node(state: AgentState) -> dict:
    try:
        hints = "\n\nPre-scan: " + json.dumps(check_dangerous_patterns(state["pr_diff"])[:10])
        findings, in_tok, out_tok = await _run_specialist(
            SECURITY_SYSTEM,
            _build_context(state["pr_diff"], state["code_files"], hints),
            "SEC", "security",
        )
        return {
            "security_findings":   findings,
            "agents_run":          ["security"],
            "total_input_tokens":  in_tok,
            "total_output_tokens": out_tok,
        }
    except Exception as e:
        return {"security_findings": [], "agents_failed": ["security"], "errors": [str(e)]}

async def performance_node(state: AgentState) -> dict:
    try:
        cx = {f: count_complexity_indicators(c) for f, c in list(state["code_files"].items())[:5]}
        extra = f"\n\nComplexity pre-scan: {json.dumps(cx)}"
        findings, in_tok, out_tok = await _run_specialist(
            PERFORMANCE_SYSTEM,
            _build_context(state["pr_diff"], state["code_files"], extra),
            "PERF", "performance",
        )
        return {
            "performance_findings": findings,
            "agents_run":           ["performance"],
            "total_input_tokens":   in_tok,
            "total_output_tokens":  out_tok,
        }
    except Exception as e:
        return {"performance_findings": [], "agents_failed": ["performance"], "errors": [str(e)]}

async def style_node(state: AgentState) -> dict:
    try:
        findings, in_tok, out_tok = await _run_specialist(
            STYLE_SYSTEM,
            _build_context(state["pr_diff"], state["code_files"]),
            "STYLE", "style",
        )
        return {
            "style_findings":      findings,
            "agents_run":          ["style"],
            "total_input_tokens":  in_tok,
            "total_output_tokens": out_tok,
        }
    except Exception as e:
        return {"style_findings": [], "agents_failed": ["style"], "errors": [str(e)]}

async def critic_node(state: AgentState) -> dict:
    try:
        payload = {
            "security_findings":    [f.model_dump() for f in state.get("security_findings", [])],
            "performance_findings": [f.model_dump() for f in state.get("performance_findings", [])],
            "style_findings":       [f.model_dump() for f in state.get("style_findings", [])],
        }
        async with asyncio.timeout(20.0):
            raw, in_tok, out_tok = await asyncio.to_thread(
                _call_llm, CRITIC_SYSTEM, json.dumps(payload), CRITIC_MODEL, 6000
            )
        ranked = [Finding(**item) for item in raw.get("ranked_findings", [])]
        return {
            "all_findings":        ranked,
            "agents_run":          ["critic"],
            "total_input_tokens":  in_tok,
            "total_output_tokens": out_tok,
        }
    except Exception as e:
        all_f = (
            state.get("security_findings", []) +
            state.get("performance_findings", []) +
            state.get("style_findings", [])
        )
        return {"all_findings": all_f, "agents_failed": ["critic"], "errors": [str(e)]}

async def formatter_node(state: AgentState) -> dict:
    all_findings = state.get("all_findings", [])
    try:
        user_content = (
            f"PR: {state['pr_url']}\nFiles: {list(state['code_files'].keys())}\n\n"
            f"Findings ({len(all_findings)}):\n"
            f"{json.dumps([f.model_dump() for f in all_findings[:30]], indent=2)}"
        )
        async with asyncio.timeout(15.0):
            raw, in_tok, out_tok = await asyncio.to_thread(
                _call_llm, FORMATTER_SYSTEM, user_content, FORMATTER_MODEL, 3000
            )
    except Exception as e:
        raw = {
            "executive_summary": f"{len(all_findings)} findings. Formatter error: {e}",
            "quick_wins": [],
            "praise": [],
        }
        in_tok, out_tok = 0, 0

    total_in  = state.get("total_input_tokens", 0) + in_tok
    total_out = state.get("total_output_tokens", 0) + out_tok
    duration  = time.monotonic() - state.get("start_time", time.monotonic())
    active    = [f for f in all_findings if not f.is_duplicate]

    by_sev: dict[Severity, list[Finding]] = {s: [] for s in Severity}
    by_cat: dict[Category, int] = {c: 0 for c in Category}
    for f in active:
        by_sev[f.severity].append(f)
        by_cat[f.category] = by_cat.get(f.category, 0) + 1

    report = ReviewReport(
        pr_url=state["pr_url"],
        repo=state["repo"],
        pr_number=state["pr_number"],
        commit_sha=state.get("commit_sha", ""),
        reviewed_at=datetime.utcnow(),
        executive_summary=raw.get("executive_summary", ""),
        findings_by_severity=by_sev,
        all_findings=active,
        quick_wins=[QuickWin(**q) for q in raw.get("quick_wins", [])],
        praise=[Praise(**p) for p in raw.get("praise", [])],
        total_findings=len(active),
        findings_by_category=by_cat,
        metadata=ReviewMetadata(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            estimated_cost_usd=_estimate_cost(total_in, total_out, FORMATTER_MODEL),
            duration_seconds=duration,
            agents_run=state.get("agents_run", []),
            agents_failed=state.get("agents_failed", []),
        ),
    )
    return {
        "final_report":        report,
        "agents_run":          ["formatter"],
        "total_input_tokens":  in_tok,
        "total_output_tokens": out_tok,
    }

# ── Graph Build ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    for name, func in [
        ("fetch",            fetch_node),
        ("security_node",    security_node),
        ("performance_node", performance_node),
        ("style_node",       style_node),
        ("critic",           critic_node),
        ("formatter",        formatter_node),
    ]:
        g.add_node(name, func)
    g.set_entry_point("fetch")
    g.add_conditional_edges(
        "fetch", fan_out,
        ["security_node", "performance_node", "style_node"]
    )
    g.add_edge("security_node",    "critic")
    g.add_edge("performance_node", "critic")
    g.add_edge("style_node",       "critic")
    g.add_edge("critic",    "formatter")
    g.add_edge("formatter", END)
    return g.compile()

REVIEW_GRAPH = build_graph()

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Code Review Agent", version="1.0.0")
jobs:       dict[str, dict]          = {}
job_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

class ReviewRequest(BaseModel):
    pr_url: str

@app.post("/review")
async def create_review(req: ReviewRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "pr_url": req.pr_url, "result": None, "error": None}
    background_tasks.add_task(_run_review_job, job_id, req.pr_url)
    return {"job_id": job_id, "status": "queued"}

@app.get("/review/{job_id}/stream")
async def stream_review(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    q = job_queues[job_id]
    async def gen():
        while True:
            item = await q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/review/{job_id}")
async def get_review(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] == "failed":
        raise HTTPException(500, job.get("error"))
    return {"job_id": job_id, "status": job["status"], "result": job.get("result")}

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "jobs":    len(jobs),
        "running": sum(1 for j in jobs.values() if j["status"] == "running"),
    }

async def _run_review_job(job_id: str, pr_url: str) -> None:
    q = job_queues[job_id]
    jobs[job_id]["status"] = "running"

    async def emit(event: str, **kw):
        await q.put(json.dumps({"event": event, "ts": datetime.utcnow().isoformat(), **kw}))

    try:
        await emit("review_start", pr_url=pr_url)
        state: AgentState = {
            "pr_url": pr_url, "repo": "", "pr_number": 0, "commit_sha": "",
            "pr_diff": "", "code_files": {}, "security_findings": [],
            "performance_findings": [], "style_findings": [], "all_findings": [],
            "final_report": None, "agents_run": [], "agents_failed": [],
            "total_input_tokens": 0, "total_output_tokens": 0,
            "start_time": time.monotonic(), "errors": [],
        }
        async with asyncio.timeout(60.0):
            result = await REVIEW_GRAPH.ainvoke(state)

        report: ReviewReport = result["final_report"]
        if report.metadata.estimated_cost_usd > MAX_COST_PER_REVIEW:
            raise ValueError(
                f"Cost ${report.metadata.estimated_cost_usd:.3f} exceeded "
                f"budget ${MAX_COST_PER_REVIEW}"
            )

        await emit(
            "review_complete",
            total_findings=report.total_findings,
            critical=len(report.findings_by_severity.get(Severity.CRITICAL, [])),
            cost_usd=report.metadata.estimated_cost_usd,
            duration_seconds=report.metadata.duration_seconds,
        )
        jobs[job_id].update({
            "status": "complete",
            "result": report.model_dump(mode="json"),
        })
    except asyncio.TimeoutError:
        jobs[job_id].update({"status": "failed", "error": "timeout"})
        await emit("review_failed", reason="timeout")
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})
        await emit("review_failed", reason=str(e))
    finally:
        await q.put(None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 8. Testing

### Unit Tests

```python
# tests/test_tools.py
import pytest
from main import check_dangerous_patterns, count_complexity_indicators, analyze_imports


def test_finds_eval():
    assert any(
        m["pattern"] == "eval() usage"
        for m in check_dangerous_patterns("result = eval(user_input)")
    )

def test_finds_hardcoded_api_key():
    code = 'api_key = "sk-1234567890abcdef"'
    assert any("API key" in m["pattern"] for m in check_dangerous_patterns(code))

def test_finds_ssl_disabled():
    code = "requests.get(url, verify=False)"
    assert any("SSL" in m["pattern"] for m in check_dangerous_patterns(code))

def test_finds_pickle():
    code = "data = pickle.loads(untrusted_bytes)"
    assert any("pickle" in m["pattern"] for m in check_dangerous_patterns(code))

def test_clean_code_no_flags():
    code = "def add(a: int, b: int) -> int:\n    return a + b"
    assert check_dangerous_patterns(code) == []

def test_detects_sleep_in_async():
    code = "async def handler():\n    time.sleep(5)\n    return {}"
    assert count_complexity_indicators(code)["sleep_calls"] == 1

def test_detects_string_query():
    code = 'query = f"SELECT * FROM users WHERE id = {user_id}"'
    assert count_complexity_indicators(code)["string_queries"] == 1

def test_detects_nested_loops():
    code = "        for z in items:\n            pass"
    assert count_complexity_indicators(code)["nested_loops"] == 1

def test_analyze_imports_standard():
    code = "import os\nimport sys\nfrom typing import List"
    imports = analyze_imports(code)
    assert "os" in imports
    assert "sys" in imports

def test_analyze_imports_syntax_error_fallback():
    # Partial diff lines — not valid Python syntax
    code = "+ import httpx\n+ from fastapi import"
    result = analyze_imports(code)
    assert isinstance(result, list)  # must not raise
```

### Integration Test with Planted Bugs

```python
# tests/test_integration.py
import pytest
import asyncio
import time
from main import REVIEW_GRAPH, AgentState, Severity

SAMPLE_DIFF = """\
--- a/user_service.py
+++ b/user_service.py
@@ -1,5 +1,28 @@
+import pickle
+import os
+
+SECRET_KEY = "hardcoded-do-not-ship-abc123"
+
+async def get_user(user_id: str):
+    # N+1: called in a loop upstream, no batching
+    query = f"SELECT * FROM users WHERE id = '{user_id}'"
+    result = await db.execute(query)
+    return result
+
+def load_session(data: bytes):
+    # insecure deserialization — pickle.loads on user-supplied bytes
+    return pickle.loads(data)
+
+def run_cmd(cmd):
+    # command injection — user-controlled cmd passed to os.system
+    os.system(f"echo {cmd}")
+
+def fetch_users():
+    # missing .limit() — could return millions of rows
+    return db.query(User).all()
"""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graph_catches_planted_bugs(monkeypatch):
    async def mock_diff(repo, pr_number):  return SAMPLE_DIFF
    async def mock_meta(repo, pr_number):  return {"commit_sha": "abc123", "title": "Test", "author": "dev"}
    async def mock_files(repo, pr_number): return {"user_service.py": SAMPLE_DIFF.replace("+", "")}

    monkeypatch.setattr("main.fetch_pr_diff",     mock_diff)
    monkeypatch.setattr("main.fetch_pr_metadata", mock_meta)
    monkeypatch.setattr("main.get_changed_files", mock_files)

    state: AgentState = {
        "pr_url": "https://github.com/test/repo/pull/1",
        "repo": "", "pr_number": 0, "commit_sha": "",
        "pr_diff": "", "code_files": {}, "security_findings": [],
        "performance_findings": [], "style_findings": [], "all_findings": [],
        "final_report": None, "agents_run": [], "agents_failed": [],
        "total_input_tokens": 0, "total_output_tokens": 0,
        "start_time": time.monotonic(), "errors": [],
    }

    result = await REVIEW_GRAPH.ainvoke(state)
    report = result["final_report"]

    assert report is not None
    assert report.total_findings >= 1

    critical_high = (
        report.findings_by_severity.get(Severity.CRITICAL, []) +
        report.findings_by_severity.get(Severity.HIGH, [])
    )
    assert len(critical_high) >= 1, "Expected at least one CRITICAL or HIGH finding"
    assert report.metadata.estimated_cost_usd < 0.50
    assert "security" in report.metadata.agents_run


def evaluate_recall(report, expected_keywords: list[str]) -> dict:
    """
    Measure what fraction of planted bugs the review caught.
    Use to track regression: if recall drops below 0.8, investigate.
    """
    finding_text = " ".join(
        f.title + " " + f.description for f in report.all_findings
    ).lower()
    caught = sum(1 for kw in expected_keywords if kw.lower() in finding_text)
    return {
        "expected": len(expected_keywords),
        "caught":   caught,
        "recall":   caught / len(expected_keywords) if expected_keywords else 0.0,
    }
```

### Running Tests

```bash
# Unit tests only — no LLM calls, runs in seconds
pip install pytest pytest-asyncio
pytest tests/test_tools.py -v

# Integration tests — require env vars, costs ~$0.05 per run
ANTHROPIC_API_KEY=... GITHUB_TOKEN=... pytest tests/test_integration.py -v -m integration

# Coverage report
pytest tests/test_tools.py --cov=main --cov-report=html

# Run everything except integration (fast CI check)
pytest tests/ -v -m "not integration"
```

---

## 9. Production Hardening

### Rate Limiting — Redis Sliding Window

```python
import redis.asyncio as aioredis
from fastapi import Request

redis_client = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))

RATE_LIMIT_WINDOW = 3600   # 1 hour in seconds
RATE_LIMIT_MAX    = 10     # reviews per user per hour


async def check_rate_limit(user_id: str) -> bool:
    """Returns True if allowed, False if rate-limited."""
    key = f"rl:{user_id}"
    count = await redis_client.incr(key)
    if count == 1:
        # Set expiry only on first increment to avoid resetting the window
        await redis_client.expire(key, RATE_LIMIT_WINDOW)
    return count <= RATE_LIMIT_MAX


# Updated /review endpoint with rate limiting
@app.post("/review")
async def create_review(
    req: ReviewRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    user_id = request.headers.get("X-User-Id", request.client.host)
    if not await check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_MAX} reviews/hour",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "pr_url": req.pr_url, "result": None, "error": None}
    background_tasks.add_task(_run_review_job, job_id, req.pr_url)
    return {"job_id": job_id, "status": "queued"}
```

### Cache by Commit SHA

```python
CACHE_TTL = 86400  # 24 hours


async def get_cached_review(repo: str, pr_number: int, commit_sha: str) -> Optional[ReviewReport]:
    """Return a cached review if one exists for this exact commit, else None."""
    key = f"review:{repo}:{pr_number}:{commit_sha}"
    raw = await redis_client.get(key)
    if raw:
        data = json.loads(raw)
        data["metadata"]["cache_hit"] = True
        return ReviewReport(**data)
    return None


async def cache_review(repo: str, pr_number: int, commit_sha: str, report: ReviewReport) -> None:
    """Store a completed review in Redis with a 24-hour TTL."""
    key = f"review:{repo}:{pr_number}:{commit_sha}"
    await redis_client.setex(key, CACHE_TTL, report.model_dump_json())


# In _run_review_job, insert this check after fetch_node sets commit_sha:
# cached = await get_cached_review(repo, pr_num, sha)
# if cached:
#     jobs[job_id].update({"status": "complete", "result": cached.model_dump(mode="json")})
#     return
```

### Retry with Exponential Backoff

```python
from functools import wraps


def with_retry(max_attempts: int = 2, base_delay: float = 1.0):
    """
    Decorator that retries the wrapped async function on transient errors.
    After max_attempts, re-raises the last exception.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (anthropic.APIError, httpx.HTTPError, asyncio.TimeoutError) as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(base_delay * (2 ** attempt))
            raise last_exc
        return wrapper
    return decorator


# Apply to any specialist node:
@with_retry(max_attempts=2)
async def security_node_resilient(state: AgentState) -> dict:
    return await security_node(state)
```

### Graceful Degradation Summary

| Failure Point | Behavior |
|---------------|----------|
| GitHub API timeout | `fetch_node` raises — whole job fails with "fetch error" |
| One specialist LLM call fails | Node returns `[]` findings, appends to `agents_failed` |
| Critic LLM call fails | Falls back to raw concatenation of all findings, unsorted |
| Formatter LLM call fails | Generates basic summary from finding counts only |
| Full graph timeout (60s) | Job marked failed, SSE emits `review_failed` |
| Cost exceeds $0.50 | Result discarded after completion, error returned to caller |

The design ensures partial results are always better than no results. Even if two of three specialists fail, the remaining specialist's findings still reach the Critic and produce a report.

---

## 10. Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Non-root user for container security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### requirements.txt

```
anthropic>=0.30.0
langgraph>=0.2.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
pydantic>=2.7.0
redis[asyncio]>=5.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...       # Anthropic API key (auto-read by SDK)
GITHUB_TOKEN=ghp_...               # GitHub PAT with repo:read scope

# Optional
REDIS_URL=redis://localhost:6379   # Enables caching and rate limiting
MAX_COST_USD=0.50                  # Per-review cost ceiling in USD
MAX_FILES=10                       # Maximum files to analyze per PR
LOG_LEVEL=INFO
```

### Docker Compose

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - REDIS_URL=redis://redis:6379
      - MAX_COST_USD=0.50
      - MAX_FILES=10
    depends_on: [redis]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --save 60 1 --loglevel warning
    restart: unless-stopped
```

### Kubernetes Notes

You already know K8s — the specifics for this workload:

**Resources.** All work is I/O-bound (asyncio, LLM API calls, GitHub API). Low CPU, moderate memory.

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

**Horizontal scaling.** The FastAPI app is stateless once `jobs` moves to Redis. With Redis-backed job store, an HPA on HTTP request rate scales immediately. Replace the in-memory `jobs` dict with Redis hashes: `HSET review:{job_id} status running result "" error ""`.

**Secrets.** Mount `ANTHROPIC_API_KEY` and `GITHUB_TOKEN` from a Kubernetes Secret. Never put them in ConfigMap or Deployment manifest YAML.

**Liveness vs readiness.** Use `/health` for both. The liveness probe restarts crashed pods. The readiness probe gates traffic during startup before the LangGraph compilation finishes.

---

## 11. Skills Mastered — Complete Curriculum Inventory

### LLM Fundamentals (8 Skills)

1. **Tokenization** — BPE encoding, ~4 chars/token for English, why token count diverges from word count, cost implications of verbosity
2. **Attention and context windows** — why 200K token windows don't mean you use all of it effectively; quality degrades in the middle
3. **Temperature and sampling** — 0.0 for structured output and determinism, 0.7-1.0 for creative generation, top_p and top_k
4. **Inference modes** — streaming for UX responsiveness, blocking for simplicity, latency vs throughput tradeoffs
5. **Model selection** — Haiku for fast classification and extraction, Sonnet for general reasoning, Opus for complex multi-step tasks
6. **Context budget management** — prioritizing what enters context when near the limit: recency bias, task relevance, compression
7. **Prompt engineering** — system vs user prompts, XML tags for structure, chain-of-thought elicitation, few-shot examples
8. **Structured output** — JSON mode, tool forcing, Pydantic validation of responses, stripping markdown fences from LLM output

### API and Tool Use (8 Skills)

1. **Streaming** — `stream=True`, `text_delta` events, forwarding to SSE endpoints, handling partial JSON in stream buffers
2. **Tool use** — JSON Schema tool definitions, parsing `tool_use` content blocks, multi-turn tool call loops
3. **Parallel tool calls** — requesting multiple tools in one response, merging results before the next turn
4. **Prompt caching** — `cache_control: {type: "ephemeral"}` on system prompts, 5-min TTL, 90% cost reduction on hits
5. **Token counting** — `client.messages.count_tokens()` for pre-flight cost estimation and abort guards
6. **Batch API** — 50% cost savings for non-latency-sensitive workloads, async job processing pattern
7. **Error handling** — `anthropic.RateLimitError`, `anthropic.APIError`, exponential backoff, circuit breakers
8. **Multi-modal inputs** — image content blocks (base64 or URL), when to use vision vs text extraction

### Memory and RAG (8 Skills)

1. **In-context memory** — message history compression, sliding window strategy, summarizing old turns into a single context message
2. **Vector databases** — pgvector, Chroma, Pinecone — trade-offs of hosted vs self-hosted
3. **Embedding models** — `text-embedding-3-small` vs `text-embedding-3-large`, speed vs quality
4. **Chunking strategies** — fixed-size with overlap, sentence-boundary, semantic chunking — when each is appropriate
5. **Retrieval** — cosine similarity search, HNSW vs IVFFlat index types, top-k selection, MMR for diversity
6. **Hybrid search** — combining vector similarity with BM25 keyword matching for better recall on exact terms
7. **Reranking** — cross-encoder models to reorder retrieved chunks by true relevance before the LLM call
8. **RAG evaluation** — RAGAS metrics: faithfulness, answer relevance, context precision, context recall

### LangGraph (7 Skills)

1. **StateGraph with TypedDict** — the core abstraction: graph as a typed state machine, nodes as pure functions
2. **Reducers** — `Annotated[list, operator.add]` for accumulating results from parallel branches without overwrites
3. **Node contracts** — `(state: S) -> dict` — read from state, return only changed fields, never mutate state in-place
4. **Deterministic and conditional edges** — `add_edge` for sequential flow, `add_conditional_edges` for routing logic
5. **Parallel fan-out with Send** — `Send(node_name, state)` dispatches to multiple nodes simultaneously; fan-in is automatic
6. **HITL and checkpointing** — `interrupt_before=["node_name"]`, `Command(resume=value)`, `MemorySaver` for dev, `PostgresSaver` for prod
7. **Graph streaming** — `graph.astream_events()` for real-time node events, filtering by `kind == "on_chain_start"`

### Multi-Agent Patterns (6 Skills)

1. **Orchestrator/worker pattern** — orchestrator owns state, workers are focused and stateless, no shared mutable objects
2. **Typed handoffs** — Pydantic models as the contract between agents; every inter-agent boundary is validated before use
3. **Parallel specialists** — `Send()` fan-out for independent work streams, fan-in via state reducers
4. **Critic/judge pattern** — a dedicated agent reviews outputs from other agents for quality, deduplication, and severity calibration
5. **Failure isolation** — one specialist failing does not block others; state accumulates partial results gracefully
6. **Agent trust model** — treat all agent outputs as untrusted user input; validate through Pydantic before downstream use

### Production Engineering (9 Skills)

1. **Cost control** — count tokens before calling, estimate cost per call, enforce per-request budget, cache aggressively by (repo, PR, SHA)
2. **Observability** — structured logging with trace IDs, Langfuse/LangSmith for LLM call tracing, cost in every response object
3. **Evaluation** — golden datasets, recall/precision on planted bugs, regression testing when prompts change
4. **Rate limiting** — Redis sliding window, per-user limits, 429 with `Retry-After` header, token bucket as alternative
5. **Caching** — key by input hash or commit SHA, TTL management, cache-hit flag in metadata, explicit invalidation strategy
6. **Security** — validate all external inputs, treat agent outputs as untrusted, secrets in env vars never in code
7. **Reliability** — per-node timeouts (`asyncio.timeout`), retry with exponential backoff, graceful degradation by layer
8. **Deployment** — Docker with non-root users, health endpoints, readiness vs liveness probes, Kubernetes HPA
9. **Streaming APIs** — SSE for server-push, `asyncio.Queue` as the internal bus between background tasks and HTTP responses

---

## 12. What's Next

### Fine-Tuning and LoRA

Fine-tuning is worth it when you have 1,000+ curated examples, need consistent output format that prompting cannot reliably produce, or need to reduce costs by eliminating lengthy system prompts. It is not worth it when your problem changes frequently, you lack clean labeled data, or RAG can solve the grounding problem more cheaply. LoRA (Low-Rank Adaptation) trains small adapter layers on top of a frozen base model — 10-100x less compute than full fine-tuning. Hugging Face PEFT makes this accessible. For the code review domain: fine-tune only if your evals show the base model consistently missing a class of issues that better prompting cannot fix after genuine effort.

### Multimodal Agents

Claude's vision API accepts `image` content blocks (base64 or URL). The pattern is identical to text agents — image is just another input type in the messages array. Practical extensions: accept architecture diagrams alongside PRs, review UI screenshots for accessibility issues, parse ERD diagrams to validate that schema migrations match the diagram. The engineering challenge is context budget — a single high-resolution image consumes 1,000-4,000 tokens depending on detail level.

### Agent Evals at Scale

Moving from ad-hoc testing to a systematic eval pipeline. **Braintrust** — dataset management, experiment tracking, A/B comparison of prompt variants, designed for teams running 100+ prompt experiments. **LangSmith** — automatic tracing for LangGraph (zero-config), built-in eval runners, natural choice if you stay in the LangGraph ecosystem. **Ragas** — open source, specialized for RAG evaluation, runs locally without accounts. The most important principle: your evals are only as good as your golden dataset. Spend more effort on test case quality — diversity, edge cases, adversarial inputs — than on eval framework selection.

### Constitutional AI

Constitutional AI is Anthropic's approach to alignment: a model critiques its own outputs against a set of principles ("the constitution") and revises them before returning. The practical engineering application: implement a lightweight self-correction step in your own pipelines. After the Formatter produces a summary, a second LLM call checks it against a rubric ("Did we cite the actual line? Is the recommendation concrete? Is the severity calibrated?") and revises before returning. This is the Critic agent pattern generalized to any output quality concern.

### Mixture of Agents and Model Routing

Route different tasks to the cheapest model that can handle them well. Pattern: a lightweight classifier (Haiku) reads the incoming request and routes it — simple extraction goes to Haiku, multi-step reasoning goes to Sonnet, ambiguous cases escalate to Opus. For the code review agent: route style checks to Haiku (cheap, fast, mechanical), security analysis to Sonnet (needs reasoning), final synthesis to Sonnet. This can reduce costs 40-60% with minimal quality loss if your routing classifier is accurate. Track routing decisions in your observability layer to audit when routing gets it wrong.

### Self-Hosted LLM Stack (Ollama, vLLM)

For use cases where data cannot leave your infrastructure — regulated industries, proprietary codebases — running open-weight models locally becomes necessary. **Ollama** — easiest path: `ollama run llama3.3` gives you a compatible OpenAI API locally. Works on Mac with Apple Silicon, good for development. **vLLM** — production-grade inference server with PagedAttention for efficient GPU memory use. Supports OpenAI-compatible API so your existing code works with a URL change. Throughput 10-30x better than naive Hugging Face inference. Current open-weight models competitive with Claude Sonnet-class performance on code analysis: Llama 3.3 70B, Qwen2.5-Coder 72B, DeepSeek-R1 for reasoning tasks.

---

### Resources

**Papers (in reading order):**
1. "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022) — the foundation of tool-using agents
2. "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
3. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023)
4. "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022) — how RLHF works at Anthropic
5. "Language Agents as Optimizable Graphs" (Zhuge et al., 2024) — multi-agent theory

**Courses:**
- DeepLearning.AI "Multi AI Agent Systems with crewAI" — practical, ~4 hours
- Andrej Karpathy "Let's build GPT from scratch" on YouTube — best mental model for what happens inside the model
- Fast.ai Practical Deep Learning — if you want to understand the ML layer underneath the APIs

**Communities:**
- Anthropic Discord — direct access to people building with the Claude API
- LangChain Discord — largest agent framework community, active help channel
- r/LocalLLaMA — open-weight models and local inference discussion
- Papers With Code (paperswithcode.com) — track state-of-the-art on agent benchmarks in real time

**What to build next:** Add one capability per week to this code review agent — GitHub Actions integration (auto-run on every PR), a web dashboard with per-PR history, TypeScript and Go support via additional specialist agents, or a feedback loop where human-accepted suggestions are stored and used to improve future reviews via few-shot examples. The pattern scales to any domain: swap Security, Performance, and Style for any three specialist perspectives appropriate to your work. Shipping real systems with real users beats studying more frameworks.

---

*Day 7 complete. You have designed, implemented, tested, and hardened a production-grade multi-agent system from scratch. The patterns here — stateless specialists, parallel fan-out with typed reducers, structured output with Pydantic validation, SSE streaming, per-layer graceful degradation, and cost controls — are the patterns that appear in every serious agentic application in production. They transfer directly to the next system you build.*
