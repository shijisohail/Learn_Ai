"""
Day 1 - Activity 4: Multi-Tool Agent (BOSS LEVEL)
===================================================
CONCEPT: Real agents use multiple tools in a single turn.
         The model decides WHICH tools to call and in what order.
         You don't hardcode the flow — the LLM plans it.

TASK: You are given 4 tool implementations. Your job is to:
  1. Write the tool schemas (the JSON definitions the model sees)
  2. Write the agent loop (the while loop from exercise 5)
  3. Wire the dispatcher

All 4 queries in the test must produce correct answers.

Tools provided:
  - word_count(text)       → counts words in text
  - reverse_string(text)   → reverses a string
  - is_palindrome(text)    → checks if text is a palindrome
  - repeat_text(text, n)   → repeats text n times
"""

import json
import anthropic

client = anthropic.Anthropic()


# ── Tool implementations (DO NOT CHANGE) ─────────────────────────────────────

def word_count(text: str) -> dict:
    words = text.split()
    return {"text": text, "word_count": len(words)}

def reverse_string(text: str) -> dict:
    return {"original": text, "reversed": text[::-1]}

def is_palindrome(text: str) -> dict:
    cleaned = text.lower().replace(" ", "")
    result = cleaned == cleaned[::-1]
    return {"text": text, "is_palindrome": result}

def repeat_text(text: str, n: int) -> dict:
    return {"result": (text + " ") * n, "times": n}


# ── TODO 1: Define tool schemas ───────────────────────────────────────────────
# Follow the same structure as exercise 5.
# Each tool needs: name, description, input_schema (with properties + required)

TOOLS = [
    # TODO: word_count schema

    # TODO: reverse_string schema

    # TODO: is_palindrome schema

    # TODO: repeat_text schema — note: n is an integer, not a string
]


# ── TODO 2: Tool dispatcher ───────────────────────────────────────────────────

def run_tool(tool_name: str, tool_input: dict) -> str:
    """
    TODO: Route tool_name to the correct function above.
    Return json.dumps() of the result.
    """
    pass


# ── TODO 3: Agent loop ────────────────────────────────────────────────────────

def agent(user_input: str) -> str:
    """
    TODO: Implement the agent loop from exercise 5.
    - Single-turn (no conversation history needed)
    - Keep looping while stop_reason == "tool_use"
    - Return the final text when stop_reason == "end_turn"
    """
    pass


# ── Tests — all must pass ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "query": "How many words are in: 'The quick brown fox jumps over the lazy dog'?",
            "check": "9",
        },
        {
            "query": "Reverse the string 'Sharjeel'",
            "check": "leejrahS",
        },
        {
            "query": "Is 'racecar' a palindrome?",
            "check": "yes",   # answer should contain "yes" (case-insensitive)
        },
        {
            "query": "Repeat the word 'AI' 4 times",
            "check": "AI",    # answer should contain AI repeated
        },
    ]

    passed = 0
    for i, tc in enumerate(test_cases, 1):
        print(f"\nTest {i}: {tc['query']}")
        try:
            result = agent(tc["query"])
            print(f"  Answer: {result}")
            if tc["check"].lower() in result.lower():
                print(f"  ✓ PASSED")
                passed += 1
            else:
                print(f"  ✗ FAILED — expected to find '{tc['check']}' in answer")
        except Exception as e:
            print(f"  ✗ ERROR — {e}")

    print(f"\n{'='*40}")
    print(f"Result: {passed}/{len(test_cases)} tests passed")
