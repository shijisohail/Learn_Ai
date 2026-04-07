"""
Day 1 - Activity 2: System Prompt Engineering
===============================================
CONCEPT: The system prompt is your middleware. It sets the model's persona,
         constraints, output format, and rules — before any user message.
         Think of it as the app.use() layer in Express or middleware in FastAPI.

TASK: You have 3 broken system prompts. Fix each one so the test passes.
      Run the test() function to verify. All 3 must pass.

Rules:
  - Do NOT change the test() function
  - Only edit the SYSTEM_PROMPT_* strings
  - Each system prompt must be under 200 words
"""

import anthropic
import json

client = anthropic.Anthropic()


def llm(system: str, user: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return r.content[0].text.strip()


# ─────────────────────────────────────────────────────────
# PROMPT 1: Make the model respond ONLY in valid JSON
# The response must be parseable by json.loads() — no extra text, no markdown
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_1 = """
TODO: Write a system prompt that forces the model to always respond
with raw valid JSON only. No explanation, no markdown code blocks.
"""


# ─────────────────────────────────────────────────────────
# PROMPT 2: Make the model act as a strict code reviewer
# It must: always find at least one issue, respond in bullet points,
# never compliment the code, always suggest a fix
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_2 = """
TODO: Write a system prompt for a strict senior code reviewer.
"""


# ─────────────────────────────────────────────────────────
# PROMPT 3: Make the model be extremely concise
# Every response must be 10 words or fewer. No exceptions.
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_3 = """
TODO: Write a system prompt that enforces max 10 words per response.
"""


def test():
    passed = 0

    # Test 1: JSON only
    try:
        response = llm(SYSTEM_PROMPT_1, "Tell me about Python in 3 key points")
        json.loads(response)
        print("✓ Test 1 passed: Response is valid JSON")
        passed += 1
    except json.JSONDecodeError:
        print(f"✗ Test 1 failed: Not valid JSON\n  Got: {response[:100]}")

    # Test 2: Code review
    code = """
def get_user(id):
    result = db.execute("SELECT * FROM users WHERE id = " + id)
    return result
"""
    response = llm(SYSTEM_PROMPT_2, f"Review this code:\n{code}")
    has_bullet = any(line.strip().startswith(("-", "•", "*")) for line in response.split("\n"))
    if has_bullet and len(response) > 50:
        print("✓ Test 2 passed: Got bullet-point code review")
        passed += 1
    else:
        print(f"✗ Test 2 failed: Expected bullet points with substance\n  Got: {response[:150]}")

    # Test 3: Max 10 words
    response = llm(SYSTEM_PROMPT_3, "Explain quantum computing in detail please")
    word_count = len(response.split())
    if word_count <= 10:
        print(f"✓ Test 3 passed: Response is {word_count} words")
        passed += 1
    else:
        print(f"✗ Test 3 failed: {word_count} words (max 10)\n  Got: {response}")

    print(f"\n{passed}/3 tests passed")


if __name__ == "__main__":
    test()
