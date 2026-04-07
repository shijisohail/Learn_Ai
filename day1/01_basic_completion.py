"""
Day 1 - Exercise 1: Basic completion
Goal: understand the raw API, tokens, and the stateless nature of LLMs
"""

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# Most basic call possible
response = client.messages.create(
    model="claude-haiku-4-5-20251001",  # cheapest model, good for learning
    max_tokens=256,
    messages=[
        {"role": "user", "content": "What is 2 + 2? Answer in one word."}
    ]
)

print("=== Response object ===")
print(f"Stop reason : {response.stop_reason}")   # 'end_turn' or 'max_tokens' or 'tool_use'
print(f"Input tokens : {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Content      : {response.content[0].text}")

# Key insight: the model is stateless — it has NO memory of this call.
# If you call it again, it doesn't know this happened.
# "Memory" = what you put in the messages array.
