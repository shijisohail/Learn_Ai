"""
Day 1 - Exercise 2: Conversation memory
Goal: understand that YOU manage the conversation history, not the API.
This is the most important concept for building agents.
"""

import anthropic

client = anthropic.Anthropic()

def chat(messages: list[dict], user_input: str) -> tuple[str, list[dict]]:
    """
    Send a message and return (reply, updated_messages).
    The caller owns the messages list — this is stateless by design.

    Think of messages like a request body you append to on every call.
    """
    messages = messages + [{"role": "user", "content": user_input}]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="You are a helpful assistant. Be concise.",
        messages=messages,
    )

    reply = response.content[0].text
    messages = messages + [{"role": "assistant", "content": reply}]

    return reply, messages


if __name__ == "__main__":
    history = []

    # Turn 1
    reply, history = chat(history, "My name is Sharjeel.")
    print(f"Assistant: {reply}")
    print(f"History length: {len(history)} messages\n")

    # Turn 2 — model remembers because WE sent the history
    reply, history = chat(history, "What's my name?")
    print(f"Assistant: {reply}")
    print(f"History length: {len(history)} messages\n")

    # Turn 3 — prove it: start fresh history — model forgets
    reply, _ = chat([], "What's my name?")
    print(f"Assistant (fresh context): {reply}")
    print("\nKey insight: memory = what's in the messages array. Nothing more.")
