"""
Day 1 - Exercise 3: Streaming
Goal: stream tokens as they're generated — essential for any real app.
This is like consuming a chunked HTTP response or a Kafka stream.
"""

import anthropic

client = anthropic.Anthropic()

print("Streaming response:\n")

# Streaming: tokens arrive as they're generated, not after full completion
with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print("\n\n=== Final message stats ===")
final = stream.get_final_message()
print(f"Total input tokens : {final.usage.input_tokens}")
print(f"Total output tokens: {final.usage.output_tokens}")

# For production: use streaming for all user-facing responses.
# Non-streaming is fine for internal agent steps (tool calls, planning).
