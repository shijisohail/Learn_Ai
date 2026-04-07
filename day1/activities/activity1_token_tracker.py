"""
Day 1 - Activity 1: Token Usage Tracker
========================================
CONCEPT: Tokens are your cost unit. Every API call has input + output tokens.
         As an agent builder you MUST track these — runaway agents can burn $$$

TASK: Build a chat function that tracks cumulative token usage across turns.
      Complete all the TODOs below.

Expected output:
  Turn 1 | in: 42  out: 18  | session total: 60  tokens | $0.000045
  Turn 2 | in: 67  out: 31  | session total: 158 tokens | $0.000119
  Turn 3 | in: 95  out: 22  | session total: 275 tokens | $0.000206
"""

import anthropic

client = anthropic.Anthropic()

# Haiku pricing (as of 2025): $0.00025 per 1k input, $0.00125 per 1k output
INPUT_PRICE_PER_1K  = 0.00025
OUTPUT_PRICE_PER_1K = 0.00125


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """TODO: Calculate the dollar cost for a given token usage."""
    # Hint: (tokens / 1000) * price_per_1k
    pass


class TokenTrackedChat:
    def __init__(self):
        self.messages: list[dict] = []
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.turn = 0

    def chat(self, user_input: str) -> str:
        """
        TODO:
        1. Append user message to self.messages
        2. Call the API (model: claude-haiku-4-5-20251001, max_tokens: 256)
        3. Extract reply text from response
        4. Append assistant message to self.messages
        5. Update self.total_input_tokens and self.total_output_tokens
        6. Increment self.turn
        7. Print the turn summary line (see expected output format above)
        8. Return the reply text
        """
        pass

    @property
    def total_tokens(self) -> int:
        """TODO: Return total tokens used so far (input + output)."""
        pass

    @property
    def total_cost(self) -> float:
        """TODO: Return total cost so far in dollars."""
        pass


if __name__ == "__main__":
    chat = TokenTrackedChat()

    questions = [
        "What is a neural network in one sentence?",
        "What is a transformer in one sentence?",
        "What is an LLM in one sentence?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = chat.chat(q)
        print(f"A: {answer}\n")

    print("=" * 50)
    print(f"Session summary: {chat.total_tokens} tokens, ${chat.total_cost:.6f}")
