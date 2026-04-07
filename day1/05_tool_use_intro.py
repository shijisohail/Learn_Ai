"""
Day 1 - Exercise 5: Tool use (function calling) — MOST IMPORTANT CONCEPT
Goal: understand how the model requests tools and how you execute them.

This is the foundation of EVERY agentic system.

Flow:
  1. You define tools (like OpenAPI operation schemas)
  2. Model decides to call a tool → returns tool_use block (NOT text)
  3. YOU execute the tool (the model cannot — it only requests)
  4. You send the result back as tool_result
  5. Model continues generating
"""

import anthropic

client = anthropic.Anthropic()

# Step 1: Define tools — these are JSON schemas, like OpenAPI
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Use this when the user asks about weather.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London' or 'Karachi'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation. Use for any math operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(100 * 1.18) / 12'"
                }
            },
            "required": ["expression"]
        }
    }
]


# Step 2: Implement the actual tool functions (the model REQUESTS, you RUN)
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Fake weather API — in real life, call OpenWeatherMap or similar."""
    fake_data = {
        "karachi": {"temp": 32, "condition": "Sunny", "humidity": 65},
        "london": {"temp": 14, "condition": "Cloudy", "humidity": 80},
        "rawalpindi": {"temp": 28, "condition": "Clear", "humidity": 55},
    }
    data = fake_data.get(city.lower(), {"temp": 20, "condition": "Unknown", "humidity": 50})
    return {"city": city, "temperature": data["temp"], "unit": unit, "condition": data["condition"]}


def calculate(expression: str) -> dict:
    """Safe math evaluator."""
    try:
        # In production: use a proper math parser, not eval
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def run_tool(tool_name: str, tool_input: dict) -> str:
    """Tool dispatcher — like a router in your FastAPI app."""
    import json
    if tool_name == "get_weather":
        result = get_weather(**tool_input)
    elif tool_name == "calculate":
        result = calculate(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)


def agent_turn(messages: list[dict], user_input: str) -> tuple[str, list[dict]]:
    """
    One full agent turn: may involve multiple tool calls before final response.
    This is the CORE of every agent.
    """
    messages = messages + [{"role": "user", "content": user_input}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        print(f"  [stop_reason: {response.stop_reason}]")

        # Case 1: Model is done — return text response
        if response.stop_reason == "end_turn":
            reply = response.content[0].text
            messages = messages + [{"role": "assistant", "content": response.content}]
            return reply, messages

        # Case 2: Model wants to call tools
        if response.stop_reason == "tool_use":
            # Add assistant's tool request to history
            messages = messages + [{"role": "assistant", "content": response.content}]

            # Execute each requested tool
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool call: {block.name}({block.input})]")
                    result = run_tool(block.name, block.input)
                    print(f"  [tool result: {result}]")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Send tool results back — model continues from here
            messages = messages + [{"role": "user", "content": tool_results}]
            # Loop: model may call more tools or give final answer


if __name__ == "__main__":
    history = []

    queries = [
        "What's the weather like in Rawalpindi?",
        "If I earn 150000 PKR/month, what's my daily rate? (assume 22 working days)",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        reply, history = agent_turn(history, query)
        print(f"Assistant: {reply}")

    print(f"\nTotal messages in history: {len(history)}")
