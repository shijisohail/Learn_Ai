"""
AI Coach local server — OpenRouter backend with auto-fallback
Run: python server.py  (reads OPENROUTER_API_KEY from .env)
Requires: pip install fastapi uvicorn httpx python-dotenv
"""
import os, json
import httpx
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tried in order — first available wins
FALLBACK_MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e4b-it:free",
    "arcee-ai/trinity-mini:free",
]

app = FastAPI(title="AI Coach Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list
    system: str = ""
    model: str = FALLBACK_MODELS[0]
    max_tokens: int = 1800

def build_messages(req: ChatRequest):
    def to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content)

    messages = []
    for i, msg in enumerate(req.messages):
        content = to_text(msg["content"])
        # Gemma doesn't support system role — prepend to first user message
        if i == 0 and msg["role"] == "user" and req.system:
            content = f"{req.system}\n\n{content}"
        messages.append({"role": msg["role"], "content": content})
    return messages

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": bool(OPENROUTER_API_KEY)}

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = build_messages(req)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build fallback list: requested model first, then others
    models_to_try = [req.model] + [m for m in FALLBACK_MODELS if m != req.model]

    async def generate():
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": req.max_tokens,
                "stream": True,
            }
            got_content = False
            error_msg = None
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", OPENROUTER_URL,
                                             json=payload, headers=headers) as resp:
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                                # Check for provider error in stream
                                if "error" in obj:
                                    error_msg = obj["error"].get("message", "Provider error")
                                    break
                                text = obj["choices"][0]["delta"].get("content", "")
                                if text:
                                    got_content = True
                                    yield f"data: {json.dumps({'text': text})}\n\n"
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
            except Exception as e:
                error_msg = str(e)

            if got_content:
                break  # success — stop trying fallbacks
            # else: rate-limited or error, try next model silently
            print(f"Model {model} failed ({error_msg}), trying next...")

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
