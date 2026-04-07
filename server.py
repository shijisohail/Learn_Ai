"""
AI Coach local server
Run: ANTHROPIC_API_KEY=sk-ant-... python server.py
Requires: pip install fastapi uvicorn anthropic
"""
import os, json, asyncio
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="AI Coach Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None

@app.on_event("startup")
async def startup():
    global client
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("⚠️  ANTHROPIC_API_KEY not set. Set it before starting.")
    else:
        client = anthropic.Anthropic(api_key=key)
        print("✅ Anthropic client ready")

class ChatRequest(BaseModel):
    messages: list
    system: str = ""
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1800

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": client is not None}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not client:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    def generate():
        try:
            with client.messages.stream(
                model=req.model,
                max_tokens=req.max_tokens,
                system=req.system,
                messages=req.messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
