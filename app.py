import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from smart_uae_agent import build_agent

SMARTUAE_LLM = os.getenv("SMARTUAE_LLM", "openai")
SMARTUAE_KB = os.getenv("SMARTUAE_KB", "uae_knowledge.json")

app = FastAPI(title="SmartUAE Tourism Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Build once; reuse
agent = build_agent(SMARTUAE_KB, SMARTUAE_LLM)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    latency_ms: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    import time
    t0 = time.time()
    resp = agent.invoke({"input": req.message})
    out = resp.get("output") or ""
    return ChatResponse(reply=out, latency_ms=int((time.time() - t0) * 1000))
