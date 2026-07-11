
"""
main.py — FastAPI application

GET  /health  — readiness probe, returns {"status": "ok"}
POST /chat    — stateless conversation endpoint
GET  /        — chat UI (served from app/static/index.html)

The FAISS catalog index is built once at startup via FastAPI's lifespan event.
Cold starts on Render's free tier take ~2–3 min while embeddings are rebuilt.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models — these enforce the API contract at the boundary
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {v!r}")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# App lifecycle — index built once at startup
# ---------------------------------------------------------------------------

# We store the index in app.state so it's accessible to route handlers
# without using a global variable (cleaner, easier to test).
from app.retrieval import CatalogIndex
from app import agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    catalog_path = os.environ.get("CATALOG_PATH", "data/catalog.json")
    if not Path(catalog_path).exists():
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. "
            "Run: python scripts/build_catalog.py"
        )
    print(f"Building catalog index from {catalog_path}...")
    app.state.index = CatalogIndex(catalog_path)
    print(f"Index ready: {len(app.state.index.items)} items loaded.")
    yield
    # === SHUTDOWN ===
    # Nothing to clean up — FAISS index is in-memory only.


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for selecting SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_static_dir / "index.html")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Readiness check. Returns 200 only after the catalog index is loaded.
    The spec allows up to 2 minutes for the first /health call on cold start.
    """
    # If lifespan hasn't completed, app.state.index won't exist yet.
    # hasattr check ensures we return 503 rather than 500 during startup.
    if not hasattr(app.state, "index"):
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}




@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Stateless chat: caller sends full history on every request."""
    if not hasattr(app.state, "index"):
        raise HTTPException(status_code=503, detail="Service is still starting up")

    # Convert Pydantic models back to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if len(messages) > 8:
        messages = messages[-8:]

    try:
        result = agent.chat(messages, app.state.index)
    except Exception as e:
        # Return a safe fallback rather than a 500.
        print(f"ERROR in agent.chat: {e}")
        return ChatResponse(
            reply="I encountered an unexpected issue. Could you repeat your last message?",
            recommendations=[],
            end_of_conversation=False,
        )

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )