"""
CerebroForge (铸脑) - FastAPI Application
============================================
Complete REST API for the cognitive agent framework.

Provides endpoints for:
- Chat (sync and SSE streaming)
- Clarification handling
- Tool/skill forging
- Memory management
- Evolution control
- Configuration
- File upload
- Computer use
"""

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import uvicorn

# ────────────────────────────────────────────────────────────────────────────
# Setup paths for imports
# ────────────────────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# Add both project root (for backend.xxx imports) and backend dir (for xxx imports)
for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Now import config (will work with either import style)
try:
    from backend.config import NVIDIA_BASE_URL, NVIDIA_API_KEY, DEFAULT_MODEL
except ImportError:
    from config import NVIDIA_BASE_URL, NVIDIA_API_KEY, DEFAULT_MODEL

try:
    from backend.schemas import ToolStatus, MemoryLevel
except ImportError:
    from schemas import ToolStatus, MemoryLevel

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Request Models
# ────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=10000, description="User query text")
    stream: bool = Field(default=False, description="Enable streaming response")
    mode: Optional[str] = Field(default=None, description="Override system mode (1, 2)")


class ClarifyRequest(BaseModel):
    """Request model for clarification handling."""
    original_query: str = Field(..., min_length=1, description="Original user query")
    answers: List[str] = Field(..., min_length=1, description="User's answers to clarification questions")
    clarification_id: Optional[str] = Field(default=None, description="Clarification session ID")


class ToolForgeRequest(BaseModel):
    """Request model for forging new tools."""
    requirement: str = Field(..., min_length=10, max_length=5000, description="Description of what the tool should do")
    context: Optional[str] = Field(default=None, description="Additional context")
    is_public: bool = Field(default=False, description="Make tool publicly available")


class ConfigUpdate(BaseModel):
    """Request model for configuration updates."""
    base_url: Optional[str] = Field(default=None, description="LLM API base URL")
    api_key: Optional[str] = Field(default=None, description="LLM API key")
    model: Optional[str] = Field(default=None, description="Default model name")


class ComputerUseRequest(BaseModel):
    """Request model for computer use operations."""
    action: str = Field(..., description="Computer action to perform")
    parameters: dict = Field(default_factory=dict, description="Action parameters")


class BatchEvolveRequest(BaseModel):
    """Request model for batch evolution."""
    tool_names: Optional[List[str]] = Field(default=None, description="Specific tools to evolve (None = all)")
    feedback: Optional[str] = Field(default=None, description="General feedback for evolution")


# ────────────────────────────────────────────────────────────────────────────
# Initialize Agent (lazy)
# ────────────────────────────────────────────────────────────────────────────

_agent = None


def get_agent():
    """Lazy-initialize the agent singleton."""
    global _agent
    if _agent is None:
        try:
            from backend.agent_core import ZhuNaoAgent
        except ImportError:
            from agent_core import ZhuNaoAgent
        _agent = ZhuNaoAgent()
        logger.info("ZhuNaoAgent initialized")
    return _agent


# ────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CerebroForge (铸脑)",
    description="Self-evolving cognitive agent framework API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
# Exception Handlers
# ────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": time.time(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
        },
    )


# ────────────────────────────────────────────────────────────────────────────
# Health Check
# ────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CerebroForge",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Frontend
# ────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html><head><title>CerebroForge 铸脑</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #0a0a0a; color: #e0e0e0; }
h1 { color: #00d4ff; } .endpoint { background: #1a1a2e; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 3px solid #00d4ff; }
.endpoint .method { color: #4caf50; font-weight: bold; } .endpoint .path { color: #00d4ff; }
a { color: #00d4ff; }
</style></head><body>
<h1>🧠 CerebroForge (铸脑)</h1>
<p>Self-evolving cognitive agent framework</p>
<p><a href="/docs">📖 API Documentation</a> | <a href="/redoc">📄 ReDoc</a></p>
<h2>Available Endpoints</h2>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/chat</span> - Main chat endpoint</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/chat/stream</span> - SSE streaming chat</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/clarify</span> - Handle clarification</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/forge</span> - Forge new tool/skill</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/sleep</span> - Force memory compression</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/evolve</span> - Trigger evolution step</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/state</span> - Full agent state</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/memory</span> - Memory stats and items</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/tools</span> - Tool listing</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/evolution</span> - Evolution stats</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/interactions</span> - Recent interactions</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/upload</span> - File upload</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/config</span> - Current configuration</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/config/update</span> - Update config</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/memory/edit</span> - Memory viewer/editor</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/api/skills</span> - List dynamic skills</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/batch_evolve</span> - Batch evolution</div>
<div class="endpoint"><span class="method">POST</span> <span class="path">/api/computer</span> - Computer use operations</div>
<div class="endpoint"><span class="method">GET</span> <span class="path">/health</span> - Health check</div>
</body></html>""",
        status_code=200,
    )


# ────────────────────────────────────────────────────────────────────────────
# Chat Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat", tags=["chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - process a user query through the full cognitive pipeline.
    """
    try:
        agent = get_agent()
        result = agent.process(user_query=request.query, stream=request.stream)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@app.post("/api/chat/stream", tags=["chat"])
async def chat_stream(request: ChatRequest):
    """
    SSE streaming response endpoint.
    """
    async def event_generator():
        try:
            agent = get_agent()

            # Emit start event
            yield f"data: {json.dumps({'type': 'start', 'timestamp': time.time()})}\n\n"

            # Run agent in a thread to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: agent.process(user_query=request.query, stream=True)
            )

            # Stream the response in chunks
            response_text = result.get("response", "")
            chunk_size = 50

            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)

            # Emit metadata
            metadata = {k: v for k, v in result.items() if k != "response"}
            yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

            # Emit end event
            yield f"data: {json.dumps({'type': 'end', 'timestamp': time.time()})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clarify", tags=["chat"])
async def clarify(request: ClarifyRequest):
    """
    Handle clarification answers and re-run the agent.
    """
    try:
        agent = get_agent()
        result = agent.get_clarify_response(
            orig=request.original_query,
            answers=request.answers,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Clarify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clarification error: {str(e)}")


# ────────────────────────────────────────────────────────────────────────────
# Tool & Skill Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/forge", tags=["tools"])
async def forge_tool(request: ToolForgeRequest):
    """
    Forge a new tool/skill based on a requirement description.
    """
    try:
        try:
            from backend.skill_forge import get_skill_forge
        except ImportError:
            from skill_forge import get_skill_forge
        forge = get_skill_forge()
        result = forge.forge_skill(
            task_description=request.requirement,
            required_capabilities=[],
        )

        if result:
            return JSONResponse(content={
                "success": True,
                "tool": result,
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": "Tool forge failed - check logs for details",
            }, status_code=500)
    except Exception as e:
        logger.error(f"Forge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forge error: {str(e)}")


@app.get("/api/tools", tags=["tools"])
async def list_tools(include_evolved: bool = Query(default=True)):
    """
    List all available tools (base + dynamic).
    """
    try:
        agent = get_agent()
        tools_list = []
        for name in agent.tools.get_tool_names():
            tool_info = agent.tools.get_tool(name)
            if tool_info:
                is_dynamic = tool_info.get("is_dynamic", False) if isinstance(tool_info, dict) else False
                if not include_evolved and is_dynamic:
                    continue
                tools_list.append({
                    "name": name,
                    "description": tool_info.get("doc", "") if isinstance(tool_info, dict) else str(tool_info),
                    "is_evolved": is_dynamic,
                    "schema": tool_info.get("schema", {}) if isinstance(tool_info, dict) else {},
                })

        return JSONResponse(content={
            "tools": tools_list,
            "base_count": len([t for t in tools_list if not t["is_evolved"]]),
            "evolved_count": len([t for t in tools_list if t["is_evolved"]]),
            "total": len(tools_list),
        })
    except Exception as e:
        logger.error(f"Tools list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/skills", tags=["tools"])
async def list_skills():
    """
    List dynamic skills (evolved tools only).
    """
    try:
        agent = get_agent()
        dynamic = agent.tools.get_dynamic_tools()
        skills = [
            {
                "name": name,
                "description": info.get("doc", "") if isinstance(info, dict) else str(info),
                "code": info.get("code", "") if isinstance(info, dict) else "",
            }
            for name, info in dynamic.items()
        ]
        return JSONResponse(content={
            "skills": skills,
            "count": len(skills),
        })
    except Exception as e:
        logger.error(f"Skills list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# Memory Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/sleep", tags=["memory"])
async def force_sleep():
    """
    Force memory compression (L1 → L2, L2 → L3).
    """
    try:
        agent = get_agent()
        result = agent.force_compress()
        return JSONResponse(content={"success": True, "compression_result": result})
    except Exception as e:
        logger.error(f"Sleep error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory", tags=["memory"])
async def get_memory(
    level: Optional[str] = Query(default=None, description="Filter by level: L1, L2, L3"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    Get memory stats and items, optionally filtered by level.
    """
    try:
        agent = get_agent()
        stats = agent.memory.get_memory_stats()

        levels = [level] if level else None
        items = agent.memory.retrieve_relevant("", top_k=limit, levels=levels)

        return JSONResponse(content={
            "stats": stats,
            "items": items[:limit],
            "count": len(items[:limit]),
        })
    except Exception as e:
        logger.error(f"Memory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/edit", tags=["memory"])
async def memory_editor(
    action: str = Query(default="list", description="Action: list, delete, update"),
    item_id: Optional[str] = Query(default=None, description="Item ID for delete/update"),
    content: Optional[str] = Query(default=None, description="New content for update"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags for update"),
):
    """
    Memory viewer/editor endpoint.
    """
    try:
        agent = get_agent()

        if action == "list":
            stats = agent.memory.get_memory_stats()
            return JSONResponse(content={"stats": stats, "action": "list"})

        elif action == "delete":
            if not item_id:
                raise HTTPException(status_code=400, detail="item_id required for delete action")
            # SQLite-based delete would need to be implemented in MemorySystem
            return JSONResponse(content={"success": False, "error": "Direct deletion not yet supported via SQLite backend"})

        elif action == "update":
            if not item_id:
                raise HTTPException(status_code=400, detail="item_id required for update action")
            return JSONResponse(content={"success": False, "error": "Direct update not yet supported via SQLite backend"})

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory edit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# Evolution Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/evolve", tags=["evolution"])
async def trigger_evolution():
    """
    Trigger an evolution step.
    """
    try:
        agent = get_agent()
        agent._light_evolve()
        return JSONResponse(content={
            "success": True,
            "evolution": agent._evolution_metrics.model_dump(mode="json"),
        })
    except Exception as e:
        logger.error(f"Evolve error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evolution", tags=["evolution"])
async def get_evolution_stats():
    """
    Get evolution statistics.
    """
    try:
        agent = get_agent()
        evo_stats = agent.memory.get_evolution_stats()
        return JSONResponse(content={
            "evolution": {
                **agent._evolution_metrics.model_dump(mode="json"),
                **evo_stats,
            },
        })
    except Exception as e:
        logger.error(f"Evolution stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch_evolve", tags=["evolution"])
async def batch_evolve(request: BatchEvolveRequest):
    """
    Trigger batch evolution of tools.
    """
    try:
        try:
            from backend.skill_forge import get_skill_forge
        except ImportError:
            from skill_forge import get_skill_forge
        forge = get_skill_forge()
        result = forge.batch_evolve(
            tasks=request.tool_names or [],
        )
        return JSONResponse(content={
            "success": True,
            "result": result,
        })
    except Exception as e:
        logger.error(f"Batch evolve error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# State & Interactions Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/state", tags=["state"])
async def get_agent_state():
    """
    Get the full agent state.
    """
    try:
        agent = get_agent()
        return JSONResponse(content=agent.get_full_state())
    except Exception as e:
        logger.error(f"State error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interactions", tags=["state"])
async def get_interactions(limit: int = Query(default=20, ge=1, le=100)):
    """
    Get recent interaction history.
    """
    try:
        agent = get_agent()
        interactions = agent._interactions[-limit:]
        return JSONResponse(content={
            "interactions": interactions,
            "count": len(interactions),
            "total": len(agent._interactions),
        })
    except Exception as e:
        logger.error(f"Interactions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# Configuration Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.get("/api/config", tags=["config"])
async def get_config():
    """
    Get current configuration (with API key masked).
    """
    return JSONResponse(content={
        "base_url": NVIDIA_BASE_URL,
        "api_key": NVIDIA_API_KEY[:8] + "..." if NVIDIA_API_KEY and len(NVIDIA_API_KEY) > 8 else "***",
        "model": DEFAULT_MODEL,
    })


@app.post("/api/config/update", tags=["config"])
async def update_config(request: ConfigUpdate):
    """
    Update runtime configuration (base_url, api_key, model).
    """
    try:
        try:
            from backend.llm_client import llm
        except ImportError:
            from llm_client import llm

        llm.reconfigure(
            base_url=request.base_url,
            api_key=request.api_key,
            model=request.model,
        )

        return JSONResponse(content={
            "success": True,
            "base_url": llm.base_url,
            "model": llm.default_model,
            "api_key_set": bool(llm.api_key),
        })
    except Exception as e:
        logger.error(f"Config update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# File Upload
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["files"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the workspace directory.
    """
    try:
        import os
        filename = os.path.basename(file.filename or "unnamed")
        filepath = WORKSPACE_DIR / filename
        content = await file.read()
        filepath.write_bytes(content)

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "size": len(content),
            "path": str(filepath),
        })
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# Computer Use
# ────────────────────────────────────────────────────────────────────────────

@app.post("/api/computer", tags=["computer"])
async def computer_use(request: ComputerUseRequest):
    """
    Computer use operations.
    """
    try:
        agent = get_agent()
        result = agent.tools.execute("computer_use", {
            "action": request.action,
            **request.parameters,
        })
        return JSONResponse(content={
            "action": request.action,
            "result": result,
        })
    except Exception as e:
        logger.error(f"Computer use error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# Static Files (mount after all routes)
# ────────────────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
