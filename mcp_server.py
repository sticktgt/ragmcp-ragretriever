from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# Local modules you'll copy in
from .config import CONFIG
from .utils.logger import get_logger
from .storage.factory import get_vector_store
from .embedding.embedding import get_embedding_function

import asyncio

logger = get_logger("ragretriever")

# @dataclass
# class AppCtx:
#     store: Any

# The MCP server object; Streamable HTTP app is created from this.
mcp = FastMCP(
    name="RAG Retriever",
    instructions=(
        "Use `rag.search` to fetch the most relevant chunks from the RAG "
        "along with scores and provenance. Keep answers concise and cite sources."
    ),
)

# Serve the MCP Streamable HTTP transport at /mcp
mcp.settings.streamable_http_path = "/mcp"

# ---- lazy, concurrency-safe bootstrap (no custom lifespan) ----
_STORE: Any | None = None
_INIT_LOCK = asyncio.Lock()

async def _get_store() -> Any:
    global _STORE
    if _STORE is not None:
        return _STORE
    async with _INIT_LOCK:
        if _STORE is None:  # double-checked locking
            store_cfg = CONFIG.get("storage") or {}
            emb_cfg = CONFIG.get("embedding") or {}
            emb = get_embedding_function(emb_cfg)
            _STORE = get_vector_store(store_cfg, emb)
            logger.info("[MCP] retriever initialized (store=%s)",
                        store_cfg.get("store_type") or "unknown")
    return _STORE

# ----- tool implementation -----
@mcp.tool(name="rag.search")
async def rag_search(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,  # reserved for later
    ctx: Context[ServerSession, object] | None = None,
) -> Dict[str, Any]:
    """
    Vector search only (no LLM). Return top-k chunks with scores and provenance.
    """
    store = await _get_store()
    results = store.similarity_search_with_score(query, k=k)

    items: List[Dict[str, Any]] = []
    for doc, score in results:
        items.append({
            "text": getattr(doc, "page_content", "") or "",
            "score": float(score),
            "provenance": dict(getattr(doc, "metadata", {}) or {}),
        })

    # Optional: client log hint (if ctx supported by your SDK version)
    if ctx:
        preview = ", ".join([f"{it['score']:.3f}·{it['provenance'].get('original_name','?')}" for it in items[:5]])
        await ctx.info(f"rag.search → {len(items)} hits [{preview}]")

    return {"results": items}

# Export the FastMCP ASGI app directly (let *its* lifespan run!)
app = mcp.streamable_http_app()
