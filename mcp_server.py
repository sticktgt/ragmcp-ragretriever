from __future__ import annotations
# from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from .config import CONFIG
from .utils.logger import get_logger
from .storage.factory import get_vector_store
from .embedding.embedding import get_embedding_function

from .rerank import rerank_items_from_config

from .schemas import ru as schema

import asyncio

logger = get_logger()

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
@mcp.tool(name="rag.search", description=schema.TOOL_DESCRIPTION)
async def rag_search(
    input: schema.RagSearchArgs,
) -> schema.RagSearchOutput:
    query   = input.query
    k       = input.k
    filters = input.filters
    top_n   = input.top_n
    rerank = bool(input.rerank)
    
# 1) vector search (with filters)
    logger.debug(f"[SEARCH] query={query!r} k={k} filters={(filters or {})!r} rerank={rerank} top_n={top_n}")

    store = await _get_store()
    try:
        results = store.similarity_search_with_score(query, k=k, filters=filters)
    except Exception as e:
        return {"results": [], "error": "search_failed"}

    items: List[Dict[str, Any]] = []
    for doc, score in results:
        items.append({
            "text": getattr(doc, "page_content", "") or "",
            "score": float(score),
            "provenance": dict(getattr(doc, "metadata", {}) or {}),
        })
    logger.debug(f"[SEARCH] Retrieved {len(items)} candidates (k={k}, filters={(filters or {})!r})")    

    # 2) Optional LLM rerank (via LangChain models)

    #decide whether to rerank
    rerank_cfg = CONFIG.get("rerank") or {}
    cfg_provider = (rerank_cfg.get("provider") or "none").strip().lower()
    do_rerank = bool(rerank)  # from the request

    if do_rerank:
        if cfg_provider in ("none", ""):
            logger.warning("[RERANK] Requested by client, but CONFIG['rerank']['provider'] is not set; skipping")
        else:
            try:
                before_preview = [it["provenance"].get("original_name", "?") for it in items[:3]]
                items = await rerank_items_from_config(query, items, rerank_cfg)
                after_preview = [it["provenance"].get("original_name", "?") for it in items[:3]]
                logger.debug(f"[RERANK] Provider={cfg_provider} before={before_preview} after={after_preview}")
            except Exception as e:
                logger.error(f"[RERANK] Unexpected failure: {e}")

    # 3) apply top_n AFTER rerank (or after vector if no rerank)
    if isinstance(top_n, int) and top_n > 0:
        if top_n < len(items):
            items = items[:top_n]
        logger.debug(f"[TRIM] Applied top_n={top_n}; final={len(items)}")

    # 4) client hint
    # Optional: client log hint (if ctx supported)


    # 5) Typed return; Pydantic will coerce dicts to SearchHit models
    # return {"results": items}
    return schema.RagSearchOutput(results=items)

# Export the FastMCP ASGI app directly 
app = mcp.streamable_http_app()
