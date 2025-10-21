from __future__ import annotations
# from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from time import perf_counter

# from fastapi import FastAPI, Response
# from fastapi.responses import JSONResponse

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

        "Используй инструмент `rag.search`, когда для ответа нужны фрагменты из RAG.\n"
        "Возвращай краткие ответы и обязательно цитируй источники из `provenance` (в приоритете `uri`, иначе `original_name`).\n\n"

        "Аргументы `rag.search` (объект JSON):\n"
        "- query: string — формулируй на языке пользователя.\n"
        "- k: integer (1..100, по умолчанию 5) — размер пула кандидатов.\n"
        "- filters: object|null — детерминированная фильтрация по метаданным (допускаются любые ключи).\n"
        "- rerank: boolean (по умолчанию false) — включай только если нужна повышенная точность.\n"
        "- top_n: integer|null — финальная обрезка после переранжировки.\n\n"

        "Правила вызовов инструментов:\n"
        "- Если нужен доступ к RAG — верни **вызов функции** (tool call) `rag.search` с объектом аргументов.\n"
        "- **Не печатай** вызов как текст. **Не используй** кодовые блоки/бэктики.\n"
        "- Не добавляй неописанные поля в аргументы.\n"
        "- Получив результаты инструмента, сформируй финальный ответ и процитируй источники." \
    ),
)

# Serve the MCP Streamable HTTP transport at /mcp
mcp.settings.streamable_http_path = "/mcp"

# ---- lazy, concurrency-safe bootstrap (no custom lifespan) ----
_STORE: Any | None = None
_INIT_LOCK = asyncio.Lock()

_limits = CONFIG.get("limits") or {}
_MAX_CONCURRENT_SEARCH = int(_limits.get("max_concurrent_search", 32))
_MAX_CONCURRENT_RERANK = int(_limits.get("max_concurrent_rerank", 4))

_SEARCH_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_SEARCH)
_RERANK_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_RERANK)

async def _get_store() -> Any:
    global _STORE
    if _STORE is not None:
        return _STORE
    async with _INIT_LOCK:
        if _STORE is None:  # double-checked locking
            store_cfg = CONFIG.get("storage") or {}
            emb_cfg = CONFIG.get("embedding") or {}
            try:
                emb = get_embedding_function(emb_cfg)
                _STORE = get_vector_store(store_cfg, emb)
                logger.info("[MCP] retriever initialized (store=%s)",
                            store_cfg.get("store_type") or "unknown")
            except Exception:
                logger.exception("[INIT] Vector store initialization failed (store_type=%s)", store_cfg.get("store_type"))
                raise                
    return _STORE

# ----- tool implementation -----
@mcp.tool(name="rag.search", description=schema.TOOL_DESCRIPTION)
async def rag_search(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    rerank: bool = False,
    top_n: Optional[int] = None,
) -> schema.RagSearchOutput:
    input = schema.RagSearchArgs(
        query=query, k=k, filters=filters, rerank=rerank, top_n=top_n
    )

    query   = input.query
    k       = input.k
    filters = input.filters
    top_n   = input.top_n
    rerank  = bool(input.rerank)
    
# 1) vector search (with filters)
    logger.debug(f"[SEARCH] query={query!r} k={k} filters={(filters or {})!r} rerank={rerank} top_n={top_n}")

    try:
        store = await _get_store()
    except Exception as e:
        error_message = f"[INIT]: {e}"
        logger.error(error_message)
        return {"results": [], "error": error_message}

    loop = asyncio.get_running_loop()
    try:
        t0 = perf_counter()
        async with _SEARCH_SEMAPHORE:
            # run blocking search in executor so we don't block the event loop
            results = await loop.run_in_executor(
                None,
                lambda: store.similarity_search_with_score(query, k=k, filters=filters),
            )
        logger.debug("[METRIC] vec_ms=%.1f k=%d size=%d",
                     (perf_counter() - t0) * 1000, k, len(results) if results else 0)
    except Exception as e:
        error_message = f"[SEARCH]: {e}"
        logger.error(error_message)
        return {"results": [], "error": error_message}

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
                t0 = perf_counter()
                async with _RERANK_SEMAPHORE:
                    items = await rerank_items_from_config(query, items, rerank_cfg)
                logger.debug("[METRIC] rerank_ms=%.1f", (perf_counter() - t0) * 1000)
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
    return schema.RagSearchOutput(results=items)

# Export the FastMCP ASGI app directly 
app = mcp.streamable_http_app()
