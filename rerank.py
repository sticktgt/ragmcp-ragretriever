from __future__ import annotations
from typing import Dict, Any, List
import json, math
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .llm.factory import get_chat_model_from_config
from .utils.logger import get_logger

logger = get_logger()

# ---------- Prompts ----------

_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system",
     "Output only compact JSON. No prose, no code fences, no keys, no trailing text."),
    ("user",
     "You rank knowledge snippets by relevance.\n"
     "Query: {query}\n"
     "You are given a JSON array of snippets. There are exactly {n} snippets.\n"
     "Return a JSON array of exactly {n} floating-point scores in [0.0, 1.0]\n"
     "(1.0 = maximally relevant). If a snippet is irrelevant or unclear, use 0.0.\n"
     "Return ONLY the JSON array, nothing else.\n"
     "{snippets_json}")
])

_PROMPT_RU = ChatPromptTemplate.from_messages([
    ("system",
     "Выводи только компактный JSON. Без пояснительного текста, без кодовых блоков, только массив."),
    ("user",
     "Ты оцениваешь фрагменты знаний по релевантности.\n"
     "Запрос: {query}\n"
     "Тебе дан JSON-массив из {n} фрагментов.\n"
     "Верни JSON-массив из ровно {n} чисел с плавающей точкой в диапазоне [0.0, 1.0].\n"
     "1.0 — максимально релевантный, 0.0 — нерелевантный или непонятный.\n"
     "Выведи ТОЛЬКО JSON-массив, ничего больше.\n"
     "{snippets_json}")
])

_PARSER = JsonOutputParser()

# ---------- Helpers ----------

def _normalize_scores(raw, n: int) -> List[float]:
    scores: List[float] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return [0.0] * n
    if not isinstance(raw, list):
        return [0.0] * n
    for x in raw:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        scores.append(v)
    if len(scores) < n:
        scores.extend([0.0] * (n - len(scores)))
    elif len(scores) > n:
        scores = scores[:n]
    return scores

# ---------- Main ----------

async def rerank_items_from_config(
    query: str, items: List[Dict[str, Any]], rerank_cfg: Dict[str, Any], lang: str = "ru"
) -> List[Dict[str, Any]]:
    """
    lang: "en" -> use English prompt, "ru" -> Russian prompt
    """
    if not items:
        logger.debug("[RERANK] No items to rerank")
        return items

    provider = (rerank_cfg.get("provider") or "none").strip().lower()
    if provider in ("none", ""):
        logger.debug("[RERANK] Provider disabled in config; skipping")
        return items

    texts = [it["text"] for it in items]
    n = len(texts)

    try:
        llm = get_chat_model_from_config(rerank_cfg)
        prompt = _PROMPT_RU if lang.lower().startswith("ru") else _PROMPT_EN
        chain = prompt | llm | _PARSER
    except Exception as e:
        logger.error(f"[RERANK] Model init failed: {e}")
        return items

    try:
        payload = {"query": query, "snippets_json": json.dumps(texts, ensure_ascii=False), "n": n}
        raw = await chain.ainvoke(payload)
        logger.debug(f"[RERANK] raw model output: {raw}")
    except Exception as e:
        logger.error(f"[RERANK] LLM call failed: {e}")
        return items

    

    scores = _normalize_scores(raw, n)
    if len(scores) != n:
        logger.error(f"[RERANK] Unexpected scores length after normalize: got={len(scores)} expected={n}")
        return items
    if isinstance(raw, list) and len(raw) != n:
        logger.warning(f"[RERANK] Model returned {len(raw)} scores, normalized to {n}")

    logger.debug(f"[RERANK] Normalized scores: {scores}")

    ranked = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
    out = [it for _, it in ranked]
    logger.debug(f"[RERANK] Completed. n={len(out)} top1_score={scores[0] if scores else 'n/a'}")
    return out
