from __future__ import annotations
from typing import Dict, Any
import json

from langchain_openai import ChatOpenAI
from langchain_community.chat_models.yandex import ChatYandexGPT
from langchain_core.language_models import BaseLanguageModel as _BaseModel


def get_chat_model_from_config(rerank_cfg: Dict[str, Any]) -> _BaseModel:
    """
    Build a LangChain chat model from CONFIG['rerank'].
    Supports:
      - provider: "litellm"    (OpenAI-compatible proxy via ChatOpenAI)
      - provider: "yandexgpt"  (direct via ChatYandexGPT)
      - provider: "none"       (handled by caller)
    """
    provider = (rerank_cfg.get("provider") or "none").strip().lower()

    if provider == "litellm":
        p = rerank_cfg.get("liteLLM") or {}
        api_base = p["api_base"]                # must include '/v1'
        api_key  = p["api_key"]
        # the PROXY ROUTE name in litellm.yaml
        model    = p.get("model")
        temperature = float(p.get("temperature"))
        timeout     = float(p.get("timeout"))

        # Extra payload for your custom handler (read from OpenAI 'user' field):
        y_user = json.dumps({
            "folder_id": p["folder_id"],
            "api_key":   p["api_key"],
            "yandex_model": p.get("yandex_model"),
            "disable_logging": bool(p.get("disable_logging", False)),
        })

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            timeout=timeout,
            user=y_user,   # same slot as in embeddings
        )

    if provider == "yandexgpt":
        p = rerank_cfg.get("yandexGPT") or {}
        return ChatYandexGPT(
            api_key=p["api_key"],
            folder_id=p["folder_id"],
            model=p.get("model"),
            temperature=float(p.get("temperature")),
            timeout=float(p.get("timeout")),
        )

    raise RuntimeError("No rerank provider configured")
