import os
import json
import aiohttp
import time
import uuid
from typing import List, Dict, Any
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta

# How much tool output to keep if a tool message arrives
_TOOL_FLATTEN_MAX_CHARS = int(os.getenv("TOOL_FLATTEN_MAX_CHARS", "1200"))

def _safe_join_content(content: Any) -> str:
    """
    OpenAI content can be str or a list of parts; normalize to string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                # prefer text; ignore images for now
                txt = p.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content) if content is not None else ""

def _truncate(s: str, n: int) -> str:
    if not s or len(s) <= n:
        return s or ""
    return s[:n] + "…"

def _int_usage_from_text(txt: str) -> Dict[str, int]:
    # If you don’t count tokens, return reasonable integers (no nulls)
    approx = max(1, len(txt) // 4)
    return {"prompt_tokens": 0, "completion_tokens": approx, "total_tokens": approx}

def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    try:
        # pydantic v2
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # pydantic v1
        if hasattr(obj, "dict"):
            return obj.dict()
    except Exception:
        pass
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"raw": str(obj)}
    
def _chunk_text(s: str, n: int = 200):
    for i in range(0, len(s), n):
        yield s[i:i+n]    

class YandexCustomLLM:
    """
    Custom LiteLLM provider for YandexGPT (OpenAI-incompatible shape -> OpenAI Chat shape).
    Implements:
      - aembedding(input, model, **kwargs) -> Embeddings-like dict
      - acompletion(messages, model, **kwargs) -> Chat Completions dict
      - astreaming(messages, model, **kwargs) -> yields 2 valid chat.completion.chunk dicts
    Environment fallbacks:
      YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL, YANDEX_DISABLE_LOGGING
    """

    def __init__(self, *args, **kwargs):
        self.default_api_key = os.getenv("YANDEX_API_KEY")
        if self.default_api_key:
            print("[Yandex] Using default YANDEX_API_KEY from environment")
        self.default_folder_id = os.getenv("YANDEX_FOLDER_ID")
        if self.default_folder_id:
            print("[Yandex] Using default YANDEX_FOLDER_ID from environment")
        self.default_model = os.getenv("YANDEX_MODEL", "yandexgpt-lite/rc")
        if self.default_model:
            print(f"[Yandex] Using default YANDEX_MODEL='{self.default_model}' from environment")

    # ---------- Embeddings ----------
    async def aembedding(self, input, model, **kwargs):
        print(f"[Yandex] aembedding called")
        # Read user-scoped metadata if provided (tolerant)
        user_info_str = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key_end_user_id", "")
        try:
            meta = json.loads(user_info_str) if user_info_str else {}
        except Exception:
            meta = {}

        api_key = meta.get("api_key")
        if not api_key:
            api_key = self.default_api_key
        else:
            print("[Yandex] Using API key from metadata")
        folder_id = meta.get("folder_id")
        if not folder_id:
            folder_id = self.default_folder_id
        else:
            print("[Yandex] Using folder_id from metadata")
        if not api_key or not folder_id:
            raise Exception("Missing folder_id or api_key for embeddings")

        # Ensure list
        texts = input if isinstance(input, list) else [input]

        model_uri = f"emb://{folder_id}/text-search-doc/latest"
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}

        data_items = []
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for text in texts:
                payload = {"modelUri": model_uri, "text": text}
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        raise Exception(f"Yandex Embeddings Error: {await resp.text()}")
                    jd = await resp.json()
                    emb = jd.get("embedding")
                    if not emb:
                        raise Exception("No embedding in response")
                    data_items.append({"embedding": emb})

        # OpenAI embedding-like shape (minimal)
        return {"object": "list", "data": data_items}

    # ---------- Chat Completions (non-stream) ----------
    async def acompletion(self, messages, model, **kwargs):
        print(f"[Yandex] acompletion called")
        # Metadata (tolerant)
        user_info_str = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key_end_user_id", "")
        try:
            meta = json.loads(user_info_str) if user_info_str else {}
        except Exception:
            meta = {}

        # Fallthrough order: meta -> env -> route arg -> default
        api_key = meta.get("api_key")
        if not api_key:
            api_key = self.default_api_key
        else:
            print("[Yandex] Using API key from metadata")
        folder_id = meta.get("folder_id")
        if not folder_id:
            folder_id = self.default_folder_id
        else:
            print("[Yandex] Using folder_id from metadata")
        yandex_model = meta.get("yandex_model")
        if not yandex_model:
            yandex_model = self.default_model or "yandexgpt-lite/rc"
        else:
            print(f"[Yandex] Using model '{yandex_model}' from metadata or route")

        if not api_key or not folder_id:
            raise Exception("Missing folder_id or api_key for chat")

        # Disable logging?
        disable_logging = meta.get("disable_logging")
        if disable_logging is None:
            disable_logging = os.getenv("YANDEX_DISABLE_LOGGING", "false").lower() in ("1", "true", "yes", "on")
        else:
            disable_logging = bool(disable_logging)

        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", None)

        headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
        if disable_logging:
            headers["x-data-logging-enabled"] = "false"

        # DEBUG
        print(f"[Yandex] Request: {messages}")

        # Flatten OpenAI messages -> Yandex messages
        ynx_messages: List[Dict[str, str]] = []
        for m in messages or []:
            role = m.get("role", "user")
            text = _safe_join_content(m.get("content", ""))

            # 1) Tool outputs -> fold into 'user' text for Yandex
            if role == "tool":
                name = m.get("name") or "tool"
                txt = _truncate(text, _TOOL_FLATTEN_MAX_CHARS)
                # always send non-empty text to Yandex
                if not txt:
                    txt = "(no tool output provided)"
                ynx_messages.append({"role": "user", "text": f"Tool output ({name}):\n{txt}"})
                continue

            # 2) Assistant tool-call turn (usually has empty content) -> skip or summarize
            if role == "assistant" and (not text or not text.strip()):
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    # Optional: summarize what the assistant tried to call (helps the model)
                    try:
                        summaries = []
                        for tc in tool_calls:
                            fn = (tc.get("function") or {})
                            nm = fn.get("name") or "function"
                            args = fn.get("arguments")
                            if isinstance(args, dict):
                                args = json.dumps(args, ensure_ascii=False)
                            summaries.append(f"{nm}({args})")
                        if summaries:
                            ynx_messages.append({"role": "user", "text": "Assistant requested tool call(s): " + "; ".join(summaries)})
                    except Exception:
                        pass
        # Do NOT forward an empty assistant message
                continue

            # 3) Skip any other empty messages to avoid Yandex 400
            if not text or not text.strip():
                continue

            # 4) Regular messages
            ynx_messages.append({"role": role, "text": text})

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        model_uri = f"gpt://{folder_id}/{yandex_model}"

        completion_opts = {
            "stream": False,
            "temperature": float(temperature) if temperature is not None else 0.2,
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            completion_opts["maxTokens"] = max_tokens

        payload = {
            "modelUri": model_uri,
            "completionOptions": completion_opts,
            "messages": ynx_messages,
        }

        # DEBUG
        print(f"[Yandex] Request payload: {json.dumps(payload, ensure_ascii=False)}")

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Yandex Chat Error: {await resp.text()}")
                jd = await resp.json()

        # DEBUG
        print(f"[Yandex] Response: {json.dumps(jd, ensure_ascii=False)}")

        # ---- Extract first alternative ----
        result_root = jd.get("result", {}) or {}
        alts = result_root.get("alternatives", []) or []
        alt = alts[0] if alts else {}
        msg = alt.get("message", {}) or {}
        status = alt.get("status")

        # ---- Map usage (Yandex -> OpenAI) ----
        u = result_root.get("usage", {}) or {}
        def _to_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default
        usage_mapped = {
            "prompt_tokens": _to_int(u.get("inputTextTokens", 0)),
            "completion_tokens": _to_int(u.get("completionTokens", 0)),
            "total_tokens": _to_int(u.get("totalTokens", 0)),
        }
        reasoning = int((u.get("completionTokensDetails") or {}).get("reasoningTokens", 0))
        if reasoning:
            usage_mapped["completion_tokens_details"] = {"reasoning_tokens": reasoning}
        # ---- Detect tool calls vs plain text ----
        tool_calls = None
        tool_call_list = msg.get("toolCallList")
        if isinstance(tool_call_list, dict):
            raw_calls = tool_call_list.get("toolCalls") or []
            tool_calls = []
            for rc in raw_calls:
                fc = (rc or {}).get("functionCall") or {}
                name = fc.get("name") or "function"
                args_obj = fc.get("arguments") or {}
                # OpenAI expects arguments as a JSON string
                args_str = json.dumps(args_obj, ensure_ascii=False)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                })        

        now = int(time.time())
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": yandex_model,
            "usage": usage_mapped,
            # drop x_yandex_usage to avoid duplication
            # "x_yandex_usage": u,
        }

        if tool_calls:
            # Return *instructions* for tool execution, not JSON text
            openai_choice = {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
            }
        else:
            # Plain assistant text
            text = msg.get("text")
            if not isinstance(text, str):
                text = "" if text is None else str(text)

            # As a last resort, don’t dump the whole JD into content anymore
            if not text:
                text = ""

            openai_choice = {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        result = {**base, "choices": [openai_choice]}

        # DEBUG
        print(f"[Yandex] Result: {json.dumps(result, ensure_ascii=False)}")
        return result

    # ---------- Chat Completions (streaming) ----------
    async def astreaming(self, messages, model, **kwargs):
        """
        Anthropic-shaped streaming for LiteLLM.
        - Plain text: unchanged (your last working behavior).
        - Tool calls: emit Anthropic event-style content blocks for tool_use,
          and ALWAYS include a "text" key (even "") on every chunk to satisfy LiteLLM.
        """
        print(f"[Yandex] astreaming called")
        kwargs = {**kwargs}
        kwargs.pop("stream", None)

        # Reuse your non-streaming conversion (already correct for tool_calls + usage)
        full = await self.acompletion(messages, model, **kwargs)

        first_choice = (full.get("choices") or [{}])[0]
        msg = first_choice.get("message") or {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        finish_reason = first_choice.get("finish_reason") or "stop"

        u = full.get("usage") or {}
        usage_stream = {
            "input_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
            "total_tokens": int(u.get("total_tokens") or 0),
        }
        # if present
        reasoning = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0))
        if reasoning:
            usage_stream["completion_tokens_details"] = {"reasoning_tokens": reasoning}

        # ---------- TOOL CALLS PATH ----------
        if isinstance(tool_calls, list) and tool_calls:
            # optional priming chunk so parsers are happy
            yield {"text": "", "is_finished": False, "finish_reason": None, "usage": None}

            # Emit one OPENAI-SHAPED tool call per chunk via GenericStreamingChunk.tool_use
            for idx, tc in enumerate(tool_calls):
                fn = (tc.get("function") or {})
                name = fn.get("name") or "function"
                args = fn.get("arguments") or "{}"
                if not isinstance(args, str):
                    try:
                        import json
                        args = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        args = str(args)

                tool_call_chunk = {
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,   # MUST be a STRING per OpenAI shape
                    },
                    "index": idx,
                }

                yield {
                    "text": "",                    # keep anthropic-simple keys present
                    "tool_use": tool_call_chunk,   # <— LiteLLM reads this into delta.tool_calls[*]
                    "is_finished": False,
                    "finish_reason": None,
                    "usage": None,
                }

            # Final stop chunk
            yield {
                "text": "",
                "is_finished": True,
                "finish_reason": "tool_calls",     # LiteLLM maps this correctly
                "usage": usage_stream,
            }
            return

        # ---------- PLAIN TEXT PATH (unchanged) ----------
        step = 240
        if not isinstance(content, str):
            content = "" if content is None else str(content)

        if not content:
            yield {"text": "", "is_finished": False, "finish_reason": None, "usage": None}
        else:
            for i in range(0, len(content), step):
                piece = content[i:i+step]
                yield {"text": piece, "is_finished": False, "finish_reason": None, "usage": None}

        yield {
            "text": "",
            "is_finished": True,
            "finish_reason": finish_reason or "stop",
            "usage": usage_stream if usage_stream["total_tokens"] else {
                "input_tokens": 0,
                "output_tokens": max(1, len(content) // 4),
                "total_tokens": 0,
            },
        }


# IMPORTANT: this identifier must match litellm.yaml -> custom_provider_map.custom_handler
my_custom_llm = YandexCustomLLM()
