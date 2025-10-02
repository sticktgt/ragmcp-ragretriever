import os
import json
import aiohttp
from litellm.types.utils import EmbeddingResponse, TextCompletionResponse

class YandexCustomLLM:
    def __init__(self, *args, **kwargs):
        # Optional: fallback env vars
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_key = os.getenv("YANDEX_API_KEY")

    async def aembedding(self, input, model, **kwargs):
        # Extract folder_id and api_key from `user` JSON string passed via LangChain
        user_info_str = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key_end_user_id", "")
        try:
            user_info = json.loads(user_info_str)
            api_key = user_info.get("api_key") or self.api_key
            folder_id = user_info.get("folder_id") or self.folder_id
            # disable_logging = bool(user_info.get("disable_logging", False))
            # emb_model = user_info.get("embedding_model") or "text-search-doc"  # Yandex embedding family            
        except Exception as e:
            raise Exception(f"Failed to parse folder_id/api_key from: {user_info_str}. Error: {e}")

        if not folder_id or not api_key:
            raise Exception("Missing folder_id or api_key")

        # Ensure we're working with a list of texts
        if not isinstance(input, list):
            input = [input]

        model_uri = f"emb://{folder_id}/text-search-doc/latest" # override via user_info.get("embedding_model")
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json"
        }
        # if disable_logging:
        #     headers["x-data-logging-enabled"] = "false"

        payload = {
            "modelUri": model_uri,
            "text": input
        }

        # Remove OpenAI-specific params
        for param in ("encoding_format", "user"):
            kwargs.pop(param, None)

        results = []
        async with aiohttp.ClientSession() as session:
            for text in input:
                payload = {
                    "modelUri": model_uri,
                    "text": text
                }

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Yandex API Error: {await response.text()}")
                    result = await response.json()
                    embedding = result.get("embedding")
                    if not embedding:
                        raise Exception("No embedding in response")
                    results.append({"embedding": embedding})

        return EmbeddingResponse(data=results)

    async def acompletion(self, messages, model, **kwargs):

        user_info_str = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key_end_user_id", "")
        try:
            meta = json.loads(user_info_str) if user_info_str else {}
        except Exception as e:
            raise Exception(f"Failed to parse `user` JSON: {e}")

        api_key   = meta.get("api_key")   or self.default_api_key
        folder_id = meta.get("folder_id") or self.default_folder_id
        disable_logging = bool(meta.get("disable_logging", False))
        yandex_model = meta.get("yandex_model") or model  # fall back to the route's model string

        if not api_key or not folder_id:
            raise Exception("Missing folder_id or api_key for chat")

        # Headers
        headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
        if disable_logging:
            headers["x-data-logging-enabled"] = "false"

        # Convert OpenAI messages -> Yandex format
        ynx_messages = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
            ynx_messages.append({"role": role, "text": content})

        # Payload
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        model_uri = f"gpt://{folder_id}/{yandex_model}/latest"
        temperature = kwargs.get("temperature", 0.0)

        payload = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                # "maxTokens": 800,  # optional hard cap if you want
            },
            "messages": ynx_messages,
        }

        # Call Yandex
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"Yandex Chat Error: {await resp.text()}")
                jd = await resp.json()

        # Extract assistant text
        try:
            text = jd["result"]["alternatives"][0]["message"]["text"]
        except Exception:
            text = json.dumps(jd, ensure_ascii=False)  # fallback for debugging

        # Normalize to OpenAI-like response for LiteLLM
        return TextCompletionResponse(
            id="yandex-chat",
            object="chat.completion",
            model=model,
            created=None,
            usage=None,
            choices=[{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }],
        )

# IMPORTANT: this must match the litellm.yaml config
my_custom_llm = YandexCustomLLM()
