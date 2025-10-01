from ..config import CONFIG
from .yandex import YandexEmbedder
from .dummy import CustomDummyEmbedder, LangchainFakeEmbedder
from .litellm import LiteLLMEmbedder
from .base import BaseEmbedder


def get_embedder(cfg: dict) -> BaseEmbedder:
    provider = cfg.get("provider", "fake")

    if provider == "yandexGPT":
        return YandexEmbedder(cfg["yandexGPT"])
    elif provider == "liteLLM":
        return LiteLLMEmbedder(cfg["liteLLM"])
    elif provider == "langchain_fake":
        return LangchainFakeEmbedder()
    else:
        return CustomDummyEmbedder()
