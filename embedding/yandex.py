from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from ..utils.logger import get_logger
from .base import BaseEmbedder

logger = get_logger()

class YandexEmbedder(BaseEmbedder):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        logger.debug("Using YandexGPT embeddings provider.")

    def get_embedding_function(self) -> Embeddings:
        return YandexGPTEmbeddings(
            api_key=self.cfg["api_key"],
            folder_id=self.cfg["folder_id"],
            doc_model_name=self.cfg.get("doc_model_name", "text-search-doc"),
            model_version=self.cfg.get("model_version", "latest"),
            disable_request_logging=self.cfg.get("disable_request_logging", False),
            sleep_interval=self.cfg.get("sleep_interval", 2.0),
            grpc_metadata=[],
        )

    def embed_documents(self, docs: List[Document]) -> List[dict]:
        embedder = self.get_embedding_function()
        vectors = embedder.embed_documents([doc.page_content for doc in docs])
        return [
            {"embedding": vec, "metadata": doc.metadata}
            for doc, vec in zip(docs, vectors)
        ]
