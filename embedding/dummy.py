from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from ..config import CONFIG
from .base import BaseEmbedder
from ..utils.logger import get_logger
from langchain_community.embeddings import FakeEmbeddings

logger = get_logger()

#  1. LangChain built-in FakeEmbedder
class LangchainFakeEmbedder(BaseEmbedder):
    def __init__(self):
        logger.debug("Using LangchainFakeEmbedder embeddings provider.")    

    def get_embedding_function(self) -> Embeddings:
        return FakeEmbeddings(size=CONFIG["embedding"]["dim"])

    def embed_documents(self, docs: List[Document]) -> List[dict]:
        embedder = self.get_embedding_function()
        vectors = embedder.embed_documents([doc.page_content for doc in docs])
        return [
            {"embedding": vec, "metadata": doc.metadata}
            for doc, vec in zip(docs, vectors)
        ]


#  2. Your custom dummy implementation
class CustomDummyEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        dim = CONFIG["embedding"]["dim"]
        return [[0.01 * i for i in range(1, dim + 1)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class CustomDummyEmbedder(BaseEmbedder):
    def __init__(self):
        logger.debug("Using CustomDummyEmbedder embeddings provider.")

    def get_embedding_function(self) -> Embeddings:
        return CustomDummyEmbeddings()

    def embed_documents(self, docs: List[Document]) -> List[dict]:
        embedder = self.get_embedding_function()
        vectors = embedder.embed_documents([doc.page_content for doc in docs])
        return [
            {"embedding": vec, "metadata": doc.metadata}
            for doc, vec in zip(docs, vectors)
        ]
