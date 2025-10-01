from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class BaseEmbedder(ABC):
    @abstractmethod
    def get_embedding_function(self) -> Embeddings:
        pass

    @abstractmethod
    def embed_documents(self, docs: List[Document]) -> List[dict]:
        pass
