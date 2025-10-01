from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from langchain_core.documents import Document

class VectorStoreBase(ABC):
    @abstractmethod

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        pass