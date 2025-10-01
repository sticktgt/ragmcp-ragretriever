from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # base class type
from typing import List, Optional, Tuple
# from pymilvus import Collection, utility
from ..utils.logger import get_logger
import traceback
from .base import VectorStoreBase

logger = get_logger()


class MilvusStore(VectorStoreBase):

    def __init__(self, cfg: dict, embedding_function: Embeddings):
        try:
            self.cfg = cfg["milvus"]
            self.vstore = Milvus(
                collection_name=cfg["milvus"]["collection"],
                connection_args={"uri": cfg["milvus"]["uri"]},
                embedding_function=embedding_function,
                drop_old=cfg["milvus"]["drop_old"],
                auto_id=cfg["milvus"]["auto_id"],
            )
            logger.info(f"Using Milvus at {cfg["milvus"]['uri']}")
        except Exception as e:
            logger.error(f"[MILVUS INIT ERROR]: {e}")
            logger.debug(traceback.format_exc())
            raise


    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        try:
            logger.info(f"Performing vector search for: {query}")
            results = self.vstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"[MILVUS QUERY ERROR]: {e}")
            logger.debug(traceback.format_exc())
            return []        