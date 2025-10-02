from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # base class type
from typing import List, Optional, Tuple
# from pymilvus import Collection, utility
from ..utils.logger import get_logger
import traceback
from .base import VectorStoreBase

logger = get_logger()

def _to_milvus_expr(filters: dict | None) -> Optional[str]:
    if not filters:
        return None
    clauses = []
    for k, v in filters.items():
        if isinstance(v, str):
            # escape quotes
            v = v.replace('"', '\\"')
            clauses.append(f'{k} == "{v}"')
        elif isinstance(v, (int, float)):
            clauses.append(f"{k} == {v}")
        elif isinstance(v, bool):
            clauses.append(f"{k} == {str(v).lower()}")
        else:
            # ignore complex types for now
            continue
    return " and ".join(clauses) if clauses else None

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


    def similarity_search_with_score(self, query: str, k: int = 5, filters: Optional[dict] = None) -> List[Tuple[Document, float]]:
        try:
            logger.info(f"Performing vector search for: {query}")

            expr = _to_milvus_expr(filters)
            results = self.vstore.similarity_search_with_score(query, k=k, expr=expr) if expr else \
                      self.vstore.similarity_search_with_score(query, k=k)

            return results
        except Exception as e:
            logger.error(f"[MILVUS QUERY ERROR]: {e}")
            logger.debug(traceback.format_exc())
            return []        