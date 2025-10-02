import logging
import traceback
import psycopg2
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_postgres import PGVector
from .base import VectorStoreBase
from ..utils.logger import get_logger

logger = get_logger()

class PGVectorStore(VectorStoreBase):
    def __init__(self, cfg: dict, embedding_function):
        self.cfg = cfg["pgvector"]
        self.pg_config = {
            "host": cfg["pgvector"]["host"],
            "port": cfg["pgvector"].get("port", 5432),
            "user": cfg["pgvector"]["user"],
            "password": cfg["pgvector"]["password"],
            "dbname": cfg["pgvector"]["database"],
        }

        # Build the connection string dynamically
        logger.info(f"Using PGVector at {cfg["pgvector"]['host']}:{cfg["pgvector"]['port']}")
        connection_string = (
            f"postgresql+psycopg2://{cfg["pgvector"]['user']}:{cfg["pgvector"]['password']}"
            f"@{cfg["pgvector"]['host']}:{cfg["pgvector"].get('port', 5432)}/{cfg["pgvector"]['database']}"
        )

        self.vstore = PGVector(
            collection_name=cfg["pgvector"]["collection"],
            connection=connection_string,
            embeddings=embedding_function,
            use_jsonb=cfg["pgvector"]["use_jsonb"],
        )


    def similarity_search_with_score(self, query: str, k: int = 5, filters: Optional[dict] = None) -> List[Tuple[Document, float]]:
        logger.info(f"Performing vector search for: {query}")
        try:
            results = self.vstore.similarity_search_with_score(query, k, filter=filters or None)
            return results
        except Exception as e:
            logger.error(f"[PGVECTOR QUERY ERROR]: {e}")
            logger.debug(traceback.format_exc())
            return []   
