from .milvus_store import MilvusStore
from .pgvector_store import PGVectorStore
from .base import VectorStoreBase

def get_vector_store(cfg: dict, embedding_function) -> VectorStoreBase:
    store_type = cfg.get("store_type", "milvus").lower()
    if store_type == "pgvector":
        return PGVectorStore(cfg, embedding_function)
    return MilvusStore(cfg, embedding_function)
