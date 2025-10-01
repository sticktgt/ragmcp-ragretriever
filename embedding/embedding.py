from typing import List, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .embedding_factory import get_embedder
from ..utils.logger import get_logger
import traceback

logger = get_logger()

def get_embedding_function(cfg: dict) -> Embeddings:
    embedder = get_embedder(cfg)
    return embedder.get_embedding_function()

def embed_documents(docs: Union[Document, List[Document]], cfg: dict) -> Tuple[List[dict], str | None]:
    if isinstance(docs, Document):
        docs = [docs]
    try:
        embedder = get_embedder(cfg)
        return embedder.embed_documents(docs), None
    except Exception as e:
        error_message = f"Embedding error: {e}"
        logger.error(error_message)
        logger.debug(traceback.format_exc())
        return [], error_message
