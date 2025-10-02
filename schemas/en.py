from __future__ import annotations
from typing import Any, Dict, List, Optional, Annotated
from pydantic import BaseModel, Field, ConfigDict

TOOL_DESCRIPTION = (
    "Retrieve relevant knowledge snippets from the RAG index. "
    "Use `filters` for deterministic store-side filtering; set `rerank=true` to apply an LLM reranker; "
    "use `top_n` to trim the final results after rerank. "
    "Always ground answers by citing sources from `provenance` (prefer `uri`, else `original_name`)."
)

class RagSearchArgs(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Fast ANN search (no rerank)",
                    "value": {"query": "keda autoscaling", "k": 5}
                },
                {
                    "summary": "High-precision search with LLM rerank + trim",
                    "value": {
                        "query": "data catalog key metrics",
                        "k": 12,
                        "filters": {"source": "documents", "lang": "en"},
                        "rerank": True,
                        "top_n": 5
                    }
                }
            ]
        }
    )

    query: str = Field(
        ...,
        description="User query to search for. Provide a concise, information-seeking question or keywords."
    )
    k: Annotated[int, Field(ge=1, le=100)] = Field(
        5,
        description="Candidate pool size from the vector index before any reranking (1–100). Higher = better recall, more latency."
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Deterministic store-side metadata filter applied inside the vector database "
            "before retrieval, e.g. {\"source\":\"documents\", \"lang\":\"en\"}."
        )
    )
    rerank: bool = Field(
        False,
        description=(
            "If true, apply server-configured LLM reranking to the top-k candidates. "
            "This trades a bit more latency/tokens for better precision."
        )
    )
    top_n: Optional[Annotated[int, Field(ge=1)]] = Field(
        None,
        description=(
            "Final trim *after* optional reranking. If omitted, returns all retrieved items "
            "(up to k, or fewer if the store returns less)."
        )
    )

class SearchHit(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "summary": "Typical hit",
                "value": {
                    "text": "…chunk text…",
                    "score": 0.47,
                    "provenance": {
                        "original_name": "датакаталог_temp.pptx",
                        "uri": "https://example/doc/123#p=4",
                        "page_number": 4
                    }
                }
            }]
        }
    )

    text: str = Field(description="The retrieved chunk text (plain text).")
    score: Annotated[float, Field(ge=0.0)] = Field(
        description=(
            "Similarity score from the vector store (higher = closer match). "
            "Not a probability; the absolute scale depends on the embedding/model."
        )
    )
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Source metadata used for citation and grounding. Common keys:\n"
            "- original_name: original file/resource name\n"
            "- title: human-readable title (if available)\n"
            "- uri: canonical source link (prefer for citations when present)\n"
            "- page_number, page_name: location hints within the source\n"
            "- element_id, chunk_id: internal identifiers\n"
            "- filetype, size_bytes, languages, created_at, hashcode, source_connector\n"
            "- text_as_html: HTML rendition for tables or rich text (optional)"
        )
    )

class RagSearchOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "summary": "Successful response",
                "value": {
                    "results": [
                        {
                            "text": "…",
                            "score": 0.49,
                            "provenance": {"original_name": "doc.docx"}
                        }
                    ],
                    "error": None
                }
            }]
        }
    )

    results: List[SearchHit] = Field(
        description="Ranked results after optional rerank and final trimming (top_n)."
    )
    error: Optional[str] = Field(
        default=None,
        description="Present only if the server encountered a recoverable error (e.g., store unavailable)."
    )
