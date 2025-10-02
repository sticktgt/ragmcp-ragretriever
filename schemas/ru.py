from __future__ import annotations
from typing import Any, Dict, List, Optional, Annotated
from pydantic import BaseModel, Field, ConfigDict

TOOL_DESCRIPTION = (
    "Извлекает релевантные фрагменты знаний из RAG-индекса. "
    "Используйте `filters` для детерминированной фильтрации на стороне хранилища; укажите `rerank=true`, чтобы применить LLM-переранжировку; "
    "параметр `top_n` обрезает итог после переранжировки. "
    "Всегда обосновывайте ответы, цитируя источники из `provenance` (в приоритете `uri`, иначе `original_name`)."
)

class RagSearchArgs(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Быстрый поиск без переранжировки",
                    "value": {"query": "модель данных", "k": 5}
                },
                {
                    "summary": "Точный поиск с LLM-переранжировкой и обрезкой",
                    "value": {
                        "query": "ключевые показатели данных",
                        "k": 12,
                        "filters": {"source": "documents", "lang": "ru"},
                        "rerank": True,
                        "top_n": 5
                    }
                }
            ]
        }
    )

    query: str = Field(
        ..., description="Пользовательский запрос для поиска. Краткий вопрос или ключевые слова."
    )
    k: Annotated[int, Field(ge=1, le=100)] = Field(
        5,
        description="Размер пула кандидатов из векторного индекса до переранжировки (1–100). Больше — выше полнота, но выше задержка."
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Детерминированный фильтр по метаданным, применяемый внутри векторного хранилища "
            "до извлечения, например {\"source\":\"documents\", \"lang\":\"ru\"}."
        )
    )
    rerank: bool = Field(
        False,
        description=(
            "Если true — применить LLM-переранжировку к top-k кандидатам (по конфигурации сервера). "
            "Даёт лучшую точность при небольшой дополнительной задержке/токенах."
        )
    )
    top_n: Optional[Annotated[int, Field(ge=1)]] = Field(
        None,
        description=(
            "Финальная обрезка *после* возможной переранжировки. Если не задано — возвращаются все найденные элементы "
            "(до k, либо меньше, если хранилище вернуло меньше)."
        )
    )

class SearchHit(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "summary": "Типичный результат",
                "value": {
                    "text": "…фрагмент…",
                    "score": 0.52,
                    "provenance": {
                        "original_name": "датакаталог_temp.pptx",
                        "page_number": 4
                    }
                }
            }]
        }
    )

    text: str = Field(description="Найденный текстовый фрагмент (plain text).")
    score: Annotated[float, Field(ge=0.0)] = Field(
        description=(
            "Оценка близости из векторного хранилища (выше — ближе). "
            "Это не вероятность; абсолютная шкала зависит от модели эмбеддингов."
        )
    )
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Метаданные источника для обоснования и цитирования. Типичные ключи:\n"
            "- original_name: исходное имя файла/ресурса\n"
            "- title: человекочитаемое название (если есть)\n"
            "- uri: каноническая ссылка на источник (предпочтительна для цитирования)\n"
            "- page_number, page_name: позиция внутри источника\n"
            "- element_id, chunk_id: внутренние идентификаторы\n"
            "- filetype, size_bytes, languages, created_at, hashcode, source_connector\n"
            "- text_as_html: HTML-представление таблиц/форматированного текста (опционально)"
        )
    )

class RagSearchOutput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "summary": "Успешный ответ",
                "value": {
                    "results": [
                        {"text": "…", "score": 0.49, "provenance": {"original_name": "doc.docx"}}
                    ],
                    "error": None
                }
            }]
        }
    )

    results: List[SearchHit] = Field(
        description="Отсортированные результаты после (возможной) переранжировки и финальной обрезки (top_n)."
    )
    error: Optional[str] = Field(
        default=None,
        description="Заполняется только при восстановимой ошибке на сервере (например, недоступно хранилище)."
    )
