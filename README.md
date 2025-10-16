# RAG Retriever (MCP)

Микросервис на базе **Model Context Protocol (MCP)**, публикующий один инструмент — **`rag.search`** —
для извлечения (и возможного дополнительного ранжирования) фрагментов из векторной базы знаний (RAG‑индекса).
Сервис поддерживает подключаемые **LLM для эмбеддинга**, **LLM для ранжирования** (re‑rank) и **Хранилища векторов**.
Базовая конфигурация задаётся в `config.yaml` и может быть переопределена переменными окружения.

Текущие поддерживаемые конфигурации
- Эмбеддинги: YandexGPT через шлюз LiteLLM (по умолчанию) + YandexGPT (OpenAI порт с исп. LangChain)
- Ражжирование результатов: YandexGPT напрямую или через LiteLLM
- Хранилища векторов: pgvector (по умолчанию), Milvus (требуется протестировать работу)
- Транспорт: HTTP (ASGI) с MCP‑эндпоинтом **`/mcp`**

> Точка запуска: `python -m ragretriever.main` (стартует ASGI‑сервер).

---

## Возможности по настройке

**Подход «env‑сначала»**: переопределение любого ключа YAML переменными окружения вида `RS__SECTION__SUBKEY=value`. (Любые переменные, начинающиеся с `RS__`, могут динамически добавлять новые ключи конфигурации)

Список переменных на базе текущего файла конфигурации приведен для справки внизу страницы
---

## Возможности запуска

Запуск из командной строки (пример):

```bash
python -m ragretriever.main
```

Сборка docker образа (пример):

```bash
docker build --pull --rm -f 'ragretriever/Dockerfile' -t 'ragretriever:latest' 'ragretriever'
```

Запуск образ (пример):

```bash
docker run --rm -it -p 8080:8080 --add-host=host.docker.internal:host-gateway -e "RS__STORAGE__PGVECTOR__HOST=host.docker.internal" -e "RS__RERANK__LITELLM__API_BASE=http://host.docker.internal:4000/v1" -e "RS__EMBEDDING__LITELLM__API_BASE=http://host.docker.internal:4000" -e "RS__RERANK__LITELLM__API_KEY=AQ#######################" -e "RS__RERANK__LITELLM__FOLDER_ID=#######################" -e "RS__EMBEDDING__LITELLM__API_KEY=AQ#######################" -e "RS__EMBEDDING__LITELLM__FOLDER_ID=#######################"  ragretriever:latest /bin/bash
```

## Конфигурация

### YAML‑файл

По умолчанию сервис читает `config.yaml` (в каталоге установленного пакета).

Типовые разделы верхнего уровня:

- `server`: host/port для ASGI‑приложения;
- `embedding`: настройки провайдера эмбеддингов;
- `rerank`: настройки провайдера доранжирования;
- `storage`: конфигурация векторного хранилища (например, `store_type: pgvector|milvus`).

### Переопределение переменными окружения

Любой ключ YAML может быть переопределён переменными окружения. Используются два механизма:

1. Для существующих ключей установите `RS__SECTION__KEY=value`. Значения приводятся к нужным типам (bool/int/float/list) автоматически.
   Примеры:
   ```bash
   RS__SERVER__PORT=9090
   RS__RERANK__PROVIDER=litellm
   ```

2. **Дополнительно**: В текущей версии такие параметры не используются! Любая переменная, начинающаяся с `RS__`, будет разложена по сегментам (`__`) и добавлена в конфиг, даже если такого ключа нет в YAML. (В структуре конфигурации название переменной и название сегментов будут в нижнем регистре)
   Примеры:
   ```bash
   RS__EMBEDDING__LITELLM__API_BASE=https://litellm.internal/v1
   RS__EMBEDDING__LITELLM__API_KEY=sk-...
   ```

### Провайдеры эмбеддингов

Выбираются ключом `embedding.provider`:

- `yandexGPT` — прямой вызов эмбеддингов Yandex (ожидаются поля `embedding.yandexGPT.*`);
- `liteLLM` — OpenAI‑совместимый прокси LiteLLM (ожидаются поля `embedding.liteLLM.*`);
- `langchain_fake` — встроенный фейковый эмбеддер LangChain (для тестов);
- (по умолчанию) — упрощённый «заглушка»‑эмбеддер (если не выбран другой).

### Провайдеры доранжирования

Ключ `rerank.provider` может принимать значения:

- `litellm` — через `ChatOpenAI` на прокси LiteLLM (поля `rerank.liteLLM.*`).  
  Обычно требуются: `api_base` (с включённым суффиксом `/v1`), `api_key`, `model`, `temperature`, `timeout`, а также дополнительные Yandex‑поля, которые передаются в `user`‑payload: `folder_id`, `api_key`, `yandex_model`, `disable_logging`.
- `yandexgpt` — прямой `ChatYandexGPT` (поля `rerank.yandexGPT.*`: `api_key`, `folder_id`, `model`, `temperature`, `timeout`).
- `none` — отключить доранжирование (или не указывать `rerank.provider`). Запросы с `rerank=true` будут приняты, но шаг доранжирования будет пропущен.

### Векторные хранилища

Выбираются через `storage.store_type`:

- `pgvector` — PostgreSQL + расширение pgvector;
- `milvus` — Milvus (необходимо протестировать!).

У каждого адаптера свой набор параметров подключения (DSN/host/port/учётные данные). Укажите их в `config.yaml` или через переменные `RS__STORAGE__...`.

---

## Контракт MCP‑инструмента

ASGI‑приложение публикует MCP‑эндпоинт по пути **`/mcp`**. Доступен один инструмент — `rag.search`:

- **Аргументы**
  - `query` *(string, обязательно)* — пользовательский вопрос на любом языке;
  - `k` *(int, 1..100, по умолчанию 5)* — размер пула кандидатов из векторного индекса;
  - `filters` *(object|null)* — детерминированные фильтры по метаданным (зависят от хранилища) - необходимо протестировать!;
  - `rerank` *(bool, по умолчанию false)* — включить LLM‑доранжирование;
  - `top_n` *(int|null)* — финальная обрезка количества результаов после (возможного) доранжирования.

- **Ответ**
  - `results[]` — список объектов `{text, score, provenance}`, отсортированных по релевантности;
  - `error` — опциональная «мягкая» ошибка (строка), не приводящая к падению сервиса.

Внутри: выполняется векторный поиск по выбранному хранилищу, затем опциональное LLM‑доранжирование и финальная обрезка до `top_n`.

- **Пример результата**
result.json в папке example

## Дополнительные конфигурации внешних инструментов

В папке liteLLM находится конфигурация litellm.yaml и custom handler для поддержки YandexGPT.
Параметры для YandexGPT передаются или через параметры от сервиса, или получаются из ENV переменных.

   ```bash
   export TOOL_FLATTEN_MAX_CHARS=999999
   litellm --config litellm.yaml
   ```

В папке libreChat находится пример конфигурации LibreChat - librechat.yaml, для использования совместно с MCP сервисом и LiteLLM для доступа к модели.
Провайдер liteLLM при вызове из LibreChat берет параметры соединения с YandexGPT из ENV переменных
   
   ```bash
   export YANDEX_API_KEY=AQ*****************************************
   export YANDEX_FOLDER_ID=b1**********************
   export YANDEX_MODEL=yandexgpt-lite/rc
   export YANDEX_DISABLE_LOGGING=false
   ```

## 📝 TODO

- Уменньшить размер собираемого docker образа
- Протестировать корректную работу с Milvus 
- Протестировать корректную работу текстовых фильтров в поиске
- Добавить больше вариантов подключения LLM 
- Добавить unit тесты
- ...

## Список ENV переменных для переопределения. Значения по умолчанию проставлены в config.yaml
RS__SERVER__HOST
RS__SERVER__PORT
RS__STORAGE__STORE_TYPE
RS__STORAGE__MILVUS__URI
RS__STORAGE__MILVUS__COLLECTION
RS__STORAGE__MILVUS__DROP_OLD
RS__STORAGE__MILVUS__AUTO_ID
RS__STORAGE__MILVUS__ALIAS
RS__STORAGE__PGVECTOR__HOST
RS__STORAGE__PGVECTOR__PORT
RS__STORAGE__PGVECTOR__USER
RS__STORAGE__PGVECTOR__PASSWORD
RS__STORAGE__PGVECTOR__DATABASE
RS__STORAGE__PGVECTOR__COLLECTION
RS__STORAGE__PGVECTOR__USE_JSONB
RS__RERANK__PROVIDER
RS__RERANK__TOP_N
RS__RERANK__LITELLM__MODEL
RS__RERANK__LITELLM__API_BASE
RS__RERANK__LITELLM__API_KEY
RS__RERANK__LITELLM__FOLDER_ID
RS__RERANK__LITELLM__YANDEX_MODEL
RS__RERANK__LITELLM__DISABLE_LOGGING
RS__RERANK__LITELLM__TEMPERATURE
RS__RERANK__LITELLM__TIMEOUT
RS__RERANK__YANDEXGPT__API_KEY
RS__RERANK__YANDEXGPT__FOLDER_ID
RS__RERANK__YANDEXGPT__MODEL
RS__RERANK__YANDEXGPT__TEMPERATURE
RS__RERANK__YANDEXGPT__TIMEOUT
RS__EMBEDDING__PROVIDER
RS__EMBEDDING__DIM
RS__EMBEDDING__LITELLM__MODEL
RS__EMBEDDING__LITELLM__API_BASE
RS__EMBEDDING__LITELLM__API_KEY
RS__EMBEDDING__LITELLM__FOLDER_ID
RS__EMBEDDING__YANDEXGPT__API_KEY
RS__EMBEDDING__YANDEXGPT__FOLDER_ID
RS__EMBEDDING__YANDEXGPT__DOC_MODEL_NAME
RS__EMBEDDING__YANDEXGPT__MODEL_VERSION
RS__EMBEDDING__YANDEXGPT__SLEEP_INTERVAL
RS__EMBEDDING__YANDEXGPT__DISABLE_REQUEST_LOGGING
RS__STOP_ON_ERROR
RS__DEBUG