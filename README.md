# BasketballGPT

Streamlit chat app for asking natural-language questions against local PostgreSQL basketball analytics schemas.

The app is designed around copied PostgreSQL data from the Kubernetes/cluster database. It supports the configured schemas:

- `bronze`
- `silver`
- `gold`

## Components

- `app.py`: Streamlit UI, schema browser, result summaries, and chart rendering.
- `query_engine.py`: LLM text-to-SQL, SQL validation, query execution, deterministic chart helpers, and answer formatting.
- `lightweight_rag.py`: Small local RAG-style retriever that injects relevant SQL examples into the LLM prompt.
- `knowledge/sql_examples.json`: Curated question, SQL, and documentation examples used by the retriever.
- `db.py`: PostgreSQL connection helpers and schema search-path configuration.
- `scripts/copy_bronze_to_local.*`: Copy one source schema into local Postgres. Despite the historical filename, set `SOURCE_DB_SCHEMA` to copy `bronze`, `silver`, or `gold`.
- `scripts/verify_schema_counts.py`: Compare source and local row counts for selected schemas.
- `docker-compose.yml`: Runs local PostgreSQL and the Streamlit app.

## Setup

```bash
cp .env.example .env
```

Default local PostgreSQL settings:

```bash
POSTGRES_DB=basketballgpt
POSTGRES_USER=basketballgpt
POSTGRES_PASSWORD=basketballgpt
POSTGRES_PORT=55432
DATABASE_URL=postgresql://basketballgpt:basketballgpt@localhost:55432/basketballgpt
NBA_DB_SCHEMAS=bronze,silver,gold
```

Default LLM settings use Ollama cloud:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.com/api
OLLAMA_MODEL=qwen3-coder:480b-cloud
OLLAMA_API_KEY=your_ollama_key
```

For local Ollama:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:latest
```

For Gemini:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
```

For an OpenAI-compatible endpoint (e.g. an in-cluster LiteLLM proxy):

```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://litellm.litellm.svc.cluster.local:4000/v1
OPENAI_MODEL=qwen3.6-35b-a3b-coder
OPENAI_API_KEY=your_litellm_key
```

## Lightweight SQL Memory

BasketballGPT includes a small Vanna-like retrieval layer without adding Vanna or a vector database. It loads curated examples from:

```text
knowledge/sql_examples.json
```

For each user question, the app retrieves the most relevant examples and adds them to the SQL-generation prompt. This is RAG-style prompt augmentation; it does not fine-tune Ollama or Gemini.

Configuration:

```bash
SQL_RAG_ENABLED=true
SQL_RAG_MAX_EXAMPLES=4
SQL_RAG_KNOWLEDGE_PATH=./knowledge/sql_examples.json
SQL_RAG_REJECTED_PATH=./knowledge/rejected_sql_examples.json
```

Every answer has **Save good** and **Mark bad** buttons in the UI. "Save good" appends the question and generated SQL to `sql_examples.json`. "Mark bad" asks for a reason first, then appends to `rejected_sql_examples.json` as an anti-pattern the model is told to avoid - a good reason (what was wrong, what the correct approach is) matters much more than the SQL itself, since it is what actually reaches the model on a similar future question.

This is the primary way the knowledge base grows day to day - editing the JSON files by hand is only needed for bulk changes. In the cluster deployment, `SQL_RAG_KNOWLEDGE_PATH`/`SQL_RAG_REJECTED_PATH` point at a PersistentVolumeClaim mounted at `/data/knowledge` (not `/app/knowledge`, which holds the image's seed copy), so feedback saved through the UI survives pod restarts and redeploys.

## Run

```bash
docker compose up --build app
```

Open:

```text
http://localhost:8501
```

## Copy Schemas From Cluster Postgres

Start your Kubernetes Postgres port-forward first, then run one schema at a time.

PowerShell:

```powershell
$env:SOURCE_DATABASE_URL = "postgresql://user:password@127.0.0.1:5432/database"
$env:SOURCE_DB_SCHEMA = "silver"
.\scripts\copy_bronze_to_local.ps1
```

Bash:

```bash
export SOURCE_DATABASE_URL='postgresql://user:password@127.0.0.1:5432/database'
export SOURCE_DB_SCHEMA=silver
./scripts/copy_bronze_to_local.sh
```

Repeat with `SOURCE_DB_SCHEMA=bronze`, `silver`, or `gold` as needed. The script replaces the matching local schema.

## Verify Local Counts

```bash
docker compose exec -T \
  -e SOURCE_DATABASE_URL='postgresql://user:password@host.docker.internal:5432/database' \
  -e SCHEMAS='bronze,silver,gold' \
  app python scripts/verify_schema_counts.py
```
