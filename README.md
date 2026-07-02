# SynapseRAG-MCP

## Install

```bash
pip install -r requirements.txt
```

## Start the Dashboard

Local mode — full dashboard, reads `config/settings.yaml`, no masking:

```bash
python scripts/start_dashboard.py
```

## Ingest Files

```bash
python scripts/ingest.py --path pdfs/
python scripts/ingest.py --path pdfs/example.pdf
python scripts/ingest.py --path pdfs/ --collection my_docs
python scripts/ingest.py --path pdfs/ --force
```

## Query

```bash
python scripts/query.py --query "your question"
python scripts/query.py --query "your question" --top-k 5 --collection my_docs
python scripts/query.py --query "your question" --verbose
```

## Evaluate

```bash
python scripts/evaluate.py
```

## Run Tests

```bash
pytest
pytest -m unit
```

## Deployment (Streamlit Community Cloud)

Public mode serves the same dashboard with additional Start (sign-in) and Session pages: per-browser-session workspace isolation, a one-hour data TTL, and masking of local paths and secrets for guest visitors (administrators see unmasked data).

```bash
streamlit run streamlit_app.py
```

1. Push this repository to GitHub and create an app on share.streamlit.io with entrypoint `streamlit_app.py`.
2. Configure the app's Settings → Secrets:

```toml
ADMIN_PASSWORD = "change-me-to-a-long-random-string"
OWNER_LLM_PROVIDER = "openai"
OWNER_LLM_MODEL = "example-model-name"
OWNER_LLM_BASE_URL = "https://api.example-llm.com/v1"
OWNER_LLM_API_KEY = "sk-example-fake-key-000000"

# Optional: real embedding model for admin sessions (guests always use free local hashing)
OWNER_EMBEDDING_PROVIDER = "openai"
OWNER_EMBEDDING_MODEL = "BAAI/bge-m3"
OWNER_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
OWNER_EMBEDDING_API_KEY = "sk-example-fake-key-000000"
OWNER_EMBEDDING_DIMENSIONS = "1024"
```

3. Set a spending limit in your LLM provider's dashboard — the built-in owner quota (30 calls/hour, 200/day) resets when the process restarts.
4. Guest visitors start a session and configure their own LLM, embedding model, and rerank options on the Settings page (held in memory only); retrieval works out of the box with the free local hash embedding.

`.streamlit/secrets.toml` is git-ignored; never commit real credentials.
