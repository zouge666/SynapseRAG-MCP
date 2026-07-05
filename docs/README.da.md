# SynapseRAG-MCP

[English](../README.md) | [中文](README.zh-CN.md) | [Dansk](README.da.md) | [Deutsch](README.de.md)

SynapseRAG-MCP er et modulært RAG-system (Retrieval-Augmented Generation): hvert led i pipelinen — splitting, embeddings, hybrid dense + sparse-søgning med RRF-fusion, reranking og svargenerering — er en udskiftelig komponent med udskiftelige udbydere. Projektet indeholder en MCP-server, så agenter kan forespørge vidensbasen direkte, samt et Streamlit-dashboard til indlæsning, traces, evaluering og live-konfiguration. Kør det lokalt på din egen stack, eller udrul det offentligt med isolation pr. session og en times data-TTL.

**[Prøv live-demoen →](https://synapserag-mcp-server.streamlit.app)**

![SynapseRAG MCP dashboard](screenshot.jpg)

## Installér

```bash
pip install -r requirements.txt
```

## Start dashboardet

Lokal tilstand — fuldt dashboard, læser `config/settings.yaml`, ingen maskering:

```bash
python scripts/start_dashboard.py
```

## Indlæs filer

```bash
python scripts/ingest.py --path pdfs/
python scripts/ingest.py --path pdfs/example.pdf
python scripts/ingest.py --path pdfs/ --collection my_docs
python scripts/ingest.py --path pdfs/ --force
```

## Forespørgsel

```bash
python scripts/query.py --query "your question"
python scripts/query.py --query "your question" --top-k 5 --collection my_docs
python scripts/query.py --query "your question" --verbose
```

## Evaluér

```bash
python scripts/evaluate.py
```

## Kør tests

```bash
pytest
pytest -m unit
```

## Udrulning (Streamlit Community Cloud)

Public-tilstand serverer det samme dashboard med ekstra Start- (log ind) og Session-sider: isolation af arbejdsområde pr. browsersession, en times data-TTL samt maskering af lokale stier og hemmeligheder for gæster (administratorer ser umaskerede data).

```bash
streamlit run streamlit_app.py
```

1. Push dette repository til GitHub, og opret en app på share.streamlit.io med entrypoint `streamlit_app.py`.
2. Konfigurér appens Settings → Secrets:

```toml
ADMIN_PASSWORD = "change-me-to-a-long-random-string"
OWNER_LLM_PROVIDER = "openai"
OWNER_LLM_MODEL = "example-model-name"
OWNER_LLM_BASE_URL = "https://api.example-llm.com/v1"
OWNER_LLM_API_KEY = "sk-example-fake-key-000000"

# Valgfrit: rigtig embedding-model til admin-sessioner (gæster bruger altid gratis lokal hashing)
OWNER_EMBEDDING_PROVIDER = "openai"
OWNER_EMBEDDING_MODEL = "BAAI/bge-m3"
OWNER_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
OWNER_EMBEDDING_API_KEY = "sk-example-fake-key-000000"
OWNER_EMBEDDING_DIMENSIONS = "1024"
```

3. Sæt en forbrugsgrænse i din LLM-udbyders dashboard — den indbyggede ejer-kvote (30 kald/time, 200/dag) nulstilles, når processen genstarter.
4. Gæster starter en session og konfigurerer deres egen LLM, embedding-model og rerank-valg på Settings-siden (opbevares kun i hukommelsen); søgning fungerer direkte med den gratis lokale hash-embedding.

`.streamlit/secrets.toml` er git-ignoreret; commit aldrig rigtige legitimationsoplysninger.

## Licens

[GNU AGPL-3.0](../LICENSE) © 2026 Theo Zou
