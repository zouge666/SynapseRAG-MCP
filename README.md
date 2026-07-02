# SynapseRAG-MCP

## Install

```bash
pip install -r requirements.txt
```

## Start the Dashboard

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
