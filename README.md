## BrowserStack Multi‑Product Docs RAG (FAISS + Ollama)

This is a **tech-heavy RAG demo** that builds a local, multi-product documentation search + chat experience.

### Why this exists (pain point)

When you have **N products** and docs that are **updated frequently**, even internal teams struggle to find the right answer:
- The docs surface area explodes across products.
- New features often get added into **existing** pages/sections (not always a brand-new URL).
- The “correct” guidance is easy to miss when searching manually.

This project uses RAG to: **crawl docs via sitemap → scrape → clean → chunk → embed → store in FAISS → retrieve by product → answer with citations**.

### Repo scope (V1 vs V2)

- **This repo is intentionally kept minimal for the Medium demo (V1)**: multi-product ingestion + product-scoped retrieval.
- **V2 experiments** (e.g., LlamaIndex semantic/hierarchical chunking) are intentionally **not included** here to keep the repo reproducible and focused.

---

## What’s included (files)

- `ingest_products_faiss.py`: E2E ingestion pipeline for multiple products (with checkpointing via `manifest.json`)
- `app.py`: Streamlit chat UI + **product-scoped retrieval**
- `test_faiss_filters.py`: CLI smoke test to verify FAISS retrieval + product filtering
- `requirements.txt`: dependencies

---

## Setup

### 1) Install Ollama

Install Ollama from `https://ollama.com/download` and start it.

### 2) Pull models

Embeddings:

```bash
ollama pull nomic-embed-text
```

LLM for answering (example):

```bash
ollama pull llama3.1
```

### 3) Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Ingestion (build the FAISS index)

This creates a local FAISS index in `faiss_index/` and a checkpoint manifest at `faiss_index/manifest.json`.

### Fresh start

```bash
python3 ingest_products_faiss.py --reset
```

### Recommended (more stable): ingest one product at a time

```bash
python3 ingest_products_faiss.py --product low_code
python3 ingest_products_faiss.py --product tra
python3 ingest_products_faiss.py --product accessibility
python3 ingest_products_faiss.py --product website_scanner
```

### Optional tuning (Ollama stability)

Smaller embed flushes (more stable):

```bash
DOC_BATCH_SIZE=100 python3 ingest_products_faiss.py --product accessibility
```

Limit URLs per run (do it in smaller sessions):

```bash
MAX_URLS=200 python3 ingest_products_faiss.py --product accessibility
```

---

## Test retrieval from CLI (sanity check)

```bash
python3 test_faiss_filters.py --product low_code --query "How do I get started?"
python3 test_faiss_filters.py --product tra --query "What is Test Reporting and Analytics?"
```

---

## Run the UI

```bash
streamlit run app.py
```

Use the **Product** dropdown in the sidebar to scope retrieval and reduce cross-product mixing.

> Important: `app.py` expects a built index in `faiss_index/`. Run ingestion first.

---

## Notes

- `faiss_index/` is intentionally **not committed** (see `.gitignore`) because it’s a large generated artifact.
- If Ollama crashes during ingestion (EOF/500), restart Ollama and re-run the same ingestion command. The manifest prevents reprocessing the same URLs.


