"""
E2E ingestion pipeline (LangChain + FAISS) for multiple BrowserStack products.

Goals:
- Ingest multiple product doc sections into ONE FAISS index directory (default: faiss_index/)
- Avoid re-doing work across runs (manifest checkpointing by URL)
- Batch embeddings + persist frequently to reduce blast radius if Ollama crashes

Usage examples:
  # Start fresh (deletes index dir + manifest)
  python3 ingest_products_faiss.py --reset

  # Ingest one product at a time (recommended to reduce Ollama crashes)
  python3 ingest_products_faiss.py --product low_code
  python3 ingest_products_faiss.py --product tra
  python3 ingest_products_faiss.py --product accessibility
  python3 ingest_products_faiss.py --product website_scanner

  # Ingest all products (will checkpoint + can resume)
  python3 ingest_products_faiss.py --product all

Tuning:
  DOC_BATCH_SIZE=200 python3 ingest_products_faiss.py --product low_code
  MAX_URLS=50 python3 ingest_products_faiss.py --product all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm


SITEMAP_URL = os.environ.get("SITEMAP_URL", "https://www.browserstack.com/sitemap.xml")
BASE_URL = os.environ.get("BASE_URL", "https://www.browserstack.com")
INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "faiss_index")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

MIN_WORDS = int(os.environ.get("MIN_WORDS", "30"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# How many chunk-documents to embed+add per flush
DOC_BATCH_SIZE = int(os.environ.get("DOC_BATCH_SIZE", "200"))
# Small sleep after each flush to reduce pressure on Ollama
SLEEP_AFTER_FLUSH_SEC = float(os.environ.get("SLEEP_AFTER_FLUSH_SEC", "1.0"))

# Hard cap URLs per run (helps you run “a little at a time” safely)
MAX_URLS = int(os.environ.get("MAX_URLS", "0"))  # 0 = no cap


PRODUCTS: Dict[str, Dict[str, str]] = {
    "low_code": {
        "name": "Low Code Automation",
        "filter": "/docs/low-code-automation/",
    },
    "tra": {
        "name": "Test Reporting & Analytics",
        "filter": "/docs/test-reporting-and-analytics/",
    },
    "accessibility": {
        "name": "Accessibility",
        "filter": "/docs/accessibility/",
    },
    "website_scanner": {
        "name": "Website Scanner",
        "filter": "/docs/website-scanner/",
    },
}


def fetch_urls_from_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url, timeout=60)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.content)
    urls = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return [u for u in urls if u]


def clean_text(raw_text: str) -> str:
    noise_patterns = [
        "Table of Contents",
        "No Result Found",
        "Ask the community",
        "Submit feedback",
        "© BrowserStack",
        "All rights reserved",
    ]
    for pattern in noise_patterns:
        raw_text = raw_text.replace(pattern, "")
    return raw_text.strip()


def extract_videos(soup: BeautifulSoup, base_url: str) -> List[str]:
    video_links: List[str] = []
    main_content = soup.select_one("#main-content > div.docs--content-wrapper.docs--content-wrapper--v2")
    if main_content:
        for video in main_content.find_all("video"):
            source_tag = video.find("source", {"type": "video/mp4"})
            if source_tag:
                src = source_tag.get("src")
                if not src:
                    continue
                if src.startswith("/"):
                    src = base_url + src
                video_links.append(src)
    return video_links


def load_manifest() -> Dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {"processed_urls": {}, "stats": {}}


def save_manifest(manifest: Dict) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def ollama_healthcheck(embeddings: OllamaEmbeddings) -> bool:
    try:
        v = embeddings.embed_query("test")
        return bool(v) and len(v) > 0
    except Exception:
        return False


def load_or_create_vectorstore(embeddings: OllamaEmbeddings) -> Optional[FAISS]:
    try:
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️  Could not load existing FAISS index, will create a new one. Reason: {e}")
    return None


def flush_docs(
    vectorstore: Optional[FAISS],
    embeddings: OllamaEmbeddings,
    docs: List[Document],
) -> FAISS:
    if not docs:
        if vectorstore is None:
            raise ValueError("flush_docs called with empty docs and no existing vectorstore")
        return vectorstore

    if not ollama_healthcheck(embeddings):
        raise RuntimeError(
            "Ollama is not responding to embedding requests. Restart it (e.g. `pkill ollama && ollama serve`) and rerun."
        )

    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    else:
        vectorstore.add_documents(docs)

    vectorstore.save_local(INDEX_DIR)
    time.sleep(SLEEP_AFTER_FLUSH_SEC)
    return vectorstore


def scrape_url_to_docs(
    url: str,
    product_key: str,
    splitter: RecursiveCharacterTextSplitter,
) -> Tuple[List[Document], Optional[str]]:
    try:
        r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        main_content = soup.find("main") or soup.find("article") or soup
        raw_text = main_content.get_text(separator="\n")
        cleaned_text = clean_text(raw_text)
        if not cleaned_text or len(cleaned_text) < 200:
            return [], None

        video_links = extract_videos(soup, BASE_URL)
        chunks = splitter.split_text(cleaned_text)

        docs: List[Document] = []
        chunk_id = 0
        for chunk in chunks:
            if len(chunk.split()) < MIN_WORDS:
                continue
            docs.append(
                Document(
                    page_content=f"{chunk}\n\nSource URL: {url}",
                    metadata={
                        "source": url,
                        "chunk_id": chunk_id,
                        "videos": video_links,
                        "product": product_key,
                    },
                )
            )
            chunk_id += 1
        return docs, None
    except Exception as e:
        return [], str(e)


def ingest_product(
    product_key: str,
    urls: List[str],
    embeddings: OllamaEmbeddings,
    vectorstore: Optional[FAISS],
    manifest: Dict,
) -> FAISS:
    product = PRODUCTS[product_key]
    filt = product["filter"]

    processed_urls_by_product = manifest.setdefault("processed_urls", {}).setdefault(product_key, [])
    processed_set = set(processed_urls_by_product)

    product_urls = [u for u in urls if filt in u]
    if MAX_URLS > 0:
        product_urls = product_urls[:MAX_URLS]

    print(f"\n## {product['name']} ({product_key})")
    print(f"- filter: {filt!r}")
    print(f"- urls matched: {len(product_urls)}")
    print(f"- already processed: {len(processed_set)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    pending_docs: List[Document] = []
    new_processed: List[str] = []

    for url in tqdm(product_urls, desc=f"Ingesting {product_key}", unit="url"):
        if url in processed_set:
            continue

        docs, err = scrape_url_to_docs(url, product_key=product_key, splitter=splitter)
        if err:
            tqdm.write(f"⚠️  skipped {url}: {err}")
            continue

        pending_docs.extend(docs)
        new_processed.append(url)

        if len(pending_docs) >= DOC_BATCH_SIZE:
            # Flush batch to FAISS + persist + checkpoint manifest
            vectorstore = flush_docs(vectorstore, embeddings, pending_docs)
            pending_docs = []

            processed_urls_by_product.extend(new_processed)
            new_processed = []
            save_manifest(manifest)

    # Flush remainder
    if pending_docs:
        vectorstore = flush_docs(vectorstore, embeddings, pending_docs)
        processed_urls_by_product.extend(new_processed)
        save_manifest(manifest)

    # Stats
    stats = manifest.setdefault("stats", {})
    stats[product_key] = {
        "processed_urls": len(set(processed_urls_by_product)),
        "last_run_ts": int(time.time()),
    }
    save_manifest(manifest)

    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        default="all",
        choices=["all", *PRODUCTS.keys()],
        help="Which product section to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing FAISS index dir + manifest before ingesting",
    )
    args = parser.parse_args()

    if args.reset and os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        print(f"🧹 Deleted {INDEX_DIR!r}")

    os.makedirs(INDEX_DIR, exist_ok=True)
    manifest = load_manifest()

    print(f"Fetching sitemap: {SITEMAP_URL}")
    all_urls = fetch_urls_from_sitemap(SITEMAP_URL)
    print(f"✅ sitemap urls found: {len(all_urls)}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = load_or_create_vectorstore(embeddings)
    if vectorstore is None:
        print("ℹ️  No existing FAISS index found. Will create a new one on first flush.")
    else:
        print(f"✅ Loaded existing FAISS index from {INDEX_DIR!r}")

    product_keys = list(PRODUCTS.keys()) if args.product == "all" else [args.product]
    for pk in product_keys:
        vectorstore = ingest_product(pk, all_urls, embeddings, vectorstore, manifest)

    print("\n✅ Done.")
    print(f"- index dir: {INDEX_DIR}")
    print(f"- manifest: {MANIFEST_PATH}")
    print("If Ollama crashes with EOF/500, restart it and rerun the same command — it will resume via manifest.")


if __name__ == "__main__":
    main()


