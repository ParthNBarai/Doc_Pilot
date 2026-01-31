"""
Quick smoke test for FAISS retrieval + product filtering.

Run:
  python3 test_faiss_filters.py
  python3 test_faiss_filters.py --product low_code
  python3 test_faiss_filters.py --product tra --query "..."
"""

import argparse

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="faiss_index")
    parser.add_argument("--query", default="What is this product and how do I get started?")
    parser.add_argument(
        "--product",
        default="all",
        choices=["all", "low_code", "tra", "accessibility", "website_scanner"],
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)

    metadata_filter = None if args.product == "all" else {"product": args.product}
    docs = vs.similarity_search(args.query, k=args.k, filter=metadata_filter)

    print(f"query: {args.query!r}")
    print(f"product: {args.product}")
    print(f"hits: {len(docs)}\n")

    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        print(f"[{i}] product={md.get('product')} source={md.get('source')} chunk_id={md.get('chunk_id')}")
        print(d.page_content[:300].replace("\n", " ").strip())
        print()


if __name__ == "__main__":
    main()


