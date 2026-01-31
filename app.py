import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -----------------------------
# Load FAISS + embeddings
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Product filter (optional): helps prevent cross-product mixing when you ingest multiple products.
# These match the `product` metadata stored during ingestion in `ingest_products_faiss.py`.
PRODUCT_OPTIONS = ["all", "low_code", "tra", "accessibility", "website_scanner"]

# Define LLM
llm = OllamaLLM(model="llama3.1")

# -----------------------------
# Prompt (your exact rules)
# -----------------------------
prompt_template = """
You are BrowserStack Support Chatbot. You assist users with technical queries strictly related to BrowserStack documentation.

Goal:
- Answer the user's question by using ONLY the information present in the provided context.
- Be helpful, clear, and explanatory. Prefer step-by-step instructions when appropriate.

Rules (must follow exactly):
1. Use ONLY the content in the context to answer. Do NOT fetch, invent, or hallucinate facts, URLs, or code.
2. If the context contains **code blocks** (fenced with triple backticks ``` or clearly marked code sections), include them verbatim under a **Code Example(s)** section.
   - Do NOT modify, summarize, or reformat code; show it exactly as it appears in the context.
   - If there are multiple code blocks, include the most relevant 1–3 (prioritize those that match the question). If you cannot include the entire long block due to token limits, state that and include the first relevant portion and the Source URL where the full code can be found.
3. If the context contains **no code**, explicitly state:
   "No code snippet is available in the documentation for this query."
   Do NOT generate or infer code.
4. Always include a **Source URL(s)** section at the end with unique documentation links extracted from the context. Format as a short bulleted list, and bold the links.
5. If the context includes video links, include them in a **🎥 Video Tutorial(s)** section (bulleted, bolded).
6. Structure your answer as:
   - Short summary (1–2 sentences)
   - Step-by-step actionable instructions (if applicable)
   - **Code Example(s)** (only if present in context)
   - **🎥 Video Tutorial(s)** (only if present)
   - **Source URL(s)** (always)
7. If the context contains contradictory information across chunks, explicitly say so and list the differing sources.
8. If the context lacks the information needed to answer, say:
   "I’m sorry — the provided documentation does not contain an answer to this question. I can only use the supplied documentation."
   and then list the relevant Source URL(s) shown in the context.
9. Mask or do not reveal any credentials or secrets that appear in the context (replace with placeholders like `YOUR_USERNAME` or `YOUR_ACCESS_KEY` if they appear).
10. Keep the reply professional and user-friendly; avoid apologetic fluff beyond the required error message above.

Context (with source URL, code snippets & video links):
{context}

Question: {input}

Answer (follow the Rules and structure exactly):
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Build chain
document_chain = create_stuff_documents_chain(llm, prompt)

# -----------------------------
# Custom retriever logic
# -----------------------------
def custom_retriever(query, retriever, vectorstore, k=8, metadata_filter=None):
    retrieved_docs = retriever.get_relevant_documents(query)
    wide_docs = vectorstore.similarity_search(query, k=20, filter=metadata_filter)

    boosted_docs = []
    for doc in wide_docs:
        score = 0
        text = doc.page_content.lower()
        if "```" in text or "pipeline {" in text:
            score += 5
        if any(kw in text for kw in query.lower().split()):
            score += 2
        boosted_docs.append((score, doc))

    boosted_docs.sort(key=lambda x: x[0], reverse=True)
    merged_docs = [doc for _, doc in boosted_docs] + [
        d for d in retrieved_docs if d not in [bd for _, bd in boosted_docs]
    ]

    return merged_docs[:k]

def build_context(query, retriever, vectorstore, metadata_filter=None):
    final_docs = custom_retriever(query, retriever, vectorstore, k=8, metadata_filter=metadata_filter)
    enriched_contexts = []
    for doc in final_docs:
        chunk_text = doc.page_content
        source = doc.metadata.get("source", "")
        videos = doc.metadata.get("videos", [])

        video_text = "\n".join([f"🎥 **{v}**" for v in videos]) if videos else ""
        enriched_context = f"{chunk_text}\n\n**Source URL: {source}**"
        if video_text:
            enriched_context += "\n" + video_text

        enriched_contexts.append(enriched_context)

    return "\n\n".join(enriched_contexts)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="BrowserStack Docs Chatbot", page_icon="🤖")
st.title("🤖 BrowserStack Support Chatbot")
st.markdown("Ask any question about BrowserStack docs. If you ingested multiple products, filter below to reduce cross-product mixing.")

with st.sidebar:
    st.header("Retrieval scope")
    selected_product = st.selectbox("Product", options=PRODUCT_OPTIONS, index=0)
    top_k = st.slider("Top-K chunks", min_value=3, max_value=20, value=8)

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Fetching answer..."):
        metadata_filter = None if selected_product == "all" else {"product": selected_product}
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k, "filter": metadata_filter})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        context = build_context(user_query, retriever, vectorstore, metadata_filter=metadata_filter)
        response = retrieval_chain.invoke({"input": user_query, "context": context})
        st.markdown("### 🤖 Answer")
        st.write(response["answer"])
