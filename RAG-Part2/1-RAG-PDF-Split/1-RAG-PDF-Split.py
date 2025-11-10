# hybrid_retrieval_module.py
"""
Hybrid Retrieval System using Dense (FAISS) and Sparse (BM25) Search
-------------------------------------------------------------------
This module extracts text from a PDF, splits it into overlapping chunks,
builds both dense and sparse indices, and performs hybrid retrieval
for a given user query.

Author: (Your Name)
"""

import os
import torch
import faiss
import numpy as np
import PyPDF2
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi


# === Configurations ===
HF_TOKEN = os.getenv("HF_TOKEN", "hf_xxx")
PDF_PATH = "example.pdf"

# Global state (models initialized once)
DEVICE = None
TOKENIZER = None
EMBED_MODEL = None


# === Model Setup ===
def initialize_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load a dense embedding model and tokenizer."""
    global DEVICE, TOKENIZER, EMBED_MODEL
    if TOKENIZER and EMBED_MODEL:
        return True

    try:
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        EMBED_MODEL = AutoModel.from_pretrained(model_name)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EMBED_MODEL.to(DEVICE)
        print(f" Embedding model '{model_name}' loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f" Failed to load embedding model: {e}")
        return False


# === Embedding Generation ===
def encode_texts(text_list):
    """Generate mean-pooled dense embeddings for a list of texts."""
    if not TOKENIZER or not EMBED_MODEL:
        if not initialize_embedding_model():
            raise RuntimeError("Embedding model not initialized.")

    if not isinstance(text_list, list):
        text_list = [text_list]

    inputs = TOKENIZER(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = EMBED_MODEL(**inputs)

    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
    summed = torch.sum(embeddings * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts
    return mean_pooled.cpu().numpy()


# === PDF Utilities ===
def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    if not os.path.exists(pdf_path):
        print(f" PDF file not found: {pdf_path}")
        return ""

    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f" Error reading PDF: {e}")
    return text


def chunk_text(text, chunk_size=150, overlap=50):
    """Split long text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


# === Retriever Initialization ===
def initialize_retrievers(chunks):
    """Create BM25 and FAISS indices for the given chunks."""
    if not chunks:
        print(" No chunks provided for retriever initialization.")
        return None, None

    # BM25 setup (simple token split)
    tokenized_corpus = [doc.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print(" BM25 retriever initialized.")

    # FAISS setup
    if not EMBED_MODEL and not initialize_embedding_model():
        print(" Embedding model unavailable. Only BM25 will be used.")
        return bm25, None

    embeddings = encode_texts(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f" FAISS index built with {index.ntotal} vectors.")
    return bm25, index


# === Hybrid Retrieval ===
def hybrid_search(query, bm25, faiss_index, chunks, top_k=3, alpha=0.5):
    """Perform hybrid retrieval combining BM25 and FAISS scores."""
    if not chunks:
        print(" No chunks available for retrieval.")
        return []

    use_bm25 = bm25 is not None
    use_faiss = faiss_index is not None and faiss_index.ntotal > 0

    if not use_bm25 and not use_faiss:
        print(" Both retrievers unavailable.")
        return []

    # Sparse retrieval
    bm25_scores = np.zeros(len(chunks))
    if use_bm25:
        tokens = query.lower().split()
        try:
            bm25_scores = np.array(bm25.get_scores(tokens))
        except Exception as e:
            print(f" BM25 retrieval failed: {e}")

    # Dense retrieval
    faiss_scores = {}
    if use_faiss:
        q_embed = encode_texts([query])
        num_candidates = min(max(top_k * 2, 10), faiss_index.ntotal)
        distances, indices = faiss_index.search(q_embed, k=num_candidates)
        faiss_scores = {idx: 1 / (1 + dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1}

    # Normalize
    def normalize(arr):
        arr = np.array(arr)
        if arr.size == 0:
            return np.zeros_like(arr)
        min_v, max_v = np.min(arr), np.max(arr)
        return np.full_like(arr, 0.5) if max_v == min_v else (arr - min_v) / (max_v - min_v)

    norm_bm25 = normalize(bm25_scores)
    norm_faiss = np.zeros(len(chunks))
    for idx, val in faiss_scores.items():
        norm_faiss[idx] = val
    norm_faiss = normalize(norm_faiss)

    # Weighted hybrid
    scores = (1 - alpha) * norm_bm25 + alpha * norm_faiss
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = [{"chunk_index": i, "text": chunks[i], "score": s} for i, s in ranked]

    return results


# === Test Pipeline ===
def run_retrieval_demo():
    """End-to-end retrieval demonstration."""
    print("\n=== Running Hybrid Retrieval Demo ===")

    if not initialize_embedding_model():
        print(" Embedding model setup failed. Exiting.")
        return []

    text = extract_text_from_pdf(PDF_PATH)
    if not text.strip():
        print(f" No text extracted from '{PDF_PATH}'. Exiting.")
        return []

    chunks = chunk_text(text, chunk_size=150, overlap=50)
    print(f"✅ Created {len(chunks)} text chunks.")

    bm25, faiss_index = initialize_retrievers(chunks)
    query = "What is the self-improvement ability of an AI Agent?"

    print(f"\n🔍 Running hybrid search for query: '{query}'\n")
    results = hybrid_search(query, bm25, faiss_index, chunks, top_k=3, alpha=0.5)

    if results:
        for r in results:
            print(f"Chunk #{r['chunk_index']} | Score: {r['score']:.4f}")
            print(f"Text: {r['text'][:150]}...\n")
    else:
        print(" No results found.")
    print("=== Demo Complete ===")
    return results


if __name__ == "__main__":
    run_retrieval_demo()
