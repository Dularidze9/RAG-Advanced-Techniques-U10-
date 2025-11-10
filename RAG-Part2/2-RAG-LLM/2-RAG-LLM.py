# hybrid_rag_system.py

import os
import torch
import faiss
import PyPDF2
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

from generation_module import generate_answer


# === Configuration ===
os.environ["HF_TOKEN"] = "hf_XXX"  # Set your HuggingFace access token
PDF_PATH = "sample-document.pdf"

DEVICE = None
EMBED_TOKENIZER = None
EMBED_MODEL = None


# === Embedding Model Setup ===
def initialize_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a sentence embedding model and tokenizer to device.
    """
    global DEVICE, EMBED_MODEL, EMBED_TOKENIZER

    if EMBED_MODEL is not None and EMBED_TOKENIZER is not None:
        return True

    try:
        EMBED_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        EMBED_MODEL = AutoModel.from_pretrained(model_name)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EMBED_MODEL.to(DEVICE)
        print(f"Embedding model '{model_name}' loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f"Error loading embedding model '{model_name}': {e}")
        return False


# === Utility Functions ===
def extract_text_from_pdf(pdf_path):
    """
    Extract plain text from a PDF file.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: file '{pdf_path}' not found.")
        return ""

    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF '{pdf_path}': {e}")

    return text


def split_into_chunks(text, chunk_size=300, overlap=50):
    """
    Divide a long text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def embed_texts(texts):
    """
    Generate dense embeddings for a list of text segments.
    """
    if EMBED_MODEL is None or EMBED_TOKENIZER is None:
        if not initialize_embedding_model():
            raise EnvironmentError("Embedding model not initialized.")

    if not isinstance(texts, list):
        texts = [texts]

    inputs = EMBED_TOKENIZER(texts, padding=True, truncation=True,
                             return_tensors="pt", max_length=512).to(DEVICE)

    with torch.no_grad():
        outputs = EMBED_MODEL(**inputs)

    attention = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    expanded_mask = attention.unsqueeze(-1).expand(token_embeddings.shape).float()
    sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
    sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings.cpu().numpy()


# === Retriever Initialization ===
def setup_retrievers(chunks):
    """
    Initialize BM25 and FAISS retrievers.
    """
    if not chunks:
        print("Error: No text chunks found.")
        return None, None

    # BM25
    tokenized_corpus = [doc.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 retriever initialized.")

    # FAISS
    if EMBED_MODEL is None:
        if not initialize_embedding_model():
            print("Error: Could not initialize FAISS retriever.")
            return bm25, None

    embeddings = embed_texts(chunks)
    vector_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(vector_dim)
    faiss_index.add(embeddings)
    print(f"FAISS index built with {faiss_index.ntotal} vectors.")

    return bm25, faiss_index


# === Hybrid Retrieval ===
def hybrid_search(query, bm25_retriever, faiss_retriever, chunks, k=3, alpha=0.5):
    """
    Combine BM25 (lexical) and FAISS (semantic) retrieval.
    """
    if not chunks:
        return []

    bm25_scores = np.zeros(len(chunks))
    if bm25_retriever:
        try:
            query_tokens = query.lower().split()
            bm25_scores = np.array(bm25_retriever.get_scores(query_tokens))
        except Exception:
            pass

    faiss_scores = {}
    if faiss_retriever and faiss_retriever.ntotal > 0:
        query_emb = embed_texts([query])
        top_k = min(max(k * 2, 10), faiss_retriever.ntotal)
        distances, indices = faiss_retriever.search(query_emb, k=top_k)
        faiss_scores = {idx: 1 / (1 + dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1}

    def normalize(arr):
        if arr.size == 0:
            return arr
        min_v, max_v = np.min(arr), np.max(arr)
        if min_v == max_v:
            return np.full_like(arr, 0.5)
        return (arr - min_v) / (max_v - min_v)

    norm_bm25 = normalize(bm25_scores)
    norm_faiss = np.zeros(len(chunks))

    if faiss_scores:
        temp = np.array([faiss_scores.get(i, 0.0) for i in range(len(chunks))])
        mask = temp > 0
        if np.any(mask):
            norm_faiss[mask] = normalize(temp[mask])

    results = []
    for i in range(len(chunks)):
        score = (1 - alpha) * norm_bm25[i] + alpha * norm_faiss[i]
        results.append((score, i))

    results.sort(key=lambda x: x[0], reverse=True)
    return [{"index": idx, "text": chunks[idx], "score": sc} for sc, idx in results[:k]]


# === End-to-End Test ===
def demo_rag_pipeline():
    """
    End-to-end test for PDF → chunking → hybrid retrieval → generation.
    """
    print("\n--- Starting RAG Demo ---")

    if not initialize_embedding_model():
        print("Error: embedding model failed to load.")
        return

    text = extract_text_from_pdf(PDF_PATH)
    if not text.strip():
        print("No text extracted from PDF.")
        return

    chunks = split_into_chunks(text, chunk_size=300, overlap=70)
    print(f"Generated {len(chunks)} text chunks from PDF.")

    bm25, faiss_idx = setup_retrievers(chunks)
    if bm25 is None and (faiss_idx is None or faiss_idx.ntotal == 0):
        print("Retrievers not initialized. Aborting test.")
        return

    query = "Explain the concept of AI Agent self-improvement."
    print(f"\nPerforming hybrid search for: '{query}'")

    retrieved = hybrid_search(query, bm25, faiss_idx, chunks, k=3, alpha=0.6)
    if not retrieved:
        print("No documents retrieved.")
        return

    print("\nRetrieved Chunks:")
    for item in retrieved:
        print(f"  Index: {item['index']}, Score: {item['score']:.4f}")

    try:
        answer = generate_answer(query=query, retrieved_docs=retrieved[0]['text'], max_new_tokens=300)
        print("\nGenerated Answer:\n", answer)
    except Exception as e:
        print(f"Error generating answer: {e}")

    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    demo_rag_pipeline()
