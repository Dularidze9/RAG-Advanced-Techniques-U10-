# RAG Techniques (რაგის ტექნიკები)

#### **Phase 1 — Core RAG Techniques**

1. **`RAG_Reformulate.py`** — Implements query reformulation to improve retrieval quality.
2. **`RAG_Hybrid_Retrieval.py`** — Combines dense and sparse retrieval methods (FAISS + BM25) for hybrid search.
3. **`RAG_Re-ranking.py`** — Re-ranks retrieved documents using semantic similarity to optimize context selection.

#### **Phase 2 — End-to-End RAG Pipeline**

4. **`RAG-Part2/1-RAG-PDF-Split`** — Splits and preprocesses PDF content into overlapping chunks for retrieval.
5. **`RAG-Part2/2-RAG-LLM`** — Uses the retrieved text as context and generates responses with the **Qwen2-0.5B-Instruct** model.
6. **`RAG-Part2/3-RAG-Eval`** — Evaluates the RAG system using metrics such as **hit rate**, **retrieval accuracy**, and **answer relevance**.

---
Steps **1–3** cover the **fundamental RAG techniques**, while **4–6** build a complete **document-based RAG workflow** — from chunking and retrieval to generation and evaluation.

