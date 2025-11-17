# rag_index.py
# Simple RAG index using sentence-transformers + faiss (CPU)
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError('Please install sentence-transformers: pip install sentence-transformers') from e
try:
    import faiss
except Exception as e:
    raise ImportError('Please install faiss-cpu (or faiss)') from e

MODEL_NAME = 'all-MiniLM-L6-v2'

class SimpleRAG:
    def __init__(self, docs=None):
        self.model = SentenceTransformer(MODEL_NAME)
        self.docs = docs or []
        self.index = None
        self.embeddings = None

    def build_index(self, docs_list):
        self.docs = docs_list
        emb = self.model.encode(self.docs, convert_to_numpy=True)
        d = emb.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(emb)
        self.embeddings = emb

    def retrieve(self, query, k=3):
        if not self.index or len(self.docs) == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for i in I[0]:
            if i < len(self.docs):
                results.append(self.docs[i])
        return results

if __name__ == '__main__':
    docs = [
        "Phishing emails often use urgency and ask to click links.",
        "Check the sender domain closely; hovering reveals the URL.",
        "Use 2FA and strong unique passwords for each account.",
    ]
    rag = SimpleRAG()
    rag.build_index(docs)
    print(rag.retrieve('I got an urgent email asking for password', k=2))
