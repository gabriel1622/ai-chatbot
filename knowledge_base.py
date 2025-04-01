from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path="faiss_index.index"):
    faiss.write_index(index, path)

def load_index(path="faiss_index.index"):
    return faiss.read_index(path)

def save_texts(texts, path="texts.pkl"):
    with open(path, "wb") as f:
        pickle.dump(texts, f)

def load_texts(path="texts.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
