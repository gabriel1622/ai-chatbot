import os
import pickle
import faiss
from PyPDF2 import PdfReader
from knowledge_base import embed_texts

def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".csv") or file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def train_and_save_knowledge(file_path, index_path="vector.index", texts_path="texts.pkl"):
    # Extract text
    raw_text = extract_text_from_file(file_path)
    texts = [raw_text]

    # Embed text
    vectors = embed_texts(texts)

    # Create FAISS index and add vectors
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save index
    faiss.write_index(index, index_path)

    # Save corresponding texts
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)


