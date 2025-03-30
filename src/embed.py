from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_faiss_index(text_file, model_path):
    """Generates FAISS index from text file."""
    embeddings = HuggingFaceEmbeddings(model_name="load_model/embed_models/all-mpnet-base-v2")
    
    with open(text_file, "r", encoding="utf-8") as f:
        texts = f.readlines()
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(model_path)