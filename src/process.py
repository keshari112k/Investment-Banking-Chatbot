import os
import json
from src.preprocessing import process_pdfs
from src.embed import create_faiss_index
from src.rag_process import load_rag_pipeline

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Join with the subdirectory
pdf_dir = os.path.join(base_dir, "input_data")
processed_text = os.path.join(base_dir, "processed_data", "faq_chunks.txt")
vector_path = os.path.join(base_dir, "vectorstore")

    
# Run pipeline
# process_pdfs(pdf_dir, processed_text)
# create_faiss_index(processed_text, vector_path)
qa_pipeline = load_rag_pipeline()