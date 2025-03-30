import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdfs(pdf_dir, output_file):
    """Loads PDFs, splits text, and saves to file."""
    pdf_dir = os.path.abspath(pdf_dir)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    loaders = [PyPDFLoader(os.path.join(pdf_dir, pdf)) for pdf in os.listdir(pdf_dir) if pdf.endswith(".pdf")]
    pages = [page for loader in loaders for page in loader.load()]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    with open(output_file, "w",encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.page_content + "\n")
