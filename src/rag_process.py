import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import T5ForConditionalGeneration, pipeline, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Get the base directory (go up one level from 'src')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def load_rag_pipeline():
    """Loads FAISS and FLAN-T5 for RAG inference."""
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
   # Define the relative path to the embedding model
    embed_model_path = os.path.join(base_dir, "load_model", "embed_models", "all-mpnet-base-v2")
    # Load from local folder
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_path)
    query_embedding = embeddings.embed_query("What is machine learning?")
    print(query_embedding[:5])  # Check if embeddings are generated

    vectorstore = FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)
   
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    flant5_model_path = os.path.join(base_dir, "load_model", "flanT5_model")
    tokenizer = AutoTokenizer.from_pretrained(flant5_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(flant5_model_path)

    # ✅ Add pipeline here
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024, 
        temperature=0.7,  
        repetition_penalty=1.1  
    )

    llm = HuggingFacePipeline(pipeline=pipe)  # ✅ Use the pipeline here

    # pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    # llm = HuggingFacePipeline(pipeline=pipe)
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 5}))
    return qa_chain
