import os
import numpy as np
import faiss
import openai
import pdfplumber
import docx
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def load_document(file_path):
    """Load content from a text, PDF, or DOCX file."""
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text() is not None])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type.")

def embed_and_store_document(file_path):
    """Embed and store document content in FAISS."""
    text = load_document(file_path)
    if not text:
        raise ValueError("Document is empty or could not be read.")

    print("Loaded Document:", text[:500])

    chunk_size = 400
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    if not chunks:
        raise ValueError("No text chunks to process.")

    print("Number of Chunks:", len(chunks))

    # Create FAISS index directly using langchain's FAISS wrapper
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,  # Note: changed from embeddings to embedding
        metadatas=[{"source": file_path, "chunk_id": i} for i in range(len(chunks))]
    )
    
    print("FAISS Index Initialized.")
    return vectorstore

def generate_answer(query, vectorstore):
    """Generate an answer to the query using stored document embeddings."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Answer the following question based on the context:\n{context}\n\nQuestion: {query}"
    )

    llm = OpenAI(model="text-davinci-003", temperature=0.7)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}  # Fixed prompt parameter
    )

    result = qa_chain({"query": query})  # Changed to use dict input
    return result

def initialize_faiss_store(texts=None, embedding_model=None):
    """Create a new FAISS vector store using langchain's FAISS wrapper."""
    if texts is None:
        texts = []
    
    if embedding_model is None:
        embedding_model = embeddings  # Use the global OpenAI embeddings instance
    
    # Create FAISS index using langchain's wrapper
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model
    )
    
    return vectorstore