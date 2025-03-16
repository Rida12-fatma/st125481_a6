import os
import faiss
import numpy as np
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter

# Load Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    st.error("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")
    st.stop()

# Initialize Hugging Face LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token=hf_token)

# Load and process documents
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Create FAISS index
def create_faiss_index(documents):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [embedder.encode(doc.page_content) for doc in documents]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings, embedder

st.title("RAG Chatbot")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    documents = load_documents("temp.pdf")
    index, embeddings, embedder = create_faiss_index(documents)
    st.success("Document processed and indexed!")

# Query chatbot
query = st.text_input("Ask a question about the document:")
if query and uploaded_file is not None:
    query_embedding = embedder.encode(query)
    D, I = index.search(np.array([query_embedding]), k=5)
    retrieved_docs = [documents[i] for i in I[0]]
    qa_chain = RetrievalQA(llm=llm, retriever=retrieved_docs)
    response = qa_chain.run(query)
    st.write("### Answer:", response)
