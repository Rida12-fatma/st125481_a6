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
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")

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

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    file_path = "temp.pdf"
    file.save(file_path)
    documents = load_documents(file_path)
    index, embeddings, embedder = create_faiss_index(documents)
    return jsonify({"message": "Document processed and indexed!"})

@app.route("/query", methods=["POST"])
def query_chatbot():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    query_embedding = embedder.encode(query)
    D, I = index.search(np.array([query_embedding]), k=5)
    retrieved_docs = [documents[i] for i in I[0]]
    qa_chain = RetrievalQA(llm=llm, retriever=retrieved_docs)
    response = qa_chain.run(query)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
