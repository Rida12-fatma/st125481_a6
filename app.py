import os
import numpy as np
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain_core.documents import Document

# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_token_here"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_length": 512},
    task="text2text-generation"
)

# Try to import FAISS
try:
    import faiss
    from langchain.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Define FAISS index path
INDEX_PATH = "faiss_index.bin"

def load_faiss_index():
    if FAISS_AVAILABLE and os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

faiss_index = load_faiss_index()

# Streamlit UI Setup
st.title("AI-Powered Question Answering")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not question:
        st.error("No question provided.")
    elif not FAISS_AVAILABLE:
        st.error("FAISS is not available. Please install faiss-cpu.")
    elif faiss_index is None:
        st.error("FAISS index is not loaded properly.")
    else:
        retriever = RetrievalQA(llm=llm, retriever=faiss_index)
        response = retriever.run(question)
        st.success(f"Answer: {response}")
