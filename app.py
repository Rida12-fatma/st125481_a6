import os
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from langchain_core.documents import Document
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

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

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    if not FAISS_AVAILABLE:
        return jsonify({"error": "FAISS is not available. Please install faiss-cpu."}), 500
    
    if faiss_index is None:
        return jsonify({"error": "FAISS index is not loaded properly."}), 500
    
    retriever = RetrievalQA(llm=llm, retriever=faiss_index)
    response = retriever.run(question)
    
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
