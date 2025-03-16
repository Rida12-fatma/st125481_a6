Document-Based Question Answering using LangChain and FAISS
This project implements a document-based question-answering system that allows users to query PDF documents intelligently using natural language. It leverages LangChain, FAISS, Hugging Face Transformers, and Sentence Transformers to perform document parsing, embedding, indexing, and question-answering.

Features
PDF Document Loading and Parsing using PyPDFLoader
Text Chunking for efficient embedding and retrieval
Embedding generation with all-MiniLM-L6-v2 (Sentence Transformers)
Vector Store using FAISS for efficient similarity search
LLM-powered response generation using Hugging Face's FLAN-T5-Large
Persistent FAISS index for fast reload and inference
Installation
Install the required dependencies:

bash
Copy
Edit
pip install pypdf
pip install -U langchain-community
pip install faiss-cpu
pip install sentence_transformers
Setup
Hugging Face API Token
Set your Hugging Face API token as an environment variable:

bash
Copy
Edit
export HUGGINGFACEHUB_API_TOKEN=your_token_here
Input PDF Files
Add your PDF files to the specified location and update the file path in the notebook:

python
Copy
Edit
pdf_files = ["/path/to/your/document.pdf"]
Usage
Run the Notebook
The notebook will:

Load and split PDF text into chunks
Generate embeddings and build a FAISS index
Store/reload the FAISS index for reuse
Accept user queries and return answers using the LLM
Example Query
After the setup, you can ask questions related to the contents of your PDF:

python
Copy
Edit
query = "What are the skills mentioned in the resume?"
Model Info
LLM Used: google/flan-t5-large from Hugging Face
Embedding Model: all-MiniLM-L6-v2 from Sentence Transformers
Vector Store: FAISS
Notes
Ensure that large PDFs are split efficiently to avoid memory issues.
The FAISS index is saved as faiss_index.bin for persistence.
Avoid committing your Hugging Face API token in public repositories.
License
This project is for educational purposes. Please check licenses for Hugging Face models and LangChain before commercial use.

