📄 NLP-Based PDF Question Answering System
This project implements a Natural Language Processing (NLP) pipeline for querying PDF documents using a Retrieval-Based Question Answering (QA) approach. It utilizes Hugging Face Transformers, FAISS for vector similarity search, and SentenceTransformers for text embeddings.

🚀 Features
Load and parse PDF documents
Generate embeddings using all-MiniLM-L6-v2
Build a FAISS vector store for efficient retrieval
Integrate Hugging Face Language Model for generating answers
Ask questions from loaded PDFs interactively
🛠️ Technologies Used
Python
pypdf for PDF parsing
sentence-transformers
faiss for vector indexing
transformers and HuggingFaceHub for LLMs
RetrievalQA from LangChain (assumed)
📂 Project Structure
File	Description
st125481_a6__NLP.ipynb	Main notebook containing all code and logic
README.md	Project overview and usage instructions
📋 Installation
bash
Copy
Edit
pip install pypdf sentence-transformers faiss-cpu transformers langchain
📈 Usage
Add PDF files: Place your PDF files in the designated directory.
Run the notebook: Execute cells to process PDFs, create embeddings, and query using the LLM.
Ask Questions: Use the chatbot interface to ask questions related to the PDF content.
🔐 Hugging Face Token
Make sure to load your Hugging Face API token securely for LLM access.

📄 Example
python
Copy
Edit
answer = ask_question("What is the summary of the first section?")
print(answer)
















