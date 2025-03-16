import streamlit as st
import os
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# Streamlit App Title
st.title("PDF Chatbot with LangChain & FAISS")

# File uploader for PDFs
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load PDF and split into documents
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Initialize LLM from Hugging Face Hub (replace with your own API key in .env or here)
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

    # Create Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # User input for questions
    query = st.text_input("Ask something about the PDF:")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.run(query)
        st.write("### Answer:")
        st.write(result)

    # Clean up
    os.remove("temp.pdf")

else:
    st.info("Please upload a PDF to begin.")
