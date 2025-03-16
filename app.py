import streamlit as st
import os
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.title("🤖 Chat with your PDF")

# File uploader for PDFs
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # Hardcoded Hugging Face API token (for quick testing ONLY)
    huggingfacehub_api_token = "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"

    # Initialize LLM from Hugging Face Hub with token
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # Create Retrieval QA chain with sources
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your PDF...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain({"question": user_input})
                answer = result["answer"]
                sources = result.get("sources", "No sources found.")
                response = f"**Answer:** {answer}\n\n**Sources:** {sources}"
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clean up after processing
    os.remove("temp.pdf")
else:
    st.info("Upload a PDF to start chatting.")
