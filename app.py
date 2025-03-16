import os
import faiss
import numpy as np
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.llms import HuggingFaceHub

# Set Hugging Face API Token (securely from secrets if on Streamlit Cloud)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    st.error("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment or Streamlit secrets.")
    st.stop()

# Initialize Hugging Face LLM
hf_llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Streamlit UI
st.title("Document-Based Q&A Chatbot 🤖📄")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Load document using PyPDFLoader
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()

        # Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(documents)

        # Convert chunks to text
        texts = [doc.page_content for doc in text_chunks]

        # Generate embeddings
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        embedding_matrix = np.array(embeddings).astype("float32")

        # Build FAISS index
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # Create document store
        docstore = InMemoryStore()
        index_to_docstore_id = {}
        document_objects = []
        for i, doc in enumerate(text_chunks):
            doc_object = Document(page_content=doc.page_content, metadata=doc.metadata)
            document_objects.append(doc_object)
            index_to_docstore_id[i] = str(i)
        docstore.mset([(str(i), doc) for i, doc in enumerate(document_objects)])

        # Create FAISS vector store
        vector_store = FAISS(
            embedding_function=embedding_model.encode,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        # Create retriever and QA chain
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=hf_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        st.success("Document processed. Ask your questions below!")

        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("Getting answer..."):
                response = qa_chain.invoke({"query": query})
                st.markdown(f"**Answer:** {response['result']}")
                with st.expander("Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}:** {doc.page_content[:500]}...")
