import os
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.llms import HuggingFaceHub

# Set the Hugging Face API Token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your actual token

# Load Hugging Face API Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your environment variables.")

# Initialize Hugging Face LLM
hf_llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Define FAISS index file path
INDEX_PATH = "faiss_index.bin"

# Check if FAISS index exists and load it if available
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    print("FAISS index loaded from disk.")
else:
    print("FAISS index not found. Rebuilding...")

    # Load Personal Documents
    pdf_files = [
        "/content/RIDA FATMA Resume.pdf"
    ]

    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Extract text content from chunks
    texts = [doc.page_content for doc in text_chunks]

    # Convert text to embeddings using SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)

    # Convert embeddings to numpy array for FAISS
    embedding_matrix = np.array(embeddings).astype("float32")

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save FAISS index to disk
    faiss.write_index(index, INDEX_PATH)
    print("FAISS index saved to disk.")

# Create FAISS vector store
docstore = InMemoryStore()
index_to_docstore_id = {}

document_objects = []
for i, doc in enumerate(text_chunks):
    doc_object = Document(page_content=doc.page_content, metadata=doc.metadata)
    document_objects.append(doc_object)
    index_to_docstore_id[i] = str(i)

docstore.mset([(str(i), doc) for i, doc in enumerate(document_objects)])

vector_store = FAISS(
    embedding_function=embedding_model.encode,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Override `docstore.search` with `mget()`
def docstore_get(doc_id):
    docs = docstore.mget([doc_id])
    return docs[0] if docs else None

vector_store.docstore.search = docstore_get

# Setup Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up LangChain RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Function to ask chatbot questions
def ask_chatbot(question):
    retrieved_docs = retriever.get_relevant_documents(question)

    if not retrieved_docs:
        return "No relevant information found.", []

    response = qa_chain.invoke({"query": question})
    return response["result"], response["source_documents"]
