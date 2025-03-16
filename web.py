import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("NLP Model Interface")

# User input
user_input = st.text_area("Enter text:")

if st.button("Generate Embedding"):
    if user_input:
        embedding = model.encode(user_input)
        st.write("Embedding:", embedding)
    else:
        st.warning("Please enter text before generating embeddings.")
