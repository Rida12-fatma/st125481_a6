# web.py — UI helper module for Streamlit app

import streamlit as st

def show_title():
    st.title("📄 Document Q&A Chatbot 🤖")
    st.write("Upload a PDF and ask questions about its contents using AI.")

def show_footer():
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, LangChain, FAISS, Sentence Transformers, and Hugging Face.")

def show_error(message):
    st.error(f"⚠️ {message}")

def show_success(message):
    st.success(f"✅ {message}")

def show_spinner(message):
    return st.spinner(f"⏳ {message}")
