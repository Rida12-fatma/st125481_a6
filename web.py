import streamlit as st
from app import ask_chatbot

# Set page title and favicon
st.set_page_config(page_title="Rida Fatma's Resume Chatbot", page_icon=":robot_face:")

# Streamlit UI
st.title("Rida Fatma's Resume Chatbot :robot_face:")
st.write("Ask me anything about Rida Fatma's resume!")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer, source_documents = ask_chatbot(user_question)
        st.write("**Answer:**", answer)

        # Display source documents (optional)
        if source_documents:
            st.write("**Source Documents:**")
            for doc in source_documents:
                st.write(doc.page_content)
