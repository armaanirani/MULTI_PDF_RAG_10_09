import streamlit as st

st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
st.header("Multi PDF Chatbot 📚")

user_question = st.text_input("Ask your question from the uploaded PDF files:")