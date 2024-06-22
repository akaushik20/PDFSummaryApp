import streamlit as st
from langchain_function import summarize_article

st.subheader("Welcome to summarization Tool")
st.write("Upload PDF")
selection=st.sidebar.file_uploader('Choose .pdf file only!!', type='pdf')

if selection:
    response = summarize_article(selection)
    st.write(response)