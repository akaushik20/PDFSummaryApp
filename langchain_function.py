from OpenAIKey import openai_key
import os
import pickle
from langchain.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

os.environ['OPENAI_API_KEY']=openai_key

file_path="faiss_store_pdf.pkl"

def summarize_article(selection):

    ## Save the Uploaded PDF locally
    with open(selection.name, 'wb') as w:
        w.write(selection.getvalue())

    ## if a valid file is found use PyPDFLOader to upload
    loader = PyPDFLoader(selection.name)
    data=loader.load()
    print("file Uploaded!!")

    ## Text Splitter
    text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],
                                                 chunk_size=1000)
    docs=text_splitter.split_documents(data)

    ##Create Embeddings
    embeddings=OpenAIEmbeddings()
    vectorstore_openai=FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    ## Invoking OpenAI model
    llm=OpenAI(temperature=0.6, max_tokens=1000)
    if os.path.exists(file_path):
        with open(file_path, "rb") as r:
            vectorstore=pickle.load(r)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,
                                                       retriever=vectorstore.as_retriever())
            query=("Can you describe what is the pdf document about and summarize the document in 5 bullet points?")
            result=chain({"question" : query}, return_only_outputs=True)
            return result["answer"]









