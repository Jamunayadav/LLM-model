import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown("""""

    ## About
    This app is LLM powered chatbot
    """)
    add_vertical_space(5)
    st.write("made with love by jam.code")



def main():
    st.header("Chat with your file")
    load_dotenv()

    # upload a pdf file
    pdf = st.file_uploader("upload your file", type="pdf")
    


    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text = text)
        
        store_name = pdf.name[:-4]
    
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb")as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded from the disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
            # st.write("Embeddings loaded succesfully")
        Query = st.text_input("Ask yourquestions")
        st.write(Query)

        if Query:

            docs = VectorStore.similarity_search(Query = Query, k = 3)
            llm = OpenAI()
            chain = load_qa_chain(llm= llm,chain_type= "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = Query)
                print(cb)
            st.write(response)
            #st.write(docs)
        
if __name__ == '__main__':
    main()