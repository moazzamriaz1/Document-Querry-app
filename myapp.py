
import streamlit as st
import os
import openai
import PyPDF2

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain import VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import nltk
from streamlit_chat import message

nltk.download("punkt")

def main():
    st.title("Document Query System")

    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf'])
    if uploaded_file:
        # Save the uploaded file
        file_path = os.path.join('./uploaded_files', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Initialize OpenAIEmbeddings
        openai_api_key = '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'
        os.environ['OPENAI_API_KEY'] = openai_api_key

        # Initialize OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

        # Load the file as document
        _, ext = os.path.splitext(file_path)
        if ext == '.txt':
            loader = UnstructuredFileLoader(file_path)
        elif ext == '.pdf':
            loader = UnstructuredPDFLoader(file_path)
        else:
            st.write("Unsupported file format.")
            return

        documents = loader.load()

        # Split the documents into texts
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create Chroma vectorstore from documents
        doc_search = Chroma.from_documents(texts, embeddings)

        # Initialize VectorDBQA
        chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)


        st.write("Upload complete!")
        st.write("Enter your query below:")

        query = st.text_input("Query")
        if st.button("Run Query"):
            result = chain.run(query)
            #st.write("Query Result:")
            # st.write(result)
            message(query, is_user=True)
            message(result)

    else:
        st.write("No file uploaded.")


if __name__ == '__main__':
    main()
    



