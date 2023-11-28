import qdrant_client
import streamlit as st
from PyPDF2 import PdfReader

import time

from qdrant_client import models
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

OPEN_AI_EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"
OPEN_AI_MODEL = "gpt-3.5-turbo"

#takes in list of pdfs file, return only 1 string representing all the content
def get_pdf_text_from_files(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#takes in a pdf file, return only 1 string representing all the content
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#using langchain to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(client):
    embeddings = OpenAIEmbeddings(model=OPEN_AI_EMBEDDINGS_MODEL_NAME)
    
    vector_store = Qdrant(
        client=client, 
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
        embeddings=embeddings,
    )
    
    return vector_store

def get_text_from_list_of_files(files) :
    text = ""
    for file in files:
        text += file.read().decode("utf-8")
    return text

def create_collection(client, cname):
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE
    )

    client.recreate_collection(
        collection_name=cname,
        vectors_config=vectors_config,
    )

def main():    
    client = qdrant_client.QdrantClient(
        st.secrets["QDRANT_HOST"],
        api_key=st.secrets["QDRANT_API_KEY"],
        timeout=1000000000
    )
    # create_collection(client, "test")
    
    # config for bulk upload
    client.update_collection(
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
        # vectors_config = qdrant_client.http.models.VectorParams(
        #     size=1536,
        #     distance=qdrant_client.http.models.Distance.COSINE
        # ),
        # shard_number=2,
    )    
    
    vector_store = get_vectorstore(client)
    
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    uploaded_files = st.file_uploader(
        "Select training files: ", 
        accept_multiple_files=True, type=['pdf'],
        key=st.session_state["file_uploader_key"],)

    text_chunks = []

    # get the text chunks
    for uploaded_file in uploaded_files:
        try:
            raw_text = get_pdf_text(uploaded_file)
            # raw_text = uploaded_file.read().decode("utf-8")
            tmp_text_chunks = get_text_chunks(raw_text)
                        
            continueFlag = True
            for tmp_chunk in tmp_text_chunks:
                token_count = len(tmp_chunk)
                # st.write(token_count)
                if token_count > 1000:
                    st.write(f"file {uploaded_file.name} has at least a text vector with more than limit tokens, so it will be discarded")
                    continueFlag = False
                    break                
                
            if continueFlag==False:
                continue
            
            st.write(f"chunks length of file {uploaded_file.name}: {len(tmp_text_chunks)}")
            text_chunks.extend(tmp_text_chunks)
            
        except Exception as err:
            st.write(f"Unexpected {err=}, {type(err)=}")
            st.write(f"file {uploaded_file.name=} will be discarded")
            continue

    st.write(f"Total chunks length: {len(text_chunks)}")
    st.write(text_chunks)
    
    if st.button('Upload Training Data', disabled=len(text_chunks)==0):
        with st.spinner('Please wait...'):
            # save the text in vector store
            vector_store.add_texts(text_chunks)
        text_chunks = []
        st.success('Done!')
        time.sleep(0.2)
        st.session_state["file_uploader_key"] += 1
        st.rerun()

    # reset indexing config
    client.update_collection(
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=2000000000,
        ),
    ) 

    print("Exit")

if __name__ == '__main__':
    main()
