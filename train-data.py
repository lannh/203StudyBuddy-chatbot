import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os

from qdrant_client import QdrantClient, models
from langchain.vectorstores import Qdrant

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#takes in list of pdfs file, return only 1 string representing all the content
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#using langchain to split text into chunks 
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(client):
    embeddings = OpenAIEmbeddings()
    
    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store


def main():
    load_dotenv()
    
    client = QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    # config for bulk upload
    client.recreate_collection(
        collection_name="java-oop-doc",
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
        vectors_config = qdrant_client.http.models.VectorParams(
            size=1536,
            distance=qdrant_client.http.models.Distance.COSINE
        ),
        shard_number=2,
    )    
    
    vector_store = get_vectorstore(client)

    pdf_docs = ["Java_OOP_doc_example.pdf"]
    
    # get pdf text
    raw_text = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    print(len(text_chunks))
    
    # save the text in vector store
    vector_store.add_texts(text_chunks)

    # reset indexing config
    client.update_collection(
        collection_name="java-oop-doc",
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    ) 

    print("Exit")

if __name__ == '__main__':
    main()
