import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import ChatMessage

from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token 
        self.container.markdown(self.text) 

def get_vectorstore(client):
    embeddings = OpenAIEmbeddings()
    
    vector_store = Qdrant(
        client=client, 
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
        embeddings=embeddings,
    )
    
    return vector_store


def get_conversation_chain(vectorstore, stream_handler):
    llm = ChatOpenAI(streaming=True, callbacks=[stream_handler],temperature=0, max_tokens=1000)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, max_token_limit=1000)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

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
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    # create collection
    # create_collection(client, "collection_name")

    # get vector store
    vectorstore = get_vectorstore(client)
    
    st.set_page_config(layout="wide")

    st.header("203 Study Buddy 💬")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # create conversation chain
            stream_handler = StreamHandler(st.empty())
            conversation_chain = get_conversation_chain(vectorstore, stream_handler)
            
            response = conversation_chain({"question": prompt})
            st.session_state.messages.append(ChatMessage(role="assistant", content=response['answer']))
    
if __name__ == '__main__':
    main()
