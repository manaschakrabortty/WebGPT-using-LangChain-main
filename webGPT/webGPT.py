import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import numpy as np
import google.generativeai as palm
import pandas as pd
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os 



def get_conversation_chain(vectorstore):

    llm = GooglePalm()
    llm.temperature = 0.1
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


    return conversation_chain



def get_vectorstore(text_chunks):
    api_key = st.secrets["GOOGLE_API_KEY"]
    palm.configure(api_key=api_key)
    embeddings = GooglePalmEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def handle_userinput(user_question):

    response = st.session_state.conversation({'question' : user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user=True)
        else:
            message(msg.content)


def ChatBot(text):

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    

    with st.spinner("Processing"):
        chunked_text = get_text_chunks(text)
        vectorstore = get_vectorstore(chunked_text)

        st.session_state.conversation = get_conversation_chain(vectorstore)
        

    
    


def main():
    load_dotenv(find_dotenv())

    st.title("Chat with WebGPT")

    website_url = st.text_input(label="Enter the website URL", placeholder="It works better if its a Wikipedia page")
    
    if st.button("Chat Now"):
        if website_url:
            response = requests.get(website_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                scraped_data = soup.get_text()
                ChatBot(scraped_data)

                
            else:
                st.write(f"Failed to retrieve the page. Status code: {response.status_code}")
        else:
            st.write("Please enter a website URL.")
    

    user_question = st.text_input(label = "Ask a question related to your website", placeholder="Ask a question")
    if user_question:   
        handle_userinput(user_question)



if __name__ == '__main__':
    main()