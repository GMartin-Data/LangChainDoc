from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st


# Functions
def get_response(user_input: str):
    return "I don't know!"

def get_vectorstore_from_url(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Hello, I am a bot, how can I help you?"),
    ]

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a Website URL")
else:
    documents = get_vectorstore_from_url(website_url)
    with st.sidebar:
        st.write(documents)
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
