from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st


# Functions
def get_response(user_input: str):
    return "I don't know!"

# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")
chat_history = [
    AIMessage("Hello, I am a bot, how can I help you?"),
]

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    response = get_response(user_query)
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response))

with st.sidebar:
    st.write(chat_history)

