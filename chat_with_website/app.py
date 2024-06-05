import streamlit as st

st.set_page_config(page_title="Chat with websites", page_icon="🤖")

st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

st.chat_input("Type your message here...")

with st.chat_message("AI"):
    st.write("Hello, how can I help you?")

with st.chat_message("Human"):
    st.write("I want to know about LangChain.")

with st.chat_message("AI"):
    st.write("I don't know!")