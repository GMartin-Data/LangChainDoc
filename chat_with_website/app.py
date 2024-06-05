import streamlit as st

st.set_page_config(page_title="Chat with websites", page_icon="🤖")

st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")