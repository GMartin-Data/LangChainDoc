import streamlit as st


# Functions
def get_response(user_input: str):
    return "I don't know!"

# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    response = get_response(user_query)
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        st.write(response)
