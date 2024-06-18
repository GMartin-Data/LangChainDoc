from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# Load docs
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item
             for doc in docs
             for item in doc]  # Flatten docs list

# Split docs
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
docs_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore and associate retriever
vectorstore = Chroma.from_documents(
    documents = docs_splits,
    collection_name = "rag-chroma",
    embedding = OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()