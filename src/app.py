import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

## dockerized streamlit app wants to read from os.getenv(), otherwise use st.secrets
import os
api_key = os.getenv("LITELLM_KEY")
if api_key is None:
    api_key = st.secrets["LITELLM_KEY"]
cirrus_key = os.getenv("CIRRUS_KEY")
if cirrus_key is None:
    cirrus_key = st.secrets["CIRRUS_KEY"]        

st.title("HWC LLM Testing")


'''
(Demo will take a while to load first while processing all data!  Will be pre-processed in future...)
'''

# +
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import requests
import zipfile

def download_and_unzip(url, output_dir):
    response = requests.get(url)
    zip_file_path = os.path.basename(url)
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_file_path)

url = "https://minio.carlboettiger.info/public-data/hwc.zip"
output_dir = "hwc"
download_and_unzip(url, "hwc")

import pathlib
@st.cache_data
def pdf_loader(path):
    all_documents = []
    docs_dir = pathlib.Path(path)
    for file in docs_dir.iterdir():
        loader = PyPDFLoader(file)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

docs = pdf_loader('hwc/')



# Set up the language model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "llama3", api_key = api_key, base_url = "https://llm.nrp-nautilus.io",  temperature=0)
## Cirrus instead:
embedding = OpenAIEmbeddings(
                 model = "cirrus",
                 api_key = cirrus_key, 
                 base_url = "https://llm.cirrus.carlboettiger.info/v1",
)



# Build a retrival agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Place agent inside a streamlit application:

if prompt := st.chat_input("What are the most cost-effective prevention methods for elephants raiding my crops?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = rag_chain.invoke({"input": prompt})
        st.write(results['answer'])

        with st.expander("See context matched"):
            st.write(results['context'][0].page_content)
            st.write(results['context'][0].metadata)


# adapt for memory / multi-question interaction with:
# https://python.langchain.com/docs/tutorials/qa_chat_history/

# Also see structured outputs.
