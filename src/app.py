import streamlit as st

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


import pathlib
from langchain_community.document_loaders import PyPDFLoader
@st.cache_data
def pdf_loader(path):
    all_documents = []
    docs_dir = pathlib.Path(path)
    for file in docs_dir.iterdir():
        loader = PyPDFLoader(file)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

download_and_unzip("https://minio.carlboettiger.info/public-data/hwc.zip", "hwc")
docs = pdf_loader('hwc/')


from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
             model = "cirrus",
             api_key = cirrus_key, 
             base_url = "https://llm.cirrus.carlboettiger.info/v1",
)


# Build a retrival agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=300)
splits = text_splitter.split_documents(docs)

from langchain_core.vectorstores import InMemoryVectorStore
@st.cache_resource
def vector_store(_splits):
    vectorstore = InMemoryVectorStore.from_documents(documents=_splits, embedding=embedding)
    retriever = vectorstore.as_retriever()
    return retriever

# here we go, slow part:
retriever = vector_store(splits)

# Set up the language model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "llama3", api_key = api_key, base_url = "https://llm.nrp-nautilus.io",  temperature=0)
## Cirrus instead:
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following scientific articles as the retrieved context to answer "
    "the question. Appropriately cite the articles from the context on which your answer is based. "
    "Do not attempt to cite articles that are not in the context."
    "If you don't know the answer, say that you don't know."
    "Use up to five sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
from langchain.chains.combine_documents import create_stuff_documents_chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
from langchain.chains import create_retrieval_chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# Place agent inside a streamlit application:
if prompt := st.chat_input("What are the most cost-effective prevention methods for elephants raiding my crops?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = rag_chain.invoke({"input": prompt})
        st.write(results['answer'])

        with st.expander("See context matched"):
            # FIXME parse results dict and display in pretty format
            st.write(results['context'])


# adapt for memory / multi-question interaction with:
# https://python.langchain.com/docs/tutorials/qa_chat_history/

# Also see structured outputs.
