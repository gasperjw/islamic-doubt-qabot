from re import template
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENV = st.secrets['PINECONE_ENV']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

@st.cache_data
def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name='yaqeenapp'
    )
    return doc_db

llm = ChatOpenAI(temperature = 0)
doc_db = embedding_db()

def retrieval_answer(query):
    prompt_template = """You are an Islamic QA bot. People could either ask for normal knowledge or to clear up a doubt. If it is a doubt, you are to help deal with peoples doubts in Islam. Use the sources to create an argument to answer that doubt.  If the answer is not in the sources, just say that you don't know and refer them to a scholar, don't try to make up an ANY answer.

    {context}

    Question: {question}
    Answer by sympathizing with the person and then making your answer like an argument:"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
    )

    query = query
    result = qa.run(query)
    return result

def main():
    st.title("Islamic Doubts Q/A Bot Trained on Yaqeen Articles (TEST) ")
    with st.expander("Learn More About the Project"):
        st.write("""
        ### Project Goal
        To develop an Islamic Doubt Q/A System engineered to effectively address and clarify doubts and misconceptions pertaining to Islamic teachings, beliefs, and practices. 
        The system is trained on articles from Yaqeen Insititute. 
        
        ### Project Information
        This project is built using LLM and Pinecone to answer queries related to Islamic Doubt. 
        The app takes a query, processes it, and retrieves the most relevant answer from the database.
        
        #### Technologies Used
        - **LLM**: For generating responses based on the input queries.
        - **Pinecone**: A vector database for efficient data retrieval.
        - **Streamlit**: For creating the web interface of the application.
        
        #### How it Works
        - The user inputs a query.
        - The app processes the query and searches for the most relevant response in the database.
        - The response is then displayed on the app interface.
        """)
    text_input = st.text_input("Ask your query...") 
    ask_button = st.button("Ask Query")
    
    if text_input and (ask_button or text_input != ""):
        st.info("Your Query: " + text_input)
        answer = retrieval_answer(text_input)
        st.success(answer)


if __name__ == "__main__":
    main()
