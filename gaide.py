import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
import os
import requests

load_dotenv()

GROQ_API_KEY = st.secrets("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

class GroqLLM(LLM):
    def __init__(self, temperature=0.3, model=GROQ_MODEL):
        self.temperature = temperature
        self.model = model

    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt, stop=None, run_manager=None):
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and make quiz with 5 questions with options. 
    If the answer is not in the provided context just say, 
    \"Wow you are asking from out of syllabus Just Go on with it, but don't go deeply because it is not necessary now!\".
    Give definitions and real time examples.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    llm = ChatGroq(model_name="llama3-8b-8192") 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="GaiDE")
    st.header("Interactive RAG-based LLM for Specific-PDF", divider='rainbow')

    user_question = st.text_input("Ask your doubt?")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
