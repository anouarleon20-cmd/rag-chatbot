import streamlit as st



# ============================
# 1. FILE PATH TO YOUR PDF
# ============================
PDF_PATH = "document.pdf"   # <-- rename your PDF to document.pdf


# ============================
# 2. HuggingFace Token
# ============================
HF_TOKEN = st.secrets["HF_TOKEN"]     # Secure token from Streamlit cloud
MODEL_NAME = "google/gemma-2-2b-it"


import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from kaggle_secrets import UserSecretsClient



@st.cache_data
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

all_text = load_pdf_text(PDF_PATH)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

documents = text_splitter.create_documents([all_text])

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@st.cache_resource
def create_vectorstore(docs):
    return FAISS.from_documents(docs, embedding_model)

vectorstore = create_vectorstore(documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN
)

class GemmaWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 400

    @property
    def _llm_type(self) -> str:
        return "gemma_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2
        )
        return response.choices[0].message["content"]

gemma_llm = GemmaWrapper(client=client)

qa_chain = RetrievalQA.from_chain_type(
    llm=gemma_llm,
    retriever=retriever,
    chain_type="stuff"
)

st.title("📘 Milestone 1 RAG Chatbot")
st.write("Ask a question based on the MS1 document:")

query = st.text_input("Your question:")

if query:
    st.subheader("Answer:")
    st.write(qa_chain.run(query))

    with st.expander("Retrieved Chunks"):
        docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(docs):
            st.markdown(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.write("---")
