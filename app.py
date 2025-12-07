import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
from typing import Optional, List, Any


# ============================
# 1. FILE PATH TO YOUR PDF
# ============================
PDF_PATH = "document.pdf"   # <-- rename your PDF to document.pdf


# ============================
# 2. HuggingFace Token
# ============================
HF_TOKEN = st.secrets["HF_TOKEN"]     # Secure token from Streamlit cloud
MODEL_NAME = "google/gemma-2-2b-it"


# ============================
# 3. LOAD PDF TEXT
# ============================
@st.cache_resource
def load_pdf_text():
    reader = PdfReader(PDF_PATH)
    text = ""
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text += p + "\n"
    return text


# ============================
# 4. CUSTOM LLM WRAPPER
# ============================
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 400

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]


# ============================
# 5. BUILD RAG PIPELINE
# ============================
@st.cache_resource
def initialize_rag():
    text = load_pdf_text()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    documents = splitter.create_documents([text])

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM client (Gemma)
    client = InferenceClient(
        model=MODEL_NAME,
        token=HF_TOKEN
    )

    gemma_llm = GemmaLangChainWrapper(client=client)

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=gemma_llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain, retriever


qa_chain, retriever = initialize_rag()


# ============================
# 6. STREAMLIT UI
# ============================
st.title("📘 RAG Chatbot — Document-Based Q&A")
st.write("Ask any question based on the document.")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        answer = qa_chain.run(question)

    st.subheader("Answer:")
    st.write(answer)

    # Show retrieved chunks
    st.subheader("Source Chunks:")
    docs = retriever.get_relevant_documents(question)

    for i, d in enumerate(docs):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(d.page_content)
        st.markdown("---")
