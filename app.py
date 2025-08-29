import os
import pickle
import streamlit as st
import PyPDF2
from dotenv import load_dotenv

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

# Groq (LangChain integration)
from langchain_groq import ChatGroq

# Community integrations (moved out of core)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------
# Setup
# --------------------------
load_dotenv()  # loads GROQ_API_KEY from .env

st.set_page_config(page_title="📚 Research Paper Assistant (Groq)", layout="wide")
st.title("📄 Research Paper Summarizer + Chatbot (Groq)")

# Sidebar controls
with st.sidebar:
    st.header("⚙ Settings")
    model_choice = st.selectbox(
        "Groq model",
        options=[
            "llama3-8b-8192",      # fast, great default
            "llama3-70b-8192",     # stronger, slower
            "mixtral-8x7b-32768",  # MoE, good quality; skip if unavailable
        ],
        index=0
    )
    k_chunks = st.slider("Retrieved chunks (k)", min_value=2, max_value=6, value=3, step=1)
    chunk_size = st.slider("Chunk size", min_value=600, max_value=1500, value=900, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=20, max_value=200, value=80, step=10)
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.caption("Tip: Smaller k, smaller chunk size = faster responses.")

uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

# Keep chat history + vectorstore in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "paper_hash" not in st.session_state:
    st.session_state.paper_hash = None

def hash_bytes(b: bytes) -> str:
    # simple stable hash (no external deps)
    import hashlib
    return hashlib.sha256(b).hexdigest()

def build_vectorstore_from_pdf(pdf_bytes: bytes) -> FAISS:
    pdf = PyPDF2.PdfReader(pdf_bytes)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    if not text.strip():
        raise ValueError("No selectable text found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    # Lightweight, fast embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def get_llm():
    return ChatGroq(
        model=model_choice,
        temperature=temperature,
        max_retries=2,
        # streaming=True,  # uncomment if you add a streaming callback
    )

# Chatbot prompt (RAG-friendly, avoids hallucinations)
chatbot_prompt = PromptTemplate.from_template(
    """
You are an AI research assistant. Use the paper context when possible, but also be a knowledgeable guide.

    Rules:
    1. If the user's question can be answered from the paper context, give the answer in 20 lines, citing exact text when useful.
    2. If the question is about specific details (authors, venue, year, dataset, methods, metrics), extract them exactly as written in the paper.
    3. If the question is about lines or text, return the closest relevant passage from the paper.
    4. If the answer is not present in the paper:
        - Do NOT say only "not mentioned".
        - Instead, say: "The paper does not mention this. However, generally speaking, [give a clear explanation of the concept]."
    5. Keep answers descriptive, and accessible to a student or researcher.


Paper Context:
{context}

Question:
{question}

Answer:
"""
)

# Summarization prompt (structured)
summary_prompt = PromptTemplate.from_template(
    """
You are a research assistant. Provide a structured summary of the paper covering:
- Problem statement
- Methodology
- Key findings/results
- Conclusion
- Significance/impact

Use clear headings and keep it upto 400 words.
Paper Context:
{context}
"""
)

# --------------------------
# Main logic
# --------------------------
if uploaded_file:
    # Cache vectorstore per unique file (in session)
    file_bytes = uploaded_file.getvalue()
    this_hash = hash_bytes(file_bytes)

    if st.session_state.paper_hash != this_hash or st.session_state.vectorstore is None:
        with st.spinner("Indexing paper (one-time per file)..."):
            try:
                st.session_state.vectorstore = build_vectorstore_from_pdf(uploaded_file)
                st.session_state.paper_hash = this_hash
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
                st.stop()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_chunks})

    # Build chains
    llm = get_llm()

    # Chatbot chain (Conversational)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": chatbot_prompt},
        return_source_documents=False,
    )

    # Summarization chain (single-shot)
    summarize_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": summary_prompt},
        return_source_documents=False,
    )

    # UI: Chat
    st.subheader("💬 Chat with the paper")
    user_q = st.text_input("Ask anything about the paper (e.g., Who are the authors? What dataset was used?)")

    if user_q:
        with st.spinner("Thinking..."):
            result = chat_chain({"question": user_q, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            st.session_state.chat_history.append((user_q, answer))

        st.markdown(f"*You:* {user_q}")
        st.markdown(f"*Assistant:* {answer}")

    # UI: Summary
    if st.button("📑 Summarize the paper"):
        with st.spinner("Summarizing..."):
            summary = summarize_chain.run("Summarize this paper.")
        st.subheader("Summary")
        st.write(summary)

    # Show history (optional)
    if st.session_state.chat_history:
        with st.expander("🧠 Chat history"):
            for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"*Q{i}:* {q}")
                st.markdown(f"*A{i}:* {a}")
else:
    st.info("Upload a PDF to begin.")