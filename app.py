import streamlit as st
import json

from pathlib import Path
from datetime import datetime

from src.vector_store import (
    load_embedding_model,
    load_chunks,
    load_faiss_index,
    validate_index_mapping,
    search_policy
)

from src.llm import generate_response

PAGE_TITLE = "Company Policy Assistant"
TOP_K = 10
LOG_FILE = Path(__file__).parent / "logs" / "query_logs.jsonl"


def log_query(question, retrieved_chunks, answer):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "final_answer": answer
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"
        )


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: gray;
        font-size: 1rem;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def initialize_rag():
    model = load_embedding_model()
    chunks = load_chunks()
    index = load_faiss_index()

    validate_index_mapping(index, chunks)

    return model, chunks, index


def display_sources(sources):
    with st.expander("View policy sources"):
        for number, source in enumerate(sources, start=1):
            st.markdown(f"**Source {number}**")
            st.write(f"Document: {source['document_name']}")
            st.write(f"Section: {source['section']}")
            st.write(f"Chunk ID: {source['chunk_id']}")
            st.divider()


st.markdown(
    '<p class="main-title">Company Policy Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Ask questions about company policies and procedures.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("About")

    st.write(
        "This application uses Retrieval-Augmented Generation (RAG) "
        "to answer questions using company policy documents."
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


try:
    with st.spinner("Initializing Policy Assistant..."):
        embedding_model, chunks, faiss_index = initialize_rag()

    st.success("Policy Assistant is ready!")

except Exception as error:
    st.error(f"Initialization failed: {error}")
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            display_sources(message["sources"])


question = st.chat_input("Ask a question about company policies...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching policy documents..."):
                retrieved_chunks = search_policy(
                    query=question,
                    model=embedding_model,
                    index=faiss_index,
                    chunks=chunks,
                    top_k=TOP_K
                )

            if not retrieved_chunks:
                answer = (
                    "The available policy documents do not "
                    "contain sufficient information to answer "
                    "this question."
                )

                sources = []

            else:
                with st.spinner("Generating answer using OpenRouter..."):
                    answer = generate_response(
                        query=question,
                        retrieved_chunks=retrieved_chunks
                    )

                sources = retrieved_chunks

            log_query(
                question=question,
                retrieved_chunks=retrieved_chunks,
                answer=answer
            )

            st.markdown(answer)

            if sources:
                display_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except Exception as error:
            st.error(f"Something went wrong: {error}")