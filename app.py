import streamlit as st

from rag.youtube import (
    extract_video_id,
    create_vector_store
)

from rag.model import load_llm

from rag.chain import create_rag_chain


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="centered"
)


st.title("🎥 YouTube Chatbot")

st.caption(
    "Ask questions about a YouTube video using LangChain RAG."
)


# ---------------------------------
# Session state
# ---------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------
# Load model
# ---------------------------------

@st.cache_resource
def get_model():

    return load_llm()


# ---------------------------------
# YouTube URL
# ---------------------------------

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


if st.button("Load Video", type="primary"):

    if not youtube_url:

        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        try:

            video_id = extract_video_id(
                youtube_url
            )

            with st.spinner(
                "Loading video and building knowledge base..."
            ):

                # Create FAISS vector store
                vector_store = create_vector_store(
                    video_id
                )

                # Retriever
                retriever = vector_store.as_retriever(
                    search_kwargs={
                        "k": 4
                    }
                )

                # Load LLM
                model = get_model()

                # Create LangChain RAG chain
                rag_chain = create_rag_chain(
                    retriever,
                    model
                )

                # Store everything in session
                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.rag_chain = (
                    rag_chain
                )

                st.session_state.messages = []

            st.success(
                "Video loaded successfully! "
                "You can now ask questions."
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )


# ---------------------------------
# Display chat history
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ---------------------------------
# Chat input
# ---------------------------------

question = st.chat_input(
    "Ask something about the video..."
)


if question:

    if st.session_state.rag_chain is None:

        st.warning(
            "Please load a YouTube video first."
        )

    else:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Assistant response
        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                answer = (
                    st.session_state
                    .rag_chain
                    .invoke(question)
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )