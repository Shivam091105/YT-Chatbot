# 🎥 YouTube RAG Chatbot

A conversational **YouTube Question Answering chatbot** built using **LangChain, Hugging Face, FAISS, and Streamlit**.

The application takes a YouTube video URL, extracts its transcript, creates semantic embeddings, stores them in a FAISS vector database, retrieves relevant transcript sections for each question, and generates an answer using a Hugging Face LLM.

The project uses **LangChain LCEL (LangChain Expression Language)** to compose the complete Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features

- 🎥 Accepts a YouTube video URL
- 📝 Automatically extracts the video transcript
- ✂️ Splits long transcripts into manageable chunks
- 🧠 Generates semantic embeddings using Hugging Face
- 🔎 Performs similarity search using FAISS
- 🤖 Generates answers using a Hugging Face chat model
- 🔗 Uses LangChain LCEL to compose the RAG pipeline
- ⚡ Uses parallel execution for question passthrough and context retrieval
- 💬 Streamlit-based chat interface
- 🔐 API keys stored securely using `.env`
- 📦 Isolated Python environment using `venv`

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │    Streamlit UI     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  YouTube URL    │
                           └────────┬────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │ YouTube Transcript API  │
                      └────────────┬────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ Transcript Text  │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ RecursiveCharacter        │
                    │ TextSplitter              │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ Transcript     │
                         │ Chunks         │
                         └───────┬────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Hugging Face Embeddings  │
                    │ all-MiniLM-L6-v2         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ FAISS Vector   │
                         │ Store          │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ Retriever      │
                         └───────┬────────┘
                                 │
                                 │
              User Question     │
                    │            │
                    ▼            ▼
          ┌───────────────────────────────┐
          │       LangChain LCEL          │
          │                               │
          │     RunnableParallel          │
          │                               │
          │  ┌────────────┐ ┌───────────┐ │
          │  │  context   │ │ question  │ │
          │  │            │ │           │ │
          │  │ Retriever  │ │ Passthrough│ │
          │  └─────┬──────┘ └─────┬─────┘ │
          │        │              │       │
          │        └──────┬───────┘       │
          │               ▼               │
          │          Prompt Template      │
          │               │               │
          │               ▼               │
          │        Hugging Face LLM       │
          │               │               │
          │               ▼               │
          │       StrOutputParser         │
          └───────────────┬───────────────┘
                          │
                          ▼
                   ┌───────────────┐
                   │    Answer     │
                   └───────────────┘