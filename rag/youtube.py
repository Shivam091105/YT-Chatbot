from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def extract_video_id(url: str) -> str:

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    return url


def get_transcript(video_id: str) -> str:

    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.fetch(
            video_id,
            languages=["en"]
        )

        transcript = " ".join(
            chunk.text
            for chunk in transcript_list
        )

        return transcript

    except TranscriptsDisabled:
        raise Exception(
            "This video does not have captions available."
        )


def create_vector_store(video_id: str):

    # Get transcript
    transcript = get_transcript(video_id)

    # Split transcript
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store