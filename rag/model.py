import os

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)


load_dotenv()


def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    model = ChatHuggingFace(
        llm=llm,
        temperature=0.5
    )

    return model