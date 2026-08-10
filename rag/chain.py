from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def create_rag_chain(retriever, model):

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer ONLY from the provided context.

If the context is insufficient, say:
"I don't know based on the video."

Context:
{context}

Question:
{question}
"""
    )
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
    )

    rag_chain = (
        parallel_chain | prompt | model | StrOutputParser()
    )

    return rag_chain