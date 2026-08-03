from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import Retriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from constants import PRESIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL_NAME, LLM_NAME

def main():
    vector_store = VectorStore(
        presist_directory=PRESIST_DIRECTORY, collection_name=COLLECTION_NAME
    )
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )
    print("Retrieving context...")
    query="what can this app do?"
    context = retriever.retrieve(
        query=query
    )
    print("Context retrieved")
    
    llm = ChatOllama(
        model=LLM_NAME,
        reasoning=False,
        temperature=0,
        keep_alive="30m"
    )

    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful memory assistant that can create tasks and recall everything from my memory context.
            - Answer only by using following context.
            - If the context don't have the answer then just say I don't have enough context.

            Context: {context}
            """
        ),
        (
            "human", "{query}"
        )
    ])

    chain = prompt | llm | parser

    results = chain.invoke(
        input={
            "context": context,
            "query": query
        }
    )

    print(results.split("</think>")[1])


if __name__ == "__main__":
    main()
