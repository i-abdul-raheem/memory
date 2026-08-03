from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import Retriever
from data_ingestion import DataIngestion

from constants import PRESIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def main():
    vector_store = VectorStore(
        presist_directory=PRESIST_DIRECTORY, collection_name=COLLECTION_NAME
    )
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )
    results = retriever.retrieve(
        query="what can this app do?"
    )
    
    print(results)


if __name__ == "__main__":
    main()
