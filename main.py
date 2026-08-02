from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import Retriever
from data_ingestion import DataIngestion

PRESIST_DIRECTORY: str = "data/vector_store"
COLLECTION_NAME: str = "stt_documents"
MODEL_NAME: str = "all-MiniLM-L6-v2"


def main():
    vector_store = VectorStore(
        presist_directory=PRESIST_DIRECTORY, collection_name=COLLECTION_NAME
    )
    embedding_manager = EmbeddingManager(model_name=MODEL_NAME)
    # data_ingestion = DataIngestion(
    #     vector_store=vector_store,
    #     embedding_manager=embedding_manager
    # )
    # data_ingestion.ingest_data()
    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )
    results = retriever.retrieve(
        query="It can understand multiple languages and works well even with background"
    )
    
    print(results)


if __name__ == "__main__":
    main()
