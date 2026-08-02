from vector_store import VectorStore
from loader import LoadText
from embedding_manager import EmbeddingManager


class DataIngestion:
    """
    Data Ingestion Pipeline.
    Steps:
    1. Load txt data
    2. Extract embeddings
    3. Create vector store
    4. Store embeddings in vector store
    """

    def __init__(
        self,
        data_path: str = "data",
        embeddings_model: str = "all-MiniLM-L6-v2",
        presist_directory: str = "data/vector_store",
        collection_name: str = "stt_documents",
    ):
        """
        Initialize data ingestion pipeline

        Args:
            data_path (str, optional): _description_. Defaults to "data".
            embeddings_model (str, optional): _description_. Defaults to "all-MiniLM-L6-v2".
            presist_directory (str, optional): _description_. Defaults to "data/vector_store".
            collection_name (str, optional): _description_. Defaults to "stt_documents".
        """
        self._data_loader = LoadText(path=data_path)
        self.embedding_manager = EmbeddingManager(model_name=embeddings_model)
        self.vector_store = VectorStore(
            presist_directory=presist_directory, collection_name=collection_name
        )
        self._chunks = None
        self._embeddings = None

    def ingest_data(self):
        """
        Ingest data to chromadb vector store
        """
        self._chunks = self._data_loader.split_documents()
        self._embeddings = self.embedding_manager.generate_embeddings(self._chunks)
        self.vector_store.add_documents(self._chunks, self._embeddings)
