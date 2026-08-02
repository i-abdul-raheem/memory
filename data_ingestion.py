from vector_store import VectorStore
from embedding_manager import EmbeddingManager
from loader import LoadText


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
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        data_path: str = "data",
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
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store
        self._chunks = None
        self._embeddings = None

    def ingest_data(self):
        """
        Ingest data to chromadb vector store
        """
        self._chunks = self._data_loader.split_documents()
        self._embeddings = self._embedding_manager.generate_embeddings(self._chunks)
        self._vector_store.add_documents(self._chunks, self._embeddings)
