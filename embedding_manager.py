from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """_summary_
        Initialize the embedding manager

        Args:
            model_name (str, optional): _description_. Defaults to "all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            print("Loading embedding model...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Embedding model loaded with dimensions: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def generate_embeddings(self, chunks: list) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded!")
        print(f"Generating embedding for {len(chunks)} documents...")
        texts = [
            chunk.page_content for chunk in chunks
        ]
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Each chunk must be a string or a Document with page_content.")

        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("Embeddings generated")
        return embeddings
