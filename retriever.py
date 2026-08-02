from embedding_manager import EmbeddingManager
from vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.0) -> list:
        try:
            query_embeddings = self.embedding_manager.generate_embeddings([query])[0]
        except Exception as e:
            raise ValueError(f"Error generating embeddings for query: {e}")
        try:
            query_result = self.vector_store.collection.query(
                query_embeddings=[query_embeddings.tolist()], n_results=top_k
            )
        except Exception as e:
            raise ValueError(f"Error retrieving relavant documents: {e}")

        results = []
        if (
            query_result["documents"]
            and query_result["documents"][0]
            and query_result["metadatas"]
            and query_result["metadatas"][0]
            and query_result["distances"]
            and query_result["distances"][0]
        ):
            documents = query_result["documents"][0]
            metadatas = query_result["metadatas"][0]
            distances = query_result["distances"][0]
            ids = query_result["ids"][0]
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                similarity_score = 2 - distance
                if similarity_score >= score_threshold:
                    results.append(
                        {
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1,
                        }
                    )
        else:
            print("No documents found")
        return results
