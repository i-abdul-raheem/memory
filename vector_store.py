import os
import uuid

import chromadb
from chromadb.api.types import Metadata
import numpy as np

from constants import PRESIST_DIRECTORY, COLLECTION_NAME


class VectorStore:
    """
    Manage documents embeddings in a vector database
    """

    def __init__(
        self,
        presist_directory: str = PRESIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
    ):
        """

        Args:
            presist_directory (str, optional): Directory to store database. Defaults to "data/vector_store".
            collection_name (str, optional): Collection name for database. Defaults to "stt_documents".
        """
        self.collection_name = collection_name
        self.presist_directory = presist_directory
        self.client = None
        self._init_store()

    def _init_store(self):
        """
        Initialize vector store
        """
        try:
            os.makedirs(self.presist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.presist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SST embeddings for memory"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: list, embeddings: np.ndarray):
        """
        Add document(s) to vector store

        Args:
            documents (list): List of all documents to add
            embeddings (list): List of all embeddings of documents to add
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Length of documents list should be equal to the length of embeddings list"
            )

        ids: list[str] = []
        metadatas: list[Metadata] = []
        documents_text: list[str] = []
        embeddings_list: list[np.ndarray] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Append ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Append text and embeddings
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

            # prepare and add metadata
            metadata = dict(doc.metadata)
            metadata["content_length"] = len(doc.page_content)
            metadata["doc_index"] = i
            metadatas.append(metadata)

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"Successfully added {len(documents)} to vector store.")
        except Exception as e:
            print(f"Error adding documents: {e}")
