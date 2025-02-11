import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VectorDatabase:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", dim: int = 384, storage_path: str = "vector_db"):
        """
        VectorDatabase: A simple FAISS-based vector store with persistent storage.
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.index = faiss.IndexFlatL2(dim)
        self.data = []  # Stores raw text data
        self.storage_path = storage_path

        # Load existing data
        self.load_from_disk()

    def add_data(self, texts: List[str]) -> None:
        """
        Adds text data to FAISS and stores it persistently.
        Args:
            texts (List[str]): A list of textual data to store.
        """
        if not texts:
            return

        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype("float32"))
        self.data.extend([{"text": text} for text in texts])
        self.save_to_disk()  # Save after adding data

    def save_to_disk(self) -> None:
        """
        Saves the FAISS index and stored texts to disk.
        """
        # Save text data as JSON
        with open(f"{self.storage_path}.json", "w") as f:
            json.dump(self.data, f)

        # Save FAISS index
        faiss.write_index(self.index, f"{self.storage_path}.faiss")

    def load_from_disk(self) -> None:
        """
        Loads FAISS index and text data from disk.
        """
        try:
            # Load text data
            with open(f"{self.storage_path}.json", "r") as f:
                self.data = json.load(f)

            # Load FAISS index if data exists
            self.index = faiss.read_index(f"{self.storage_path}.faiss") if self.data else self.index
        except FileNotFoundError:
            pass  # No stored data yet

    def view_index(self) -> List[Dict]:
        """
        Retrieves stored texts with FAISS indices.
        """
        return [{"index": i, "text": self.data[i]["text"]} for i in range(self.index.ntotal)]
