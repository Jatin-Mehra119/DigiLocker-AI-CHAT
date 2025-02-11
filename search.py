import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class SearchEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", dim: int = 384, storage_path: str = "vector_db"):
        """
        SearchEngine: Loads FAISS index and performs efficient text search.
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.index = faiss.IndexFlatL2(dim)
        self.data = []
        self.storage_path = storage_path
        self.load_from_disk()

    def load_from_disk(self) -> None:
        """
        Loads stored FAISS index and text data.
        """
        try:
            with open(f"{self.storage_path}.json", "r") as f:
                self.data = json.load(f)

            self.index = faiss.read_index(f"{self.storage_path}.faiss") if self.data else self.index
        except FileNotFoundError:
            print("No stored data found.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Searches stored FAISS embeddings for similar text results.
        Args:
            query (str): Query string to search.
            top_k (int): Number of results to return.
        Returns:
            List[Dict]: List of matched texts with similarity scores.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype("float32"), top_k)

        results = [
            {"text": self.data[i]["text"], "score": distances[0][j]}
            for j, i in enumerate(indices[0]) if i < len(self.data)
        ]
        return results


if __name__ == "__main__":
    search_engine = SearchEngine()
    query = input("Enter search query: ")
    results = search_engine.search(query)

    if results:
        print("\n🔍 **Search Results:**")
        for res in results:
            print(f"- {res['text']} (Score: {res['score']:.4f})")
    else:
        print("No matching results found.")
