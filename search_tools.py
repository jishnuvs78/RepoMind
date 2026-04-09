from sentence_transformers import SentenceTransformer
from typing import List, Any
import numpy as np

# Cache embedding model at module level to load only once
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    return _embedding_model

class SearchTool:
    def __init__(
        self,
        pytorch_img_index,
        pytorch_img_vindex,
    ):
        self.pytorch_img_index = pytorch_img_index
        self.pytorch_img_vindex = pytorch_img_vindex
        self.embedding_model = get_embedding_model()

    def text_search(self, query: str) -> List[Any]:
        """
        Search the repository using text-based search.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of relevant document chunks from the repository.
        """
        return self.pytorch_img_index.search(query, num_results=5)

    def vector_search(self, query: str) -> List[Any]:
        """
        Search the repository using semantic (vector) search.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of semantically relevant document chunks.
        """
        q = self.embedding_model.encode(query)
        return self.pytorch_img_vindex.search(q, num_results=5)

    def search(self, query: str) -> List[Any]:
        """
        Perform hybrid search on the repository.
        Combines text-based and semantic (vector) search for best results.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A combined list of relevant results.
        """
        text_results = self.text_search(query)
        vector_results = self.vector_search(query)

        seen_ids = set()
        combined_results = []

        for result in text_results + vector_results:
            if result['filename'] not in seen_ids:
                seen_ids.add(result['filename'])
                combined_results.append(result)

        return combined_results