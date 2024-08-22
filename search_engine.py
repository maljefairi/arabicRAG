# search_engine.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import setup_logger
from config import Config

logger = setup_logger('search_engine')

class SearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.index = self._build_faiss_index(embeddings)
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def _build_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def search(self, query):
        try:
            query_embedding = self.model.encode([query])
            _, indices = self.index.search(query_embedding.astype('float32'), Config.TOP_K)
            return self.documents.iloc[indices[0]]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return pd.DataFrame()

