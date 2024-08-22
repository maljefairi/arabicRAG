# embedding_generator.py
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import setup_logger
from config import Config

logger = setup_logger('embedding_generator')

def generate_embeddings(documents):
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    embeddings = []
    for i in tqdm(range(0, len(documents), Config.BATCH_SIZE), desc="Generating embeddings"):
        batch = documents['content'][i:i+Config.BATCH_SIZE].tolist()
        try:
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
    return np.array(embeddings)