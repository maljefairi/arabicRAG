# config.py
import os

class Config:
    DOCUMENT_FOLDER = os.environ.get('DOCUMENT_FOLDER', 'path/to/your/documents')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    LLM_MODEL = os.environ.get('LLM_MODEL', 'CAMeL-Lab/bert-base-arabic-camelbert-ca')
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
    TOP_K = int(os.environ.get('TOP_K', 5))
    MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 1024))