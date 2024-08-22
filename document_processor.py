# document_processor.py
import os
import glob
from tqdm import tqdm
import pandas as pd
from utils import clean_text, setup_logger

logger = setup_logger('document_processor')

def load_documents(folder_path):
    documents = []
    for file_path in tqdm(glob.glob(os.path.join(folder_path, '*.txt')), desc="Loading documents"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = clean_text(file.read())
                documents.append({'path': file_path, 'content': content})
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    return pd.DataFrame(documents)