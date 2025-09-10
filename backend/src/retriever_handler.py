from langchain_community.vectorstores import FAISS
from config import RETRIEVER_SEARCH_KWARGS, RETRIEVER_SEARCH_TYPE

from logger.logger_config import logger

def get_retriever(db):
    
    if not isinstance(db, FAISS):
        raise TypeError("Input 'db' must be a FAISS vector store instance.")
    try:
        retriever = db.as_retriever(
            search_type=RETRIEVER_SEARCH_TYPE,
            search_kwargs=RETRIEVER_SEARCH_KWARGS
        )
        logger.info("Retriever created successfully.")
        
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise