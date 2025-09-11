from langchain_community.vectorstores import FAISS
from config import Config

from logger.logger_config import logger

class RetrieverHandler:
    """
    A class to handle the creation of a retriever from a FAISS vector store.
    """

    def __init__(self, db: FAISS):
        """
        Initializes the RetrieverHandler with a FAISS vector store.

        Args:
            db (FAISS): The FAISS vector store instance.
        """
        if not isinstance(db, FAISS):
            raise TypeError("Input 'db' must be a FAISS vector store instance.")
        self.db = db

    def get_retriever(self):
        """
        Creates and returns a retriever from the vector store.

        Returns:
            The retriever instance.
        
        Raises:
            Exception: If there is an error creating the retriever.
        """
        try:
            retriever = self.db.as_retriever(
                search_type=Config.RETRIEVER_SEARCH_TYPE,
                search_kwargs=Config.RETRIEVER_SEARCH_KWARGS
            )
            logger.info("Retriever created successfully.")
            
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
