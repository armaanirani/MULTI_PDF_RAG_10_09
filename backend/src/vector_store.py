from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain.docstore.document import Document
from dotenv import load_dotenv
from config import Config

from logger.logger_config import logger

load_dotenv()

class VectorStore:
    """
    A class to handle the creation, loading, and updating of a FAISS vector store.
    """

    def __init__(self, embedding_model_name=Config.EMBEDDING_MODEL_NAME):
        """
        Initializes the VectorStore with the embedding model name.

        Args:
            embedding_model_name (str): The name of the OpenAI embedding model to use.
        """
        self.embedding_model_name = embedding_model_name
        self.embeddings = self._get_embeddings_model()

    def _get_embeddings_model(self):
        """
        Initializes and returns the OpenAI embedding model.
        """
        logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        return OpenAIEmbeddings(model=self.embedding_model_name)

    def create_index(self, text_chunks: List[Document], save_path: str):
        """
        Creates a new FAISS vector store and saves it to the specified path.

        Args:
            text_chunks (List[Document]): A list of text chunks to be embedded.
            save_path (str): The file path where the FAISS index will be saved.
        """
        if not text_chunks:
            logger.info("No text chunks provided to create vector store.")
            return
        try:
            logger.info("Creating vector store from documents...")
            vectorstore = FAISS.from_documents(documents=text_chunks, embedding=self.embeddings)
            
            logger.info(f"Saving vector store to: {save_path}")
            vectorstore.save_local(save_path)
            
            logger.info("Vector store saved successfully.")
        except Exception as e:
            logger.error(f"Error creating and saving vector store: {e}")

    def load_index(self, load_path: str):
        """
        Loads an existing FAISS vector store from the specified path.

        Args:
            load_path (str): The file path from where the FAISS index will be loaded.

        Returns:
            FAISS: The loaded FAISS vector store object.
        """
        try:
            logger.info(f"loading vector store from: {load_path}")
            vectorstore = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
            
            logger.info("Vector store loaded successfully")
            
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def update_index(self, new_text_chunks: List[Document], index_path: str):
        """
        Loads an existing FAISS index, adds new documents, and saves it back.
        
        Args:
            new_text_chunks (List[Document]): The new document chunks to add.
            index_path (str): The path to the existing FAISS index.
        """
        if not new_text_chunks:
            logger.info("No new text chunks to add.")
            return
        try:
            logger.info("Loading existing index to update.")
            db = self.load_index(index_path)
            
            logger.info(f"Adding {len(new_text_chunks)} new chunks to the index.")
            db.add_documents(new_text_chunks)
            
            logger.info(f"Saving updated index back to {index_path}.")
            db.save_local(index_path)
            
            logger.info("Index updated and saved successfully.")
        except Exception as e:
            logger.error(f"Failed to update FAISS index: {e}")
            
            logger.info("Falling back to creating a new index from the new chunks.")
            self.create_index(new_text_chunks, index_path)
