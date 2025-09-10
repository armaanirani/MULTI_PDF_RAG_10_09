from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain.docstore.document import Document
from dotenv import load_dotenv
from config import EMBEDDING_MODEL_NAME

from logger.logger_config import logger

load_dotenv()

def _get_embeddings_model():
    """Initializes and returns the OpenAI embedding model"""
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )

def create_index(text_chunks: List[Document], save_path: str):
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
        embeddings = _get_embeddings_model()
        logger.info("Creating vector store from documents...")
        vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        
        logger.info(f"Saving vector store to: {save_path}")
        vectorstore.save_local(save_path)
        
        logger.info("Vector store saved successfully.")
    except Exception as e:
        logger.error(f"Error creating and saving vector store: {e}")

def load_index(load_path: str):
    """
    Loads an existing FAISS vector store from the specified path.

    Args:
        load_path (str): The file path from where the FAISS index will be loaded.

    Returns:
        FAISS: The loaded FAISS vector store object.
    """
    try:
        embeddings = _get_embeddings_model()
        
        logger.info(f"loading vector store from: {load_path}")
        vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        
        logger.info("Vector store loaded successfully")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise

def update_index(new_text_chunks: List[Document], index_path: str):
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
        db = load_index(index_path)
        
        logger.info(f"Adding {len(new_text_chunks)} new chunks to the index.")
        db.add_documents(new_text_chunks)
        
        logger.info(f"Saving updated index back to {index_path}.")
        db.save_local(index_path)
        
        logger.info("Index updated and saved successfully.")
    except Exception as e:
        logger.error(f"Failed to update FAISS index: {e}")
        
        logger.info("Falling back to creating a new index from the new chunks.")
        create_index(new_text_chunks, index_path)        