import os
import shutil

from logger.logger_config import logger
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

def process_documents_task(upload_dir: str, vector_store_path: str):
    """
    Background task to process uploaded PDF documents with enhanced logging.

    Args:
        upload_dir (str): The directory containing the uploaded files to process.
        vector_store_path (str): The path to the FAISS vector store index.
    """
    logger.info("--- Starting document processing background task ---")
    try:
        logger.info(f"1. Loading documents from: {upload_dir}")
        doc_processor = DocumentProcessor(upload_dir)
        raw_documents = doc_processor.load_documents()
        if not raw_documents:
            logger.warning("No documents found to process. Exiting task.")
            return
        logger.info(f"Successfully loaded {len(raw_documents)} document(s).")

        logger.info("2. Splitting documents into text chunks...")
        text_chunks = doc_processor.get_text_chunks(raw_documents)
        if not text_chunks:
            logger.error("Failed to create text chunks. Exiting task.")
            return
        logger.info(f"Successfully created {len(text_chunks)} text chunks.")

        vector_store = VectorStore()
        # Check if an index already exists
        if os.path.exists(vector_store_path):
            logger.info(f"3. Found existing vector store. Updating index at {vector_store_path}...")
            vector_store.update_index(text_chunks, vector_store_path)
            logger.info("Successfully updated existing vector store.")
        else:
            logger.info(f"3. No existing vector store found. Creating new index at {vector_store_path}...")
            vector_store.create_index(text_chunks, vector_store_path)
            logger.info("Successfully created and saved new vector store.")

        logger.info("--- Document processing task completed successfully! ---")
    except Exception as e:
        logger.error(f"ERROR in background task")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Details: {e}")
        logger.error("Traceback:", exc_info=True)
    finally:
        # Clean up uploaded files after processing
        logger.info("4. Cleaning up uploaded files...")
        # A check to ensure the directory exists before trying to clean it up
        if os.path.isdir(upload_dir):
            shutil.rmtree(upload_dir)
            logger.info(f"   Successfully cleaned up and removed directory: {upload_dir}")
        logger.info("--- Background task finished ---")


# def process_documents_task(upload_dir: str):
#     """
#     Background task to process uploaded PDF documents.
#     """
#     print("Starting document processing task...")
#     try:
#         raw_documents = document_processor.load_documents(upload_dir)
#         if not raw_documents:
#             print("No documents found to process.")
#             return

#         text_chunks = document_processor.get_text_chunks(raw_documents)
#         if not text_chunks:
#             print("Failed to create text chunks.")
#             return

#         # Check if an index already exists
#         if os.path.exists(VECTOR_STORE_PATH):
#             print("Updating existing vector store...")
#             vector_store.update_index(text_chunks, VECTOR_STORE_PATH)
#         else:
#             print("Creating new vector store...")
#             vector_store.create_index(text_chunks, VECTOR_STORE_PATH)

#         print("Document processing task completed.")
#     except Exception as e:
#         print(f"Error in background task: {e}")
#     finally:
#         # Clean up uploaded files after processing
#         for filename in os.listdir(upload_dir):
#             os.remove(os.path.join(upload_dir, filename))
#         print(f"Cleaned up upload directory: {upload_dir}")
