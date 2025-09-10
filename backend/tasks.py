import os
import shutil
import traceback

from src import document_processor, vector_store

def process_documents_task(upload_dir: str, vector_store_path: str):
    """
    Background task to process uploaded PDF documents with enhanced logging.

    Args:
        upload_dir (str): The directory containing the uploaded files to process.
        vector_store_path (str): The path to the FAISS vector store index.
    """
    print("--- Starting document processing background task ---")
    try:
        print(f"1. Loading documents from: {upload_dir}")
        raw_documents = document_processor.load_documents(upload_dir)
        if not raw_documents:
            print("No documents found to process. Exiting task.")
            return
        print(f"   Successfully loaded {len(raw_documents)} document(s).")

        print("2. Splitting documents into text chunks...")
        text_chunks = document_processor.get_text_chunks(raw_documents)
        if not text_chunks:
            print("Failed to create text chunks. Exiting task.")
            return
        print(f"   Successfully created {len(text_chunks)} text chunks.")

        # Check if an index already exists
        if os.path.exists(vector_store_path):
            print(f"3. Found existing vector store. Updating index at {vector_store_path}...")
            vector_store.update_index(text_chunks, vector_store_path)
            print("   Successfully updated existing vector store.")
        else:
            print(f"3. No existing vector store found. Creating new index at {vector_store_path}...")
            vector_store.create_index(text_chunks, vector_store_path)
            print("   Successfully created and saved new vector store.")

        print("--- Document processing task completed successfully! ---")
    except Exception as e:
        print(f"!!!!!! ERROR in background task !!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    finally:
        # Clean up uploaded files after processing
        print("4. Cleaning up uploaded files...")
        # A check to ensure the directory exists before trying to clean it up
        if os.path.isdir(upload_dir):
            shutil.rmtree(upload_dir)
            print(f"   Successfully cleaned up and removed directory: {upload_dir}")
        print("--- Background task finished ---")


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