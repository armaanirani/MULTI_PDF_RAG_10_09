import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src import document_processor, qa_handler, vector_store, retriever_handler

# Define the path for storing the vector index
VECTOR_STORE_PATH = "../data/faiss_index"
# Define the path for temporary file uploads
UPLOAD_DIR = "../data/uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup events.
    This context manager ensures that necessary directories are created
    when the application starts.
    """
    print("Lifespan startup: Creating necessary directories...")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    print("Lifespan startup: Directories are ready.")
    yield


app = FastAPI(
    title="Multi-PDF RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Question(BaseModel):
    """Pydantic model for a user's question."""
    query: str

class Answer(BaseModel):
    """Pydantic model for the generated answer."""
    answer: str
    source_documents: List[dict]

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

def process_documents_task(upload_dir: str):
    """
    Background task to process uploaded PDF documents with enhanced logging.
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
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"3. Found existing vector store. Updating index at {VECTOR_STORE_PATH}...")
            vector_store.update_index(text_chunks, VECTOR_STORE_PATH)
            print("   Successfully updated existing vector store.")
        else:
            print(f"3. No existing vector store found. Creating new index at {VECTOR_STORE_PATH}...")
            vector_store.create_index(text_chunks, VECTOR_STORE_PATH)
            print("   Successfully created and saved new vector store.")

        print("--- Document processing task completed successfully! ---")
    except Exception as e:
        # Log the full traceback for detailed debugging
        import traceback
        print(f"!!!!!! ERROR in background task !!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    finally:
        # Clean up uploaded files after processing
        print("4. Cleaning up uploaded files...")
        for filename in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, filename))
        print(f"   Cleaned up upload directory: {upload_dir}")
        print("--- Background task finished ---")

@app.post("/upload/", status_code=202)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Endpoint to upload PDF files and trigger processing in the background.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    # Save uploaded files to a temporary directory
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved file: {file.filename}")

    # Add the processing task to run in the background
    background_tasks.add_task(process_documents_task, UPLOAD_DIR)

    return {"message": f"Started processing {len(files)} files. This may take a moment."}

@app.post("/ask/", response_model=Answer)
async def ask_question(question: Question):
    """
    Endpoint to ask a question and get an answer from the RAG chain.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Please upload documents first."
        )

    try:
        print(f"Received query: {question.query}")
        db = vector_store.load_index(VECTOR_STORE_PATH)
        retriever = retriever_handler.get_retriever(db)
        qa_chain = qa_handler.create_qa_chain(retriever)
        
        result = qa_chain.invoke(question.query)
        
        answer = result.get("answer", "No answer found.")
        source_docs = result.get("context", [])

        return {
            "answer": answer,
            "source_documents": [
                {"source": doc.metadata.get("source", "N/A"), "content": doc.page_content}
                for doc in source_docs
            ],
        }
    except Exception as e:
        print(f"Error during question answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"status": "ok"}