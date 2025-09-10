import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src import qa_handler, vector_store, retriever_handler
import config
from tasks import process_documents_task
from logger.logger_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup events.
    This context manager ensures that necessary directories are created
    when the application starts.
    """
    logger.info("Lifespan startup: Creating necessary directories...")
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.VECTOR_STORE_PATH), exist_ok=True)
    logger.info("Lifespan startup: Directories are ready.")
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
        file_path = os.path.join(config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved file: {file.filename}")

    # Add the processing task to run in the background
    background_tasks.add_task(process_documents_task, config.UPLOAD_DIR, config.VECTOR_STORE_PATH)

    return {"message": f"Started processing {len(files)} files. This may take a moment."}

@app.post("/ask/", response_model=Answer)
async def ask_question(question: Question):
    """
    Endpoint to ask a question and get an answer from the RAG chain.
    """
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Please upload documents first."
        )

    try:
        logger.info(f"Received query: {question.query}")
        db = vector_store.load_index(config.VECTOR_STORE_PATH)
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