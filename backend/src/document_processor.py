import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document

def load_documents(source_dir: str) -> List[Document]:
    """
    Loads all PDF files from a given source directory.

    Args:
        source_dir (str): The path to the directory containing PDF files.

    Returns:
        List[Document]: A list of loaded document objects from LangChain.
        Returns an empty list if the directory doesn't exist or an error occurs.
    """
    documents = []
    if not os.path.exists(source_dir):
        print(f"Error: Directory not found at {source_dir}")
        return documents
    
    print(f"Loading documents from: {source_dir}")
    for file in os.listdir(source_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(source_dir, file)
            try:
                loader = PyMuPDFLoader(pdf_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")
    
    print(f"Loaded {len(documents)} document pages in total.")
    
    return documents

def get_text_chunks(documents: List[Document]) -> List[Document]:
    """
    Splits the loaded documents into smaller, manageable text chunks.

    Args:
        documents (List[Document]): A list of document objects.

    Returns:
        List[Document]: A list of text chunks (also as LangChain Document objects).
        Returns an empty list if the input is empty or an error occurs.
    """
    if not documents:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        
        return []