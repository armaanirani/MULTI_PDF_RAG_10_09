import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document

from logger.logger_config import logger

class DocumentProcessor:
    """
    A class to handle loading and processing of documents.
    """

    def __init__(self, source_dir: str):
        """
        Initializes the DocumentProcessor with the source directory.

        Args:
            source_dir (str): The path to the directory containing PDF files.
        """
        self.source_dir = source_dir

    def load_documents(self) -> List[Document]:
        """
        Loads all PDF files from the source directory.

        Returns:
            List[Document]: A list of loaded document objects from LangChain.
            Returns an empty list if the directory doesn't exist or an error occurs.
        """
        documents = []
        if not os.path.exists(self.source_dir):
            logger.error(f"Error: Directory not found at {self.source_dir}")
            return documents
        
        logger.info(f"Loading documents from: {self.source_dir}")
        for file in os.listdir(self.source_dir):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(self.source_dir, file)
                try:
                    loader = PyMuPDFLoader(pdf_path)
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"Error loading {pdf_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} document pages in total.")
        
        return documents

    def get_text_chunks(self, documents: List[Document]) -> List[Document]:
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
            logger.info(f"Split documents into {len(chunks)} chunks.")
            
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            
            return []
