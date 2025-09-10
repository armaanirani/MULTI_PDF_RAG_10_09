from langchain_community.vectorstores import FAISS
from .. import config

def get_retriever(db):
    
    if not isinstance(db, FAISS):
        raise TypeError("Input 'db' must be a FAISS vector store instance.")
    try:
        retriever = db.as_retriever(
            search_type=config.RETRIEVER_SEARCH_TYPE,
            search_kwargs=config.RETRIEVER_SEARCH_KWARGS
        )
        print("Retriever created successfully.")
        
        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        raise