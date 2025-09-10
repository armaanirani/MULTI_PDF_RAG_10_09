from langchain_community.vectorstores import FAISS

# Configuration for the Retriever
RETRIEVER_SEARCH_TYPE = "similarity"
# Number of source documents to retrieve
RETRIEVER_SEARCH_KWARGS = {"k": 3}

def get_retriever(db):
    
    if not isinstance(db, FAISS):
        raise TypeError("Input 'db' must be a FAISS vector store instance.")
    try:
        retriever = db.as_retriever(
            search_type=RETRIEVER_SEARCH_TYPE,
            search_kwargs=RETRIEVER_SEARCH_KWARGS
        )
        print("Retriever created successfully.")
        
        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}")
        raise