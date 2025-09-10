# Configuration for the LLM
LLM_MODEL_NAME = "gpt-5"
LLM_TEMPERATURE = 0.3

# Configuration for the embedding model 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Configuration for the Retriever
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_SEARCH_KWARGS = {"k": 3}  # Number of source documents to retrieve

# Path for storing the vector index
VECTOR_STORE_PATH = "../data/faiss_index"
# Path for temporary file uploads
UPLOAD_DIR = "../data/uploads"