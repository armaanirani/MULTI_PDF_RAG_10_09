from langchain_openai import ChatOpenAI
from config import LLM_MODEL_NAME, LLM_TEMPERATURE

from logger.logger_config import logger

def load_llm():
    """Loads an OpenAI model"""
    try:
        logger.info("Loading LLM.")
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )
        logger.info("LLM loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        # This is a critical error, so we raise it
        raise RuntimeError("Could not load the language model. Ensure the model name is correct") from e
