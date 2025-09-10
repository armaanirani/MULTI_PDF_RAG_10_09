from langchain_openai import ChatOpenAI
from .. import config

def load_llm():
    """Loads an OpenAI model"""
    try:
        print("Loading LLM.")
        llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE
        )
        print("LLM loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
        # This is a critical error, so we raise it
        raise RuntimeError("Could not load the language model. Ensure the model name is correct") from e
