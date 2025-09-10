from langchain_openai import ChatOpenAI

# Configuration for the LLM
LLM_MODEL_NAME = "gpt-5"
LLM_TEMPERATURE = 0.3

def load_llm():
    """Loads an OpenAI model"""
    try:
        print("Loading LLM.")
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )
        print("LLM loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
        # This is a critical error, so we raise it
        raise RuntimeError("Could not load the language model. Ensure the model name is correct") from e
