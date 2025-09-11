from langchain_openai import ChatOpenAI
from config import Config

from logger.logger_config import logger

class LLM:
    """
    A class to handle the loading of the language model.
    """

    def __init__(self, model_name=Config.LLM_MODEL_NAME, temperature=Config.LLM_TEMPERATURE):
        """
        Initializes the LLM with the model name and temperature.

        Args:
            model_name (str): The name of the OpenAI model to use.
            temperature (float): The temperature for the model's output.
        """
        self.model_name = model_name
        self.temperature = temperature

    def load(self):
        """
        Loads the OpenAI model.

        Returns:
            ChatOpenAI: The loaded language model instance.
        
        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        try:
            logger.info("Loading LLM.")
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature
            )
            logger.info("LLM loaded successfully.")
            return llm
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            # This is a critical error, so we raise it
            raise RuntimeError("Could not load the language model. Ensure the model name is correct") from e
