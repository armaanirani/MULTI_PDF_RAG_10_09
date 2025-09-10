import logging

# Define the format for the log messages
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logger():
    """
    Sets up a centralized logger that writes to a file.
    """
    # Create a logger instance
    logger = logging.getLogger("rag_app_logger")
    logger.setLevel(logging.INFO)  # Set the minimum level of messages to log

    # Prevent adding multiple handlers if the logger is already configured
    if logger.hasHandlers():
        return logger

    # Create a handler to write log messages to a file
    file_handler = logging.FileHandler("rag_app.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Add handler to the logger
    logger.addHandler(file_handler)

    return logger

# Create a logger instance to be imported by other modules
logger = setup_logger()