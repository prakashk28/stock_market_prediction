import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Define the log directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

# Create a log file name with a timestamp
LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Set up the logging configuration
def setup_logger():
    # Create a logger object
    logger = logging.getLogger()

    # Prevent logging from being added multiple times
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create a rotating file handler that will back up the log file once it exceeds 5MB
        file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler for live output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter to specify the log message format
        log_format = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

        # Set the formatter for the file and console handlers
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Create a logger instance
logger = setup_logger()

# Example usage of the logger
if __name__ == "__main__":
    logger.info("Starting the application.")
    try:
        logger.info("Running some process...")
        # Example code to simulate a task
        result = 10 / 2
        logger.info(f"Process completed successfully with result: {result}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
