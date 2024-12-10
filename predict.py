import pandas as pd
from model import load_model, load_encoder, predict
import logging

# Configure logging
LOG_FILE = "prediction_log.log"
logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

def get_user_input():
    """Simulate user input (can be replaced by actual user input in your application)."""
    log_info("Getting user input...")
    user_input = {'ticker': ['AAPL', 'GOOG', 'MSFT']}  # Example tickers
    user_input_df = pd.DataFrame(user_input)
    log_info(f"User input: {user_input}")
    return user_input_df

def main():
    # Load model and encoder
    model = load_model()
    encoder = load_encoder()

    # Get user input (for prediction)
    user_input_df = get_user_input()

    # Make prediction
    prediction = predict(model, encoder, user_input_df)

    # Output the prediction
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
