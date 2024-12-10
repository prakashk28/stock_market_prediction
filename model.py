import pickle
import logging

# Configure logging
LOG_FILE = "model_log.log"
logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

def load_model():
    """Load the pre-trained model from a file."""
    log_info("Loading model...")
    model_path = 'stock_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    log_info("Model loaded successfully.")
    return model

def load_encoder():
    """Load the pre-trained encoder from a file."""
    log_info("Loading encoder...")
    encoder_path = 'ticker_encoder.pkl'
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)
    log_info("Encoder loaded successfully.")
    return encoder

def predict(model, encoder, input_data):
    """Make a prediction using the trained model."""
    log_info("Making prediction...")
    
    # Ensure the 'ticker' column is transformed before prediction
    input_data['ticker'] = encoder.transform(input_data['ticker'].astype(str))
    
    # Predict using the model
    prediction = model.predict(input_data)
    
    log_info(f"Prediction: {prediction}")
    return prediction
