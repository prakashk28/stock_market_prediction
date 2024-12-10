import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(user_input, encoder):
    """Preprocess the user input (for predictions)."""
    if isinstance(user_input['ticker'], list):
        user_input['ticker'] = pd.Series(user_input['ticker'])

    # Transform the 'ticker' column using the encoder
    try:
        user_input['ticker'] = encoder.transform(user_input['ticker'].astype(str))
    except ValueError as e:
        print(f"Error during ticker transformation: {e}")
        raise
    
    return user_input
