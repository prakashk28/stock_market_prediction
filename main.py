import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
from logger import setup_logger

# Setup logger
logger = setup_logger()

# Load data
def load_data(file_path):
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    logger.info("Data loaded successfully.")
    return df

# Preprocess data
def preprocess_data(df):
    logger.info("Preprocessing data...")
    
    # Convert date to datetime if it's not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values (fill with 0 or use imputation)
    df.fillna(0, inplace=True)  # For simplicity, fill missing values with 0. Consider imputation strategies
    
    # Encode categorical columns (e.g., 'ticker')
    label_encoder = LabelEncoder()
    df['ticker'] = label_encoder.fit_transform(df['ticker'])
    
    logger.info(f"Non-numeric columns: {df.select_dtypes(exclude=[np.number]).columns}")
    
    # Drop the date column after feature extraction
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['dayofyear'] = df['date'].dt.dayofyear
    df.drop(columns=['date'], inplace=True)
    
    # Check for infinite values and large numbers in X
    X = df.drop(columns=['TARGET'])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    X.fillna(0, inplace=True)  # Replace NaN with 0 or use a different imputation strategy
    
    # Normalize or scale the features if necessary
    # You can use StandardScaler or MinMaxScaler here if your data has large values
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    logger.info("Preprocessing completed.")
    return X, df['TARGET'], label_encoder

# Train the model
def train_model(X_train, y_train):
    logger.info("Training the model...")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    logger.info("Evaluating the model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate performance using MSE, MAE, and R^2
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    logger.info(f"Training MSE: {train_mse}")
    logger.info(f"Test MSE: {test_mse}")
    logger.info(f"Training MAE: {train_mae}")
    logger.info(f"Test MAE: {test_mae}")
    logger.info(f"Training R^2: {train_r2}")
    logger.info(f"Test R^2: {test_r2}")

# Save the model and encoder
def save_model_and_encoder(model, encoder):
    logger.info("Saving model and encoder...")
    joblib.dump(model, 'stock_model.pkl')
    joblib.dump(encoder, 'ticker_encoder.pkl')
    logger.info("Model and encoder saved successfully.")

def main():
    file_path = 'D:\Personal Projects\stock_market_prediction\dataset\infolimpioavanzadoTarget.csv'
    df = load_data(file_path)
    
    # Check for data leakage
    assert 'TARGET' in df.columns, "TARGET column not found"
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(df)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Save model and encoder
    save_model_and_encoder(model, label_encoder)
    
    logger.info("Pipeline completed.")

if __name__ == "__main__":
    main()
