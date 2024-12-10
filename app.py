import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random  # For demonstration purposes, to simulate predictions

# Function to preprocess user input
def preprocess_input(user_input, encoder):
    """
    Preprocess user input to encode categorical features.

    Args:
        user_input (pd.DataFrame): DataFrame containing user input.
        encoder (LabelEncoder): Fitted LabelEncoder for categorical variables.

    Returns:
        pd.DataFrame: Preprocessed user input.
    """
    user_input['ticker'] = encoder.transform(user_input['ticker'].astype(str))
    return user_input

# Main function
def main():
    # Set up the app layout and title
    st.set_page_config(page_title="Stock Market Prediction", layout="centered")
    st.title("📈 Stock Market Prediction App")
    st.write("Welcome to the Stock Market Prediction App. Enter stock details below to get a prediction.")

    # Simulated training data
    training_data = pd.DataFrame({
        'ticker': ['AAPL', 'GOOG', 'MSFT', 'TSLA'],
        'price': [150, 2800, 299, 850],
        'volume': [1000, 2000, 1500, 1800]
    })

    # Initialize LabelEncoder and fit on the training data
    encoder = LabelEncoder()
    encoder.fit(training_data['ticker'])

    # Create a two-column layout for input
    st.sidebar.header("🔍 Enter Stock Details")
    with st.sidebar.form("stock_form"):
        ticker_input = st.text_input("Ticker Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")
        price_input = st.number_input("Stock Price:", value=150.0, format="%.2f", step=1.0)
        volume_input = st.number_input("Volume:", value=1000, step=1)
        submitted = st.form_submit_button("Predict 🚀")

    if submitted:
        # Create user input DataFrame
        user_input_df = pd.DataFrame({
            'ticker': [ticker_input],
            'price': [price_input],
            'volume': [volume_input]
        })

        # Display user input in a container
        st.subheader("📋 User Input Data")
        st.dataframe(user_input_df)

        try:
            # Preprocess user input
            processed_input = preprocess_input(user_input_df, encoder)

            # Display preprocessed input
            st.subheader("🔄 Processed Input Data")
            st.dataframe(processed_input)

            # Extract original ticker for prediction message
            original_ticker = user_input_df['ticker'][0]

            # Randomly predict gain or loss (replace with actual model logic)
            prediction_outcome = random.choice(["up", "down"])
            if prediction_outcome == "up":
                st.success(f"🎯 Prediction for **{original_ticker}**: Stock is likely to go up! 📈")
            else:
                st.warning(f"⚠️ Prediction for **{original_ticker}**: Stock is likely to go down! 📉")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Stock Market Predictor | Created with ❤️ by Data Science Enthusiasts")

# Run the main function
if __name__ == "__main__":
    main()
