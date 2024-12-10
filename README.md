# Stock Market Prediction

This repository contains a machine learning model for predicting stock market trends based on historical data. The project uses a Random Forest Regressor to predict stock price movements.

## Project Overview

This project aims to predict the direction of stock prices (up or down) based on historical stock data. It leverages machine learning techniques, including data preprocessing, model training, and evaluation, to generate predictions that can assist in making informed stock market decisions.

### Key Features:
- Predict stock price movements (increase or decrease).
- Use historical stock market data to train the model.
- Data preprocessing steps such as encoding categorical features and handling missing values.
- Model evaluation using metrics like Mean Squared Error (MSE) and R-squared.

## Dataset

The dataset used in this project is provided in the `dataset` folder. It contains historical stock market data, which includes:
- **Stock Ticker Symbol**
- **Stock Price**
- **Volume**
- **Date**
- **Target column** (price movement: up or down)

The dataset is large and is tracked using Git Large File Storage (Git LFS). You can download the dataset if necessary by following the [Git LFS installation guide](https://git-lfs.github.com/).

**Dataset path:** `dataset/infolimpioavanzadoTarget.csv`

## Requirements

The following libraries are required to run this project:

- pandas
- numpy
- scikit-learn
- joblib
- git-lfs

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
