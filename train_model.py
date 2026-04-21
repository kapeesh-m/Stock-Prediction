# =====================================================
# BiLSTM Stock Price Prediction - Top 20 Stocks
# =====================================================

# -------------------------------
# IMPORTS
# -------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import math

# -------------------------------
# CREATE MODELS DIRECTORY
# -------------------------------
if not os.path.exists("models"):
    os.makedirs("models")

# -------------------------------
# CUSTOM LOSS FUNCTION
# -------------------------------
def directional_mse_loss(y_true, y_pred):
    true_diff = y_true[1:] - y_true[:-1]
    pred_diff = y_pred[1:] - y_true[:-1]

    penalty = tf.where(
        tf.sign(true_diff) != tf.sign(pred_diff),
        2.0,
        1.0
    )

    mse = tf.square(y_true[1:] - y_pred[1:])
    return tf.reduce_mean(mse * penalty)

# -------------------------------
# STOCK LIST (20 STRONG STOCKS)
# -------------------------------
stocks = [
    # US Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "NFLX", "AMD", "INTC",

    # US Finance & Retail
    "JPM", "BAC", "WMT", "DIS", "PYPL",

    # Indian NSE Stocks
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

# -------------------------------
# DATE RANGE
# -------------------------------
start_date = "2014-01-01"
end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

# -------------------------------
# TRAIN MODEL FOR EACH STOCK
# -------------------------------
for stock in stocks:
    print(f"\n🚀 Training model for {stock}")

    # Download stock data
    df = yf.download(stock, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"❌ No data for {stock}, skipping")
        continue

    close_prices = df["Close"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Save scaler
    joblib.dump(scaler, f"models/scaler_{stock}.pkl")

    # Train split
    train_size = math.ceil(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    # Create sequences (60 days window)
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # -------------------------------
    # BUILD BiLSTM MODEL
    # -------------------------------
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=True), input_shape=(60, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss=directional_mse_loss
    )

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    model.save(f"models/bilstm_{stock}.h5")
    print(f"✅ Model saved: models/bilstm_{stock}.h5")

print("\n🎉 Training completed for all stocks!")
