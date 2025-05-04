# 📈 Stock Price Prediction Pattern Using LSTM

## 📌 Introduction  
Stock market prediction is a critical area of financial research, providing valuable insights for traders and investors. This project leverages **Long Short-Term Memory (LSTM) networks**, a type of Recurrent Neural Network (RNN), to predict stock price movements based on historical data.  

## 🎯 Use Case  
The goal of this project is to develop a **predictive model** that can analyze past stock prices and forecast future price movements. This can help:  
- Investors make data-driven decisions  
- Reduce financial risks  
- Optimize trading strategies  

## 🏗️ Project Structure  
The project is organized into the following steps:  
1. **Data Collection:** Fetching stock price data from 2020 to 2023.  
2. **Data Preprocessing:** Cleaning, scaling, and preparing the data.  
3. **Model Development:** Building an LSTM-based deep learning model.  
4. **Training & Evaluation:** Training the model and measuring performance.  
5. **Prediction & Visualization:** Forecasting stock prices and visualizing results.  

## 🔄 Project Flow  
1. **Data Extraction:** The `psx-data-reader` library retrieves historical stock data.  
2. **Data Preprocessing:**  
   - Removes unnecessary features (e.g., Volume).  
   - Normalizes stock prices using **MinMaxScaler**.  
   - Converts data into time sequences for LSTM input.  
3. **Model Training:** An **LSTM network** is trained using historical price movements.  
4. **Prediction:** The trained model forecasts future stock prices.  
5. **Evaluation & Visualization:** The predicted prices are compared with actual prices.  

## 🏗️ Model Architecture  

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(100, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(50, activation="relu"),
    Dense(25, activation="relu"),
    Dense(1)  # Output layer
])
```
The **LSTM network** consists of multiple layers:  
- LSTM layers: Capture sequential patterns in stock price movements.
- Dense layers: Further refine features and improve prediction accuracy.
- Output layer: Produces the final stock price prediction.

## ⚙️ Data Processing Techniques
- Feature Engineering: Uses high and low prices to compute mid-prices.
- Scaling: Normalizes stock prices using MinMaxScaler between 0 and 1.
- Sequence Generation: Creates 58-day sequences for LSTM training.
- Train-Test Split: 80% training, 20% testing.

## 📊 Results & Model Performance Evaluation
- Loss Function: Mean Squared Error (MSE) to measure performance.
- Training Performance: The model achieves steady convergence with decreasing loss over epochs.
- Visualization: The predicted stock prices closely follow actual trends, indicating a good fit.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## 📌 Conclusion
This project successfully demonstrates how LSTM networks can effectively predict stock prices based on historical data. Future improvements may include:

1. Incorporating external factors (e.g., news sentiment, economic indicators).
2. Using attention mechanisms to enhance model interpretability.
3. Experimenting with different time windows for better performance.
