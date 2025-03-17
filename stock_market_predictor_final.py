import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# Get today's date
today = pd.Timestamp(datetime.today().date())

# Function to get valid stock data
def get_valid_stock_data():
    while True:
        ticker = input("Enter a stock ticker (e.g., MSFT, AAPL, TSLA): ").upper()
        print(f"Fetching stock data for {ticker}...")

        try:
            stock_data = yf.download(ticker, start="2010-01-01", end=today.strftime('%Y-%m-%d'))

            if stock_data.empty:
                print(f"⚠️ Error: No data found for {ticker}. Please check the symbol and try again.")
            else:
                return ticker, stock_data

        except Exception as e:
            print(f"⚠️ Error: Failed to fetch data. {e}")
            print("Please check your internet connection and try again.")

# Get valid stock data
ticker, stock_data = get_valid_stock_data()

# Convert stock index to Pandas datetime format
stock_data["Date"] = pd.to_datetime(stock_data.index)

# Define dynamic training and testing periods
train_end_date = today - pd.Timedelta(days=180)  # Train up to 6 months before today
test_end_date = today  # Test on last 6 months

# Compute moving averages (new feature)
stock_data["MA_10"] = stock_data["Close"].rolling(window=10).mean()  # 10-day moving average
stock_data["MA_50"] = stock_data["Close"].rolling(window=50).mean()  # 50-day moving average
stock_data["Daily_Return"] = stock_data["Close"].pct_change()  # Daily percentage change

# Drop NaN values caused by moving average calculation
stock_data.dropna(inplace=True)

# Convert date to ordinal (numeric format)
stock_data["Date_Ordinal"] = stock_data["Date"].map(datetime.toordinal)

# Ensure Pandas date columns are in Timestamp format
train_end_date = pd.Timestamp(train_end_date)
test_end_date = pd.Timestamp(test_end_date)

# Split the dataset
train_data = stock_data[stock_data["Date"] <= train_end_date]
test_data = stock_data[(stock_data["Date"] > train_end_date) & (stock_data["Date"] <= test_end_date)]

# Select features and target
features = ["Date_Ordinal", "MA_10", "MA_50", "Daily_Return"]
X_train = train_data[features]
y_train = train_data["Close"].values.reshape(-1, )

X_test = test_data[features]
y_test = test_data["Close"].values.reshape(-1, )

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Train a Linear Regression Model instead of Random Forest**
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Model evaluation on last 6 months
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Ask the user for a future date
while True:
    future_date_input = input("Enter a future date to predict (YYYY-MM-DD): ")
    try:
        future_date = pd.Timestamp(datetime.strptime(future_date_input, "%Y-%m-%d"))
        if future_date > today:
            break
        else:
            print("⚠️ Error: Please enter a date in the future.")
    except ValueError:
        print("⚠️ Error: Invalid date format. Please use YYYY-MM-DD.")

# **Estimate future moving averages using recent trends**
ma_10_trend = train_data["MA_10"].pct_change().mean()
ma_50_trend = train_data["MA_50"].pct_change().mean()
daily_return_avg = train_data["Daily_Return"].mean()

# Number of days into the future
days_ahead = (future_date - today).days

# Predict future moving averages
future_ma_10 = train_data["MA_10"].iloc[-1] * (1 + ma_10_trend) ** days_ahead
future_ma_50 = train_data["MA_50"].iloc[-1] * (1 + ma_50_trend) ** days_ahead
future_daily_return = daily_return_avg  

# Create feature set for future prediction
future_features = pd.DataFrame({
    "Date_Ordinal": [future_date.toordinal()],
    "MA_10": [future_ma_10],
    "MA_50": [future_ma_50],
    "Daily_Return": [future_daily_return]
})

future_scaled = scaler.transform(future_features)
predicted_price = model.predict(future_scaled)[0]

# Print model evaluation metrics
print(f"\nModel Performance on Last 6 Months:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"Predicted stock price for {ticker} on {future_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f}")

# Plot historical vs predicted data
plt.figure(figsize=(10, 5))
plt.scatter(train_data["Date"], y_train, label="Training Data (Until 6 months ago)", color="blue", alpha=0.5)
plt.scatter(test_data["Date"], y_test, label="Actual Prices (Last 6 months)", color="green", alpha=0.5)
plt.scatter(test_data["Date"], y_pred, label="Predicted Prices (Last 6 months)", color="orange", alpha=0.5)
plt.scatter(future_date, predicted_price, label=f"Predicted Price ({future_date.strftime('%Y-%m-%d')})", color="red", marker="X", s=200)

# Format x-axis to show readable dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Stock Price Prediction for {ticker}")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
