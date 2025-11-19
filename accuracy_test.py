import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load cached CSV from the correct path
filename = "stock_cache/ITC.NS_5y.csv"

# Load and clean the data
df = pd.read_csv(filename, index_col=0, parse_dates=True)
df = df.rename(columns={df.columns[0]: "Close"})  # Ensure consistent column naming
df = df.reset_index().rename(columns={"index": "Date"})

# Prepare data for Prophet
df_prophet = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

# Split into 90% train, 10% test
train_size = int(len(df_prophet) * 0.9)
train = df_prophet.iloc[:train_size]
test = df_prophet.iloc[train_size:]

# Train Prophet model
model = Prophet(daily_seasonality=True)
model.fit(train)

# Predict future dates equal to test set
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Merge forecast with test data on 'ds' to avoid indexing errors
forecast_df = forecast[["ds", "yhat"]]
merged = pd.merge(test, forecast_df, on="ds", how="inner")

# Accuracy Evaluation
actual = merged["y"].values
predicted = merged["yhat"].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Display Results
print(f"\n📊 Prophet Accuracy Metrics:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAPE = {mape:.2f}%")
