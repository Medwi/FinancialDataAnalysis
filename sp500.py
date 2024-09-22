



import yfinance as yf
import pandas as pd

# Fetch the S&P 500 data
sp500 = yf.Ticker("^GSPC")

# Get historical market data for the past year
data = sp500.history(period="1y")

# Reset index to get the date as a column
data.reset_index(inplace=True)

# Save to CSV
data.to_csv("sp500_daily_data_last_year.csv", index=False)

print("S&P 500 daily price data for the last year saved to sp500_daily_data_last_year.csv")
