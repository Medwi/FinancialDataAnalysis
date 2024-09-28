


import yfinance as yf
from ta.trend import MACD

def is_good_for_daytrading(symbol, macd_threshold=0.02):
    try:
        # Fetch stock data from Yahoo Finance
        stock = yf.Ticker(symbol)
        
        # Attempt to fetch intraday data (1-minute interval) for the current day
        history = stock.history(period='1d', interval='1m')
        
        # Print raw fetched data for inspection
        print(f"Fetched intraday data for {symbol}:")
        print(history.head())
        
        if history.empty:
            print(f"No intraday data found for symbol {symbol}. Checking historical data...")
            
            # Fetch historical data for the past month if intraday data is not available
            history = stock.history(period='1mo')
            
            # Print raw fetched data for inspection
            print(f"Fetched historical data for {symbol}:")
            print(history.head())
        
        if history.empty:
            print(f"No valid data found for symbol {symbol}.")
            return
        
        # Calculate MACD (Moving Average Convergence Divergence)
        indicator_macd = MACD(history['Close'], window_slow=26, window_fast=12, window_sign=9)
        history['macd'] = indicator_macd.macd()
        history['macd_signal'] = indicator_macd.macd_signal()
        
        # Ensure data alignment by dropping NaN values only from specific columns
        history.dropna(subset=['Close', 'Volume', 'macd', 'macd_signal'], inplace=True)
        
        if history.empty:
            print(f"No valid data found for symbol {symbol}.")
            return
        
        # Check volume change (today's volume vs average historical volume)
        today_volume = history['Volume'].sum()
        avg_volume = history['Volume'].mean()
        
        # Check MACD trend
        last_macd = history['macd'].iloc[-1]
        last_macd_signal = history['macd_signal'].iloc[-1]
        
        print(f"Today's volume: {today_volume:.2f}")
        print(f"Average volume: {avg_volume:.2f}")
        print(f"MACD: {last_macd:.2f}, MACD Signal: {last_macd_signal:.2f}")
    
    except Exception as e:
        print(f"Error fetching or processing data for {symbol}: {str(e)}")

# Example usage:
symbol = 'GLD'
is_good_for_daytrading(symbol, macd_threshold=0.02)
