


import yfinance as yf
from datetime import datetime, timedelta
import json

def download_historical_data(symbol):
    """
    Download historical stock price data from Yahoo Finance.
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def remove_nan_values(data):
    """
    Remove rows with NaN values from the DataFrame.
    """
    data.dropna(inplace=True)
    return data

def calculate_returns(data):
    """
    Calculate 1-month, 3-month, 6-month, and 1-year returns using the 'Close' column.
    """
    if 'Close' not in data.columns:
        raise ValueError("Data does not contain 'Close' prices.")
        
    # Use only the 'Close' column for return calculations
    data['1m_return'] = data['Close'].pct_change(periods=20) * 100
    data['3m_return'] = data['Close'].pct_change(periods=60) * 100
    data['6m_return'] = data['Close'].pct_change(periods=120) * 100
    data['1y_return'] = data['Close'].pct_change(periods=240) * 100
    return data

def calculate_momentum_percentile(data):
    """
    Calculate percentile rank for each momentum factor.
    """
    percentiles = {}
    for momentum_factor in ['1m_return', '3m_return', '6m_return', '1y_return']:
        sorted_data = sorted(data[momentum_factor].dropna())
        current_value = data[momentum_factor].iloc[-1]  # Get the last value
        if len(sorted_data) == 0:
            percentile = 0
        else:
            percentile = (sorted_data.index(current_value) / len(sorted_data)) * 100
        percentiles[momentum_factor] = percentile
    return percentiles

def calculate_hqm_score(percentiles):
    """
    Calculate HQM Score based on percentile values.
    """
    score = sum(percentiles.values()) / len(percentiles)
    return score

def fetch_stock_data(symbol, period="1y"):
    """
    Fetch stock data for the given symbol.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data['Close']

def calculate_absolute_momentum(data, lookback=252):
    """
    Calculate absolute momentum as the percentage return over a given lookback period.
    """
    if len(data) < lookback:
        return None  # Not enough data
    return (data.iloc[-1] / data.iloc[-lookback] - 1) * 100

def calculate_relative_momentum(stock_data, benchmark_data, lookback=252):
    """
    Calculate relative momentum compared to a benchmark.
    """
    if len(stock_data) < lookback or len(benchmark_data) < lookback:
        return None  # Not enough data
    stock_return = (stock_data.iloc[-1] / stock_data.iloc[-lookback] - 1) * 100
    benchmark_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[-lookback] - 1) * 100
    return stock_return - benchmark_return

def check_dual_momentum(stock_data, benchmark_data, lookback=252):
    """
    Check if stock has both absolute and relative momentum greater than 40%.
    """
    abs_momentum = calculate_absolute_momentum(stock_data, lookback)
    rel_momentum = calculate_relative_momentum(stock_data, benchmark_data, lookback)
    
    if abs_momentum is not None and rel_momentum is not None:
        print(f"Absolute Momentum: {abs_momentum:.2f}%")
        print(f"Relative Momentum: {rel_momentum:.2f}%")
        return abs_momentum > 50 and rel_momentum > 50
    return False

def read_symbols_from_file(filepath):
    """
    Read stock symbols from a JSON file. The JSON file contains categories of stocks,
    so this function aggregates them into a flat list of symbols.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Aggregate symbols from all categories into a flat list
    symbols = []
    for category, stock_list in data.items():
        symbols.extend(stock_list)
    
    print(f"First 5 stock symbols from JSON file: {symbols[:5]}")
    return symbols

def process_stock(symbol, data):
    """
    Process the stock data and calculate the HQM score.
    """
    data = remove_nan_values(data)
    data = calculate_returns(data)
    momentum_percentiles = calculate_momentum_percentile(data)
    hqm_score = calculate_hqm_score(momentum_percentiles)
    
    # Print the individual percentiles and HQM score for debugging
    print(f"Percentiles for {symbol}: {momentum_percentiles}")
    print(f"HQM Score for {symbol}: {hqm_score:.2f}")
    
    return symbol, hqm_score, data

def screen_stocks(symbols, benchmark_symbol="^IXIC"):
    """
    Screen a list of stock symbols and print those with HQM score of 75 or more,
    then check dual momentum for qualifying stocks.
    """
    qualifying_stocks = []
    benchmark_data = fetch_stock_data(benchmark_symbol, period="1y")
    
    # First, filter stocks based on HQM score
    for symbol in symbols:
        print(f"Processing {symbol}...")
        try:
            data = download_historical_data(symbol)
            symbol, hqm_score, processed_data = process_stock(symbol, data)
            
            if hqm_score >= 75:
                qualifying_stocks.append(symbol)
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")
    
    print("\nStocks with HQM Score of 75 or more:")
    if qualifying_stocks:
        for symbol in qualifying_stocks:
            print(f"{symbol}")
    else:
        print("None")
    
    # Now check dual momentum for the stocks that qualified based on HQM score
    dual_momentum_stocks = []
    for symbol in qualifying_stocks:
        print(f"\nCalculating dual momentum for {symbol}...")
        stock_data = fetch_stock_data(symbol, period="1y")
        if check_dual_momentum(stock_data, benchmark_data):
            dual_momentum_stocks.append(symbol)
    
    print("\nStocks with HQM Score of 75 or more and dual momentum > 50%:")
    if dual_momentum_stocks:
        for symbol in dual_momentum_stocks:
            print(f"{symbol}")
    else:
        print("None")

if __name__ == "__main__":
    file_path = "/Users/davidsanchomarco/miniconda3/envs/analysis/Analizer-Predictor/stock_indices.json"
    symbols_to_screen = read_symbols_from_file(file_path)
    screen_stocks(symbols_to_screen)
