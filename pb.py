

import yfinance as yf
import pandas as pd
from datetime import timedelta
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_tz_aware(index, target_tz='UTC'):
    if index.tz is None:
        return index.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
    elif index.tz != target_tz:
        return index.tz_convert(target_tz)
    return index

def find_closest_date(date, date_list):
    return min(date_list, key=lambda d: abs(d - date))

def calculate_pb_ratio_change(symbol, years=5):
    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - timedelta(days=years * 365)
    
    stock = yf.Ticker(symbol)
    
    # Fetch quarterly financial data
    balance_sheet = stock.quarterly_balance_sheet
    
    possible_equity_names = ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity']
    book_value = None
    for name in possible_equity_names:
        if name in balance_sheet.index:
            book_value = balance_sheet.loc[name]
            break
    
    if book_value is None:
        logging.error(f"Could not find equity data for {symbol}. Available fields are:")
        logging.error(balance_sheet.index)
        return
    
    book_value.index = ensure_tz_aware(book_value.index)
    book_value = pd.to_numeric(book_value, errors='coerce')

    # Log book value data for debugging
    logging.info("Book Value Data:\n%s", book_value)

    # Fetch historical price data
    hist_data = stock.history(start=start_date, end=end_date)
    quarterly_prices = hist_data['Close'].resample('Q').last()
    quarterly_prices.index = ensure_tz_aware(quarterly_prices.index)

    # Log quarterly prices for debugging
    logging.info("Quarterly Prices Data:\n%s", quarterly_prices)

    # Align dates
    aligned_data = pd.DataFrame(index=book_value.index)
    aligned_data['Book_Value'] = book_value
    aligned_data['Price'] = [quarterly_prices.loc[find_closest_date(date, quarterly_prices.index)] for date in book_value.index]

    aligned_data['Price'] = pd.to_numeric(aligned_data['Price'], errors='coerce')
    aligned_data.dropna(subset=['Book_Value', 'Price'], inplace=True)
    
    if aligned_data.empty:
        logging.error(f"No overlapping data found for {symbol}.")
        return

    # Log aligned data before P/B calculation
    logging.info("Aligned Data Before P/B Calculation:\n%s", aligned_data)

    # Calculate P/B Ratio with adjustment
    aligned_data['PB_Ratio'] = aligned_data['Price'] / (aligned_data['Book_Value'] / 1e9)  # Divide book value by 1 billion

    # Log updated P/B Ratio for clarity
    logging.info("Updated P/B Ratio Values:\n%s", aligned_data[['Book_Value', 'Price', 'PB_Ratio']])
    
    # Calculate the P/B Ratio Change
    aligned_data['PB_Ratio_Change'] = aligned_data['PB_Ratio'].diff().round(4)

    def get_quarter(date):
        return f"Q{(date.month - 1) // 3 + 1} {date.year}"
    
    aligned_data['Quarter'] = aligned_data.index.to_series().apply(get_quarter)
    
    # Formatting output for readability
    pb_ratio_output = aligned_data[['Quarter', 'PB_Ratio', 'PB_Ratio_Change']]
    pb_ratio_output.loc[:, 'PB_Ratio'] = pb_ratio_output['PB_Ratio'].round(4)
    pb_ratio_output.loc[:, 'PB_Ratio_Change'] = pb_ratio_output['PB_Ratio_Change'].round(4)

    # Sort the pb_ratio_output by date to ensure chronological order
    pb_ratio_output = pb_ratio_output.sort_index()

    logging.info(f"P/B Ratio and Change by Quarter for {symbol}")
    print(pb_ratio_output.to_string(index=False))

# Example usage
symbol = "AAPL"  # Apple Inc.
calculate_pb_ratio_change(symbol)
