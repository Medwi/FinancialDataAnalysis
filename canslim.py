



import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
import yfinance as yf

# Fetch data from MacroTrends
def fetch_macrotrends_data(ticker, data_type):
    url = f'https://www.macrotrends.net/stocks/charts/{ticker}/nvidia/{data_type}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        if 'No data found' in response.text:
            print(f"No data found for {ticker} on {data_type}.")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = pd.read_html(StringIO(response.text))
        
        if not tables:
            print(f"No tables found for {ticker} on {data_type}.")
            return None
        
        df = tables[0]
        return df
    except Exception as e:
        print(f"Error fetching data from URL {url}: {e}")
        return None

# Calculate EPS Growth Rate
def calculate_eps_growth_rate(df):
    try:
        df['NVIDIA Annual EPS.1'] = df['NVIDIA Annual EPS.1'].replace('[\$,]', '', regex=True).astype(float)
        df = df.sort_values(by='NVIDIA Annual EPS', ascending=False)
        
        if len(df) >= 2:
            curr_eps = df.iloc[0]['NVIDIA Annual EPS.1']
            prev_eps = df.iloc[1]['NVIDIA Annual EPS.1']
            growth_rate = (curr_eps - prev_eps) / prev_eps * 100
            return round(growth_rate, 2)
        else:
            return None
    except Exception as e:
        print(f"Error calculating EPS growth rate: {e}")
        return None

# Check Return on Equity (ROE)
def check_roe(df):
    try:
        df[('NVIDIA ROE - Return on Equity Historical Data', 'Return on Equity')] = df[('NVIDIA ROE - Return on Equity Historical Data', 'Return on Equity')].replace('[\%,]', '', regex=True).astype(float)
        recent_roe = df.iloc[0][('NVIDIA ROE - Return on Equity Historical Data', 'Return on Equity')]
        return recent_roe
    except Exception as e:
        print(f"Error checking ROE: {e}")
        return None

# Fetch volume data using yfinance
def fetch_volume_data(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="5y")
    
    avg_volume_5y = hist_data['Volume'].mean()
    recent_volume = hist_data['Volume'].iloc[-1]
    
    return avg_volume_5y, recent_volume

# Analyze supply and demand based on volume
def analyze_supply_and_demand(avg_volume_5y, recent_volume):
    volume_increase_percentage = ((recent_volume - avg_volume_5y) / avg_volume_5y) * 100
    
    return volume_increase_percentage

# Fetch sector data
def fetch_sector_data(ticker):
    stock = yf.Ticker(ticker)
    sector = stock.info.get('sector', 'Unknown')
    return sector

# Fetch sector index data
def fetch_sector_index_data(sector):
    sector_indices = {
        'Technology': ['XLC', 'VGT', 'IYW', 'QQQ'],
        # Add more sectors and their indices as needed
    }
    return sector_indices.get(sector, [])

# Fetch stock return
def fetch_stock_return(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period='1y')
    if hist_data.empty:
        return None
    start_price = hist_data['Close'].iloc[0]
    end_price = hist_data['Close'].iloc[-1]
    return ((end_price - start_price) / start_price) * 100

# Fetch sector index return
def fetch_sector_index_return(index):
    sector_index = yf.Ticker(index)
    hist_data = sector_index.history(period='1y')
    if hist_data.empty:
        return None
    start_price = hist_data['Close'].iloc[0]
    end_price = hist_data['Close'].iloc[-1]
    return ((end_price - start_price) / start_price) * 100

# Analyze sector vs stock strength
def analyze_leader_laggard(ticker):
    sector = fetch_sector_data(ticker)
    indices = fetch_sector_index_data(sector)
    
    if not indices:
        print(f"No sector indices available for sector: {sector}")
        return False

    stock_return = fetch_stock_return(ticker)
    if stock_return is None:
        print(f"Failed to retrieve stock return for {ticker}")
        return False
    
    is_leader = True
    
    for index in indices:
        sector_return = fetch_sector_index_return(index)
        if sector_return is not None:
            if stock_return <= sector_return:
                is_leader = False
    
    return is_leader

# Fetch market data
def fetch_market_data(ticker):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="6mo")  # Get the last 6 months of data
    return hist_data

def analyze_market_direction(df):
    # Calculate Moving Averages
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    
    # Determine Market Direction
    if df['Close'].iloc[-1] > df['50_MA'].iloc[-1] and df['50_MA'].iloc[-1] > df['200_MA'].iloc[-1]:
        return "Confirmed Uptrend"
    elif df['Close'].iloc[-1] < df['50_MA'].iloc[-1] and df['50_MA'].iloc[-1] < df['200_MA'].iloc[-1]:
        return "Market in Correction"
    else:
        return "Uptrend Under Pressure"

# Stock Screening Function
def screen_stock(ticker):
    data_types = ['eps-earnings-per-share-diluted', 'roe', 'revenue']

    all_data = {}

    for data_type in data_types:
        print(f"Fetching data from URL: https://www.macrotrends.net/stocks/charts/{ticker}/nvidia/{data_type}")
        df = fetch_macrotrends_data(ticker, data_type)
        
        if df is not None:
            print(f"Fetched {len(df)} rows of data for {data_type} from ticker {ticker}.")
            all_data[data_type] = df
        else:
            print(f"Failed to fetch data for {data_type}.")
    
    # Process Annual EPS Growth Rate
    eps_growth_rate = None
    if 'eps-earnings-per-share-diluted' in all_data:
        eps_df = all_data['eps-earnings-per-share-diluted']
        eps_growth_rate = calculate_eps_growth_rate(eps_df)
    
    # Process ROE
    recent_roe = None
    if 'roe' in all_data:
        roe_df = all_data['roe']
        recent_roe = check_roe(roe_df)
    
    # Analyze supply and demand via volume
    avg_volume_5y, recent_volume = fetch_volume_data(ticker)
    volume_increase_percentage = analyze_supply_and_demand(avg_volume_5y, recent_volume)
    
    # Analyze sector vs stock strength
    is_leader = analyze_leader_laggard(ticker)
    
    # Fetch and analyze market direction
    market_indices = ['^GSPC', '^IXIC']
    market_directions = {}
    
    for index in market_indices:
        print(f"Fetching market data for {index}...")
        df = fetch_market_data(index)
        direction = analyze_market_direction(df)
        market_directions[index] = direction
    
    # Check criteria
    meets_eps_criteria = eps_growth_rate is not None and eps_growth_rate >= 25
    meets_roe_criteria = recent_roe is not None and recent_roe >= 17
    meets_volume_criteria = volume_increase_percentage >= 5
    market_uptrend = all(direction == "Confirmed Uptrend" for direction in market_directions.values())
    
    # Determine if the stock qualifies
    if meets_eps_criteria and meets_roe_criteria and meets_volume_criteria and is_leader and market_uptrend:
        print(f"{ticker} qualifies based on CANSLIM criteria.")
    else:
        print(f"{ticker} does not qualify based on CANSLIM criteria.")

if __name__ == "__main__":
    stock_to_screen = 'NVDA'
    screen_stock(stock_to_screen)
