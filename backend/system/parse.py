import pandas as pd
import requests
import os
import yfinance as yf
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

FRED_API_KEY = os.getenv("FRED_API_KEY")

nasqaq_tickers = ["NVDA","INTC", "AMD", "QCOM"]
nyse_tickers = ["TSM"]

macro_series = {
    "ticker": ["CPIAUCSL", "FEDFUNDS", "GDP", "UNRATE", "UMCSENT", "DCOILWTICO", "PSAVERT"],
    "frequency": ["m", "m", "q", "m", "m", "d", "m"]
      
  }

# get US macroeconomic data from FRED
def get_fred_data(series_id, start_date=None, end_date=None, frequency=None):
    params = {
        'series_id': series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "frequency": frequency,
        'api_key': FRED_API_KEY,
        'file_type': "json"
    }
    url = 'https://api.stlouisfed.org/fred/series/observations'
    response = requests.get(url, params=params)
    data = pd.DataFrame(response.json()["observations"])
    data.to_csv(f"../data/raw/macrodata/{series_id}.csv", index=False)

for series, freq in zip(macro_series["ticker"], macro_series["frequency"]):
    get_fred_data(series, start_date="2000-01-01", end_date="2018-01-01", frequency=freq)

# # get stocks data from nasdaq
for ticker_symbol in nasqaq_tickers:
    paststockdf = pd.read_csv(f"../data/raw/stocksdata/stocksformykyta/NASDAQ_{ticker_symbol}_30Y.csv")
    paststockdf.to_csv(f"../data/raw/parsed/{ticker_symbol}.csv")
# get stock data from nyse
for ticker_symbol in nyse_tickers:
    paststockdf = pd.read_csv(f"../data/raw/stocksdata/stocksformykyta/NYSE_{ticker_symbol}_30Y.csv")
    paststockdf.to_csv(f"../data/raw/parsed/{ticker_symbol}.csv")
    
  
# tickers_from_yfinance = ["SPY", "^VIX"]
                        
# for ticker in tickers_from_yfinance:
#     data = yf.Ticker(ticker)
#     data_hist = data.history(start="2000-01-01", end="2017-12-31", interval="1d")
#     data_df = pd.DataFrame(data_hist)
#     data_df.to_csv(f"data/raw/parsed/{ticker}.csv")
