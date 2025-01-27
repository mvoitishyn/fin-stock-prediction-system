import pandas as pd
import requests
import os
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

FRED_API_KEY = os.getenv("FRED_API_KEY")

# initialization
nasdaq_symbols = ["NVDA", "AMZN", "AVGO", "AAPL", "TSLA", "MSFT"]
nyse_symbols = ["LMT", "MDT", "IBM", "BA", "NOC","GD"]
macro_series_ids = ["FEDFUNDS", "CPIAUCSL", "GDP", "UNRATE"]

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
    data.to_csv(f"data/raw/macrodata/{series_id}.csv", index=False)

for series in macro_series_ids:
    get_fred_data(series, start_date="2000-01-01", end_date="2024-01-01",frequency="q")

for ticker_symbol in nasdaq_symbols:
    paststockdf = pd.read_csv(f"data/raw/stockdata/NASDAQ_{ticker_symbol}_30Y.csv")
    paststockdf.to_csv(f"data/processed/stockdata/{ticker_symbol}.csv")

for ticker_symbol in nyse_symbols:
    paststockdf = pd.read_csv(f"data/raw/stockdata/NYSE_{ticker_symbol}_30Y.csv")
    paststockdf.to_csv(f"data/processed/stockdata/{ticker_symbol}.csv")