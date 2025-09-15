import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_klines_batch(symbol, interval, start_time_ms, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_all_klines(symbol="BTCUSDT", interval="5m", start_date="2017-08-17"):
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(time.time() * 1000)

    all_klines = []
    print("The downloading process has been initiated")

    while start_time < end_time:
        try:
            batch = get_klines_batch(symbol, interval, start_time)
            if not batch:
                break

            all_klines.extend(batch)
            
            start_time = batch[-1][0] + (5 * 60 * 1000)  # next 5m after last open_time
            print("1000 candles batched is done.")
            time.sleep(0.2)

        except Exception as e:
            print(f"Error: {e}. Retrying in 3s...")
            time.sleep(3)

    print(f"\n✅ Finished downloading {len(all_klines)} klines.")
    return all_klines

def klines_to_df(klines):
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    float_cols = ["open", "high", "low", "close", "volume"]
    df[float_cols] = df[float_cols].astype(float)
    return df.drop(columns="ignore")

klines = fetch_all_klines()
df = klines_to_df(klines)

output_path = "../data/crypto/binance/btcusdt_5min_full.csv"
df.to_csv(output_path, index=False)
print(f"📁 Data saved to {output_path}")
