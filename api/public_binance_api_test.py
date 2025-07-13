import requests
import json
import pandas as pd 


# testing connection
try:
    response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
    response.raise_for_status()
    data = json.loads(response.text)
    
    
    print("Connected successfully")
    
    print(f"Symbols available: {len(data["symbols"])}")
    print(f"Server time: {data["serverTime"]}")

except requests.exceptions.RequestException as e:
    print(f"Connnection failed: {e}")


# test sample data
try:
    klines_response = requests.get("https://api.binance.com/api/v3/klines",
                                    params = {
                                        "symbol": "BTCUSDT",
                                        "interval": "5m",
                                        "limit": 1000
                                    })
    
    klines_response.raise_for_status()
    klines_data = json.loads(klines_response.text)
    
    print(f"BTC 5-minute data fetched: {len(klines_data)} candles")
    print(f"Latest row: ${float(klines_data[-1][4])}")
    
    klines_df = pd.DataFrame(klines_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])    
    
    klines_df["open_time"] = pd.to_datetime(klines_df["open_time"], unit="ms")
    klines_df["close_time"] = pd.to_datetime(klines_df["close_time"], unit="ms")
    klines_df[["open", "high", "low", "close", "volume"]] = klines_df[["open", "high", "low", "close", "volume"]].astype(float)
    
    klines_df = klines_df.drop(columns="ignore")
    
    klines_df.to_csv("../data/crypto/binance/btcusdt5min1000_sample.csv")

except requests.exceptions.RequestException as e:
    print(f"Failed to fetch BTC data: {e}")    