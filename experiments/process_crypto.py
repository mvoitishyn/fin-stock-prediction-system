import pandas as pd
import numpy as np



btcusdt_test = pd.read_csv("../data/crypto/binance/btcusdt5min1000_sample.csv")

def caclulate_intraday_range(df):
    df["intra_day_range"] = df["high"] - df["low"]
    return df


btcusdt_test = caclulate_intraday_range(btcusdt_test)


intra_arr = np.array(btcusdt_test["intra_day_range"])


length_intervals = btcusdt_test.shape[0]
interval = 5
position = 0

for i in range(20):
    print(intra_arr[position:position+interval])
    position += 1
    
    
