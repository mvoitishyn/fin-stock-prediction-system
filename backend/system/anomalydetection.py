import json
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from dotenv import find_dotenv, load_dotenv
from assistants import get_assistant_response

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def load_data(ticker: str):
    df = pd.read_csv(f"../data/processed/stocksdata/{ticker}_processed.csv")
    return df


def featurize_data(df: pd.DataFrame):
    close_price = df["ClosePriceReturn"].values.reshape(-1, 1)
    volume = df["DailyVolumeChange"].values.reshape(-1, 1)
    return close_price, volume



def detect_anomalies_z(ticker: str, z_thresh: float = 3):

    df = load_data(ticker)
    df = df.dropna()

    price_mean = df["ClosePriceReturn"].mean()
    price_std = df["ClosePriceReturn"].std()
    volume_mean = df["DailyVolumeChange"].mean()
    volume_std = df["DailyVolumeChange"].std()

    df["price_z_score"] = (df["ClosePriceReturn"] - price_mean) / price_std
    df["volume_z_score"] = (df["DailyVolumeChange"] - volume_mean) / volume_std

    df["price_anomaly"] = df["price_z_score"].abs() > z_thresh
    df['volume_anomaly'] = df["volume_z_score"].abs() > z_thresh

    price_anomalies_df = df[df["price_anomaly"]][["Date"]].copy()
    price_anomalies_df["Ticker"] = ticker
    price_anomalies_df["AnomalyScore"] = df.loc[df["price_anomaly"], "price_z_score"]

    volume_anomalies_df = df[df["volume_anomaly"]][["Date"]].copy()
    volume_anomalies_df["Ticker"] = ticker
    volume_anomalies_df["AnomalyScore"] = df.loc[df["volume_anomaly"], "volume_z_score"]

    return df, price_anomalies_df, volume_anomalies_df



def plot_anomalies(price_anomalies: np.ndarray, volume_anomalies: np.ndarray, ticker: str):
    plt.figure(figsize=(12, 6))
    sns.histplot(price_anomalies, bins=50, kde=True, color='blue', label='Price Anomalies')
    plt.title(f"Distribution of Price Anomalies for {ticker}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.savefig(f"../plots/iso_forest_price_anomalies_distribution_{ticker}.png")


    plt.figure(figsize=(12, 6))
    sns.histplot(volume_anomalies, bins=50, kde=True, color='orange', label='Volume Anomalies')
    plt.title(f"Distribution of Volume Anomalies for {ticker}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.savefig(f"../plots/iso_forest_volume_anomalies_distribution_{ticker}.png")
    
    
        
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
openai_assistant_id_sentiment = "asst_vHQBv2WDnfrBHNsrsUOygUxz"
    
tickers = ["QCOM", "NVDA", "INTC", "AMD", "TSM"]

for ticker in tickers:
    df, price_anomalies_df, volume_anomalies_df = detect_anomalies_z(ticker)
            
    price_anomalies_df.to_csv(f"../data/processed/sentiment/{ticker}_price_anomalies_z_score.csv", index=False)
                
    for ticker in tickers:
        file_path = f"../data/processed/sentiment/{ticker}_price_anomalies_z_score.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as fr:
                output_data_sentiment = json.load(fr)
        else:
            output_data_sentiment = []

        # iterate over anomalies dataframe
        for index, row in price_anomalies_df.iterrows():
            date = row["Date"]
                    
            message_content = json.dumps({"ticker": ticker, "date": date})
            print(message_content)
                    
            response = get_assistant_response(OPENAI_API_KEY, openai_assistant_id_sentiment, message_content)
            
            output_data_sentiment.append(response)
                    
            print(f"Response: {response}")
                    
            with open(file_path, "w") as fw:
                json.dump(output_data_sentiment, fw)