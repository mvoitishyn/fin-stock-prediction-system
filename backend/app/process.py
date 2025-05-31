import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import json

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from scipy.special import softmax

import re

def clean_assistant_response(text):
    text = text.replace("```json\n", "").replace("\n```", "").replace("```", "").replace("%", "").replace("\n", "").replace("\\","").replace("\'","\"").replace("json","").replace("'","")
    text = text.replace("ud83dude80","").replace('"{',"{").replace('}"',"}")
    text = text.replace('"What happened to my favorite stock?"',"")
    text = text.replace('"Hold tight, or is it time to cut losses?"',"")
    text = text.replace('"This feels like a bait!"',"")
    text = text.replace('"This could set the stage for growth!"', "")
    text = text.replace('"#INTC is about to soar!" versus "Caution, folks! Are we in too deep?"', "")
    text = re.sub(r'("summary_tweet"\s*:\s*".*?)""', r'\1',text, flags=re.DOTALL)
    text = text.replace(',}',"}")
    text = text.replace('"#INTC is about to soar!" versus "Caution, folks! Are we in too deep?"', "")

    return text

def run_llm(ticker: str):
        
    with open(f"../data/processed/sentiment/{ticker}_price_anomalies_z_score.json", 'r', encoding='utf-8') as f:
        text = f.read()
    text_cleaned = clean_assistant_response(text)
    resp = json.loads(text_cleaned)
    print(f"{ticker} is successfully loaded to json object")
    with open(f"../data/processed/sentiment/{ticker}_price_anomalies_z_score_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(resp, f)
        
    resp_df = pd.read_json(f"../data/processed/sentiment/{ticker}_price_anomalies_z_score_cleaned.json")

    
    resp_df["date"] = pd.to_datetime(resp_df["date"])

    messages_twitter_anomalies = resp_df["summary_tweet"].astype(str).tolist()
    dates_twitter_anomalies = resp_df["date"].tolist()

    # load the model    
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    res_scores_list_anomalies = []
    output_scores_raw_anomalies = {}
    
    # process and rearrange anoma
    for text, date in zip(messages_twitter_anomalies, dates_twitter_anomalies):
        output_scores_sorted = {"Date": date, "Sentiment_positive_score_anomaly": 0, "Sentiment_neutral_score_anomaly": 0, "Sentiment_negative_score_anomaly": 0}

        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)[::-1]
     
        for i in range(scores.shape[0]):
            label = config.id2label[ranking[i]]
            score = np.round(float(scores[ranking[i]]), 4)
            output_scores_raw_anomalies[label] = score
            
        output_scores_sorted["Sentiment_positive_score_anomaly"] = output_scores_raw_anomalies["positive"]
        output_scores_sorted["Sentiment_neutral_score_anomaly"] = output_scores_raw_anomalies["neutral"]
        output_scores_sorted["Sentiment_negative_score_anomaly"] = output_scores_raw_anomalies["negative"]   

        res_scores_list_anomalies.append(output_scores_sorted) 

        print(output_scores_sorted)
        
    results_df = pd.DataFrame(res_scores_list_anomalies)
    results_df.to_csv(f"../data/processed/sentiment/{ticker}_twitterroberta_anomalies_scores.csv", index=False)
    
    return results_df
    
def calculate_thresholds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_threshold = q1 - 1.5 * iqr
    upper_threshold = q3 + 1.5 * iqr
    return lower_threshold, upper_threshold

def get_macro_data(path):
    file_paths = glob.glob(os.path.join(path, "*.csv"))
    macro_data = None
    data_dict = {}
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.drop(columns=[col for col in ["realtime_start", "realtime_end"] if col in df.columns], errors="ignore")
        df = df[df["value"] != "."]
        file_name = os.path.basename(file_path)
        if file_name == "DCOILWTICO.csv":
            macro_data = df.rename(columns={"date": "Date", "value": "DCOILWTICO"})
        else:
            metric = os.path.splitext(file_name)[0]
            df = df.rename(columns={"date": "Date", "value": "Value"})
            df["Metric"] = metric
            data_dict[metric] = df
    for metric, df in data_dict.items():
        temp = df[["Date", "Value"]].copy().rename(columns={"Value": metric})
        macro_data = macro_data.merge(temp, on="Date", how="left")
    interpolated = macro_data.copy()
    for col in macro_data.columns:
        if col != "Date":
            interpolated[f"interpolated_{col}"] = macro_data[col].interpolate(method="linear", limit_direction="both")
            interpolated[f"is_interpolated_{col}"] = (macro_data[col] != interpolated[f"interpolated_{col}"]).astype(int)
    drop_cols = ["DCOILWTICO", "CPIAUCSL", "FEDFUNDS", "GDP", "PSAVERT", "UMCSENT", "UNRATE"]
    interpolated.drop(columns=[c for c in drop_cols if c in interpolated.columns], inplace=True)
    return interpolated

def detect_anomalies(df, contamination: float = 0.05) -> pd.DataFrame:
    
    close_price = df["ClosePriceReturn"].values.reshape(-1, 1)
    volume = df["DailyVolumeChange"].values.reshape(-1, 1)


    iso_forest_price = IsolationForest(random_state=0, contamination=contamination)
    iso_forest_volume = IsolationForest(random_state=0, contamination=contamination)
    
    iso_forest_price.fit(close_price)
    iso_forest_volume.fit(volume)
    
    resdf = pd.DataFrame()
    resdf["anomaly_score_price"] = iso_forest_price.decision_function(close_price)
    resdf["anomaly_score_volume"] = iso_forest_volume.decision_function(volume)
    resdf["Date"] = df["Date"]
    
    return resdf

def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_dataset(ticker, min_date, max_date):
    file_path = f"../data/raw/parsed/{ticker}.csv"
    raw_data = pd.read_csv(file_path)
    df = raw_data.copy()
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Date"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    
    # time features
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Day"] = pd.to_datetime(df["Date"]).dt.day
    df["DayNumeric"] = pd.to_datetime(df["Date"]).dt.weekday + 1
    df = df[(pd.to_datetime(df["Date"]) >= pd.to_datetime(min_date)) & (pd.to_datetime(df["Date"]) <= pd.to_datetime(max_date))].sort_values(by="Date").reset_index(drop=True)
    
    # percentage changes
    df["ClosePriceReturn"] = df["Close"].pct_change()
    df["DailyVolumeChange"] = df["Volume"].pct_change()
    
    # intraday features
    df["IntradayVolatility"] = (df["High"] - df["Low"]) / df["Close"]
    df["OpenCloseRange"] = df["Open"] - df["Close"]
    
    # rolling window features + technical indicators
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"ClosePriceSMA{n}"] = df["Close"].rolling(window=n).mean()
        df[f"ClosePriceEMA{n}"] = df["Close"].ewm(span=n, adjust=False).mean()
        df[f"ClosePriceStd{n}"] = df["ClosePriceReturn"].rolling(window=n).std()
        df[f"VolumeSMA{n}"] = df["Volume"].rolling(window=n).mean()
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"RSI{n}"] = compute_rsi(df["Close"], n)
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"ClosePriceROC{n}"] = df["Close"].pct_change(periods=n)
    for n in [5, 10, 20, 50, 100, 200]:
        rolling_mean = df["Close"].rolling(window=n).mean()
        rolling_std = df["Close"].rolling(window=n).std()
        df[f"Bollinger_Mean_{n}"] = rolling_mean
        df[f"Bollinger_Upper_{n}"] = rolling_mean + 2 * rolling_std
        df[f"Bollinger_Lower_{n}"] = rolling_mean - 2 * rolling_std
    

    df.dropna(inplace=True)
    
    # calculate sliding windows for target variable: 1, 3 or 5 days
    close_prices = df["ClosePriceReturn"].values
    n = len(close_prices)

    window_3 = [close_prices[i:i+3] for i in range(n - 3 + 1)]
    window_5 = [close_prices[i:i+5] for i in range(n - 5 + 1)]

    df = df.iloc[:len(window_3)]
    df["CloseReturnPrice_window_3"] = window_3

    df = df.iloc[:len(window_5)]
    df["CloseReturnPrice_window_5"] = window_5

    # anomaly features: IQR + Isolation Forest  
    lower_thr_close, upper_thr_close = calculate_thresholds(df["ClosePriceReturn"])
    lower_thr_vol, upper_thr_vol = calculate_thresholds(df["DailyVolumeChange"])
    df["ExtremeClosePriceDailyReturn_IQR"] = np.where((df["ClosePriceReturn"] < lower_thr_close) | (df["ClosePriceReturn"] > upper_thr_close), 1, 0)
    df["ExtremeVolumeDailyChangeReturn_IQR"] = np.where((df["ClosePriceReturn"] < lower_thr_vol) | (df["DailyVolumeChange"] > upper_thr_vol), 1, 0)
    
    
    iso_nomalies = detect_anomalies(df)
    df = pd.merge(df, iso_nomalies, on="Date", how="inner")  

    # macro features
    macro_df = get_macro_data("../data/raw/macrodata")
    macro_df.to_csv("../data/processed/macrodata/macrodata_processed.csv", index=False)
    df["Date"] = pd.to_datetime(df["Date"])
    macro_df["Date"] = pd.to_datetime(macro_df["Date"])
    merged_df = pd.merge(df, macro_df, on="Date", how="left")
    merged_df.dropna(inplace=True)
    
    # sentiment data based on Z_score
    anomaly_scores = run_llm(ticker)
    merged_df = pd.merge(merged_df, anomaly_scores, on = "Date", how="left")
    resdf = merged_df.fillna(0)
          
    return resdf