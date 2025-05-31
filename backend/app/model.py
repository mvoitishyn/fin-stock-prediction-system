import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(ticker: str):
    df = pd.read_csv(f"../data/processed/stocksdata/{ticker}_processed.csv")
    return df

def featurize_data(
    df: pd.DataFrame, 
    use_baseline_data, 
    use_macro_data,
    use_extra_data,
    target_window = 1
):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    features = []

    if use_baseline_data:
        raw_features = ["Date", "Open", "High", "Low", "Volume"]
        features += [col for col in raw_features if col in df.columns]

    if use_macro_data:
        macro_keywords = ["interpolated_DCOILWTICO", "interpolated_CPIAUCSL", "interpolated_FEDFUNDS", "interpolated_GDP", "interpolated_PSAVERT", "interpolated_UMCSENT", "interpolated_UNRATE"]
        features += [col for col in df.columns if any(key in col for key in macro_keywords)]

    if use_extra_data:
        rolling_keywords = ["SMA", "EMA", "RSI", "ROC", "Bollinger", "Std", "DailyVolumeChange", "IntradayVolatility", "OpenCloseRange", "ExtremeClosePriceDailyReturn_IQR", "ExtremeVolumeDailyChangeReturn_IQR", "Year", "Month", "Day", "DayNumeric", "Sentiment", "anomaly_score_price", "anomaly_score_volume"]
        features += [col for col in df.columns if any(key in col for key in rolling_keywords)]

    features = list(set(features))
    X = df[features]
    
    if target_window == 1:
        y = df[["ClosePriceReturn"]]
    elif target_window == 3:
        y = df[["CloseReturnPrice_window_3"]]
        y = np.vstack(y["CloseReturnPrice_window_3"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')))
    elif target_window == 5:
        y = df[["CloseReturnPrice_window_5"]]
        y = np.vstack(y["CloseReturnPrice_window_5"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')))
    elif target_window == 0:
        y = df[["Close"]]
        
    mask_train = df["Date"].dt.year < 2013
    mask_test = ~mask_train

    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]

    return X_train, X_test, y_train, y_test

def featurize_data_for_prediction(
    df: pd.DataFrame, 
    use_baseline_data, 
    use_macro_data,
    use_extra_data
):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    df = df.tail(1)

    features = []

    if use_baseline_data:
        raw_features = ["Date", "Open", "High", "Low", "Volume"]
        features += [col for col in raw_features if col in df.columns]

    if use_macro_data:
        macro_keywords = ["interpolated_DCOILWTICO", "interpolated_CPIAUCSL", "interpolated_FEDFUNDS", "interpolated_GDP", "interpolated_PSAVERT", "interpolated_UMCSENT", "interpolated_UNRATE"]
        features += [col for col in df.columns if any(key in col for key in macro_keywords)]

    if use_extra_data:
        rolling_keywords = ["SMA", "EMA", "RSI", "ROC", "Bollinger", "Std", "DailyVolumeChange", "IntradayVolatility", "OpenCloseRange", "ExtremeClosePriceDailyReturn_IQR", "ExtremeVolumeDailyChangeReturn_IQR", "Year", "Month", "Day", "DayNumeric", "Sentiment", "anomaly_score_price", "anomaly_score_volume"]
        features += [col for col in df.columns if any(key in col for key in rolling_keywords)]

    features = list(set(features))
    X = df[features]
    
    return X

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def plot_actual_vs_predicted(y_test, y_pred, dates):
    dates = pd.to_datetime(dates)
    
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test, label="Actual", color="blue", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted", color="orange", linestyle="--", linewidth=2)
    plt.title("Actual vs. Predicted (Entire Dataset)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close(fig1)
    
    last_year_mask = dates >= (dates.max() - pd.DateOffset(years=1))
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(dates[last_year_mask], y_test[last_year_mask], label="Actual", color="blue", linewidth=2)
    plt.plot(dates[last_year_mask], y_pred[last_year_mask], label="Predicted", color="orange", linestyle="--", linewidth=2)
    plt.title("Actual vs. Predicted (Last Year)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close(fig2)
    
    last_month_mask = dates >= (dates.max() - pd.DateOffset(months=1))
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(dates[last_month_mask], y_test[last_month_mask], label="Actual", color="blue", linewidth=2)
    plt.plot(dates[last_month_mask], y_pred[last_month_mask], label="Predicted", color="orange", linestyle="--", linewidth=2)
    plt.title("Actual vs. Predicted (Last Month)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close(fig3)
    
    last_week_mask = dates >= (dates.max() - pd.DateOffset(weeks=1))
    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(dates[last_week_mask], y_test[last_week_mask], label="Actual", color="blue", linewidth=2)
    plt.plot(dates[last_week_mask], y_pred[last_week_mask], label="Predicted", color="orange", linestyle="--", linewidth=2)
    plt.title("Actual vs. Predicted (Last Week)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close(fig4)
    
    return fig1, fig2, fig3, fig4



def plot_actual_vs_predicted_bayesian(
    y_train, y_test,
    y_pred_train, y_pred_test,
    std_train, std_test,
    x_train_dates, x_test_dates, ticker
):
    x_train_dates = pd.to_datetime(x_train_dates)
    x_test_dates = pd.to_datetime(x_test_dates)

    dates = np.concatenate([x_train_dates, x_test_dates])
    y_actual = np.concatenate([y_train, y_test]).flatten()
    y_pred = np.concatenate([y_pred_train, y_pred_test]).flatten()
    std = np.concatenate([std_train, std_test]).flatten()

    split_index = len(x_train_dates)
    test_start_date = dates[split_index]

    def plot_zoomed(dates, y_actual, y_pred, std, start_date=None, title_suffix="", tick_interval="year"):
        fig = plt.figure(figsize=(12, 6))

        if start_date:
            mask = dates >= pd.to_datetime(start_date)
            dates_zoom = dates[mask]
            y_actual_zoom = y_actual[mask]
            y_pred_zoom = y_pred[mask]
            std_zoom = std[mask]
        else:
            dates_zoom = dates
            y_actual_zoom = y_actual
            y_pred_zoom = y_pred
            std_zoom = std

        upper = y_pred_zoom + std_zoom
        lower = y_pred_zoom - std_zoom
        plt.fill_between(dates_zoom, lower, upper, color='red', alpha=0.3, label="Uncertainty estimation")
        plt.plot(dates_zoom, y_actual_zoom, label="Actual", color="blue", linewidth=2)
        plt.plot(dates_zoom, y_pred_zoom, label="Predicted", color="orange", linestyle="--", linewidth=2)
        plt.axvline(test_start_date, color='red', linestyle=':', label="Train/Test Split")

        plt.title(f"Actual vs. Predicted {title_suffix} for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Close Price Return")
        plt.legend()
        plt.grid(True)

        # Set tick frequency based on zoom level
        if tick_interval == "year":
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif tick_interval == "month":
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        elif tick_interval == "week":
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        elif tick_interval == "day":
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

        plt.xticks(rotation=45)
        plt.xlim(dates_zoom[0], dates_zoom[-1])
        plt.tight_layout()
        
        return fig


    fig_full = plot_zoomed(dates, y_actual, y_pred, std, title_suffix="(Full Time Range)", tick_interval="year")
    fig_year = plot_zoomed(dates, y_actual, y_pred, std, start_date=dates[-1] - pd.DateOffset(years=1), title_suffix="(Last Year)", tick_interval="month")
    fig_month = plot_zoomed(dates, y_actual, y_pred, std, start_date=dates[-1] - pd.DateOffset(months=1), title_suffix="(Last Month)", tick_interval="week")
    fig_week = plot_zoomed(dates, y_actual, y_pred, std, start_date=dates[-1] - pd.DateOffset(weeks=1), title_suffix="(Last Week)", tick_interval="day")


    return fig_full, fig_year, fig_month, fig_week

def reconstruct_from_windows(windowed_data):
    n, w = windowed_data.shape
    original_length = n + w - 1

    result = np.zeros(original_length)
    count = np.zeros(original_length)

    for i in range(n):
        result[i:i+w] += windowed_data[i]
        count[i:i+w] += 1

    reconstructed = (result / count).reshape(-1, 1)
    return reconstructed