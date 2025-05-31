import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from model import load_data

import mlflow

raw_close_price = True
train_arima = True
train_sarima = True

ticker = "QCOM"
raw_df = load_data(ticker)

raw_df["Date"] = pd.to_datetime(raw_df["Date"])
year = pd.to_datetime("2013-01-01")

train_df = raw_df[raw_df["Date"] < year]
test_df = raw_df[raw_df["Date"] >= year]

if raw_close_price == True:
    train_returns = train_df["Close"]
    test_returns = test_df["Close"]
else:
    train_returns = train_df["ClosePriceReturn"]
    test_returns = test_df["ClosePriceReturn"]

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

mlflow.set_experiment("ARIMA")

results_arima = []
    
if train_arima:
    
    with mlflow.start_run():
        if True:
            
            p_values = [5, 10, 20] 
            d_values = [1, 2]
            q_values = [2, 3, 4] 

            best_mse = np.inf
            best_order = None
            best_model = None

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        arima_order = (p, d, q)
                        params = {
                            "ticker": ticker,  
                            "p": p,
                            "d": d,
                            "q": q
                        }

                    
                        model = ARIMA(train_returns, order=arima_order).fit(method_kwargs={"maxiter": 10000})
                        forecast = model.forecast(steps=len(test_returns))
                            
                        mae = mean_absolute_error(test_returns, forecast)
                        mse = mean_squared_error(test_returns, forecast)
                        r2 = r2_score(test_returns, forecast)

                        print(f"ARIMA{arima_order} - Test MSE: {mse:.4f} - Test R2: {r2:.4f}")

                        mlflow.log_params(params)
                        mlflow.log_metric("Test MAE", mae)
                        mlflow.log_metric("Test MSE", mse)
                        mlflow.log_metric("Test R2", r2)

                        if mse < best_mse:
                            best_mse = mse
                            best_order = arima_order
                            best_model = model
                            
                        results = {
                            "model": "ARIMA",
                            "Order": arima_order,
                            "Test MAE": mae,
                            "Test MSE": mse,
                            "Test R2": r2
                        }
                        results_arima.append(results)


        mlflow.end_run()
    
    print(f"\nBest ARIMA Model: ARIMA{best_order} with Test MSE: {best_mse:.4f}")
    
    results_arima_df = pd.DataFrame(results_arima)
    results_arima_df.to_csv("results/arima_results.csv", index=False)

    # Best ARIMA Model: ARIMA(10, 0, 2) with Test MSE: 86.9016
    
    arima_forecast = best_model.forecast(steps=len(test_returns))

    arima_mse = mean_squared_error(test_returns, arima_forecast)
    arima_r2 = r2_score(test_returns, arima_forecast)

    print(f"ARIMA Test MSE: {arima_mse:.4f}")
    print(f"ARIMA Test R2: {arima_r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_returns, label="Train", color="blue")
    plt.plot(test_df["Date"], test_returns, label="Test", color="green")
    plt.plot(test_df["Date"], arima_forecast, label=f"ARIMA{best_order} Forecast", color="red")
    plt.title(f"ARIMA Model: Train, Test, and Forecast (Best Order: {best_order})")
    plt.xlabel("Date")
    plt.ylabel("Close Price Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/statisticalmodels/arima_best_model_forecast.png", dpi=300)
    plt.show()

# find best SARIMA Model
mlflow.set_experiment("SARIMA")

if train_sarima:
    
    results_sarima = []

    best_mse = np.inf
    best_order = None
    best_seasonal_order = None
    
    p_values = [2, 10] 
    d_values = [1, 2]
    q_values = [1, 3]
    P_values = [0, 1]
    D_values = [0]
    Q_values = [0, 1]
    S = [5, 40]

    with mlflow.start_run():
        if True:

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    for s in S:
                                        order = (p, d, q)
                                        seasonal_order = (P, D, Q, s)
                                        try:
                                            params = {
                                                "ticker": ticker,
                                                "p": p,
                                                "d": d,
                                                "q": q,
                                                "P": P,
                                                "D": D,
                                                "Q": Q,
                                                "S": s
                                            }
                                            
                                            model = SARIMAX(train_returns, order=order, seasonal_order=seasonal_order,
                                                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                                                
                                        except Exception as e:
                                            print(f"Error with SARIMA{order}x{seasonal_order}: {e}")
                                            continue
                                        
                                        forecast = model.forecast(steps=len(test_returns))
                                            
                                        mae = mean_absolute_error(test_returns, forecast)
                                        mse = mean_squared_error(test_returns, forecast)
                                        r2 = r2_score(test_returns, forecast)
                                            
                                        print(f"SARIMA{order}x{seasonal_order} - Test MSE: {mse:.4f} - Test R2: {r2:.4f}")
                                            
                                        mlflow.log_params(params)
                                        mlflow.log_metric("Test MAE", mae)
                                        mlflow.log_metric("Test MSE", mse)
                                        mlflow.log_metric("Test R2", r2)
                                            
                                        if mse < best_mse:
                                            best_mse = mse
                                            best_order = order
                                            best_seasonal_order = seasonal_order
                                            best_model = model
                                                
                                        results = {
                                            "model": "SARIMA",
                                            "Order": order,
                                            "Seasonal Order": seasonal_order,
                                            "Test MAE": mae,
                                            "Test MSE": mse,
                                            "Test R2": r2
                                        }
                                        results_sarima.append(results)

            print(f"\nBest SARIMA Model: SARIMA{best_order}x{best_seasonal_order} with Test MSE: {best_mse:.4f}")
            
    mlflow.end_run()  

    results_sarima_df = pd.DataFrame(results_sarima)
    results_sarima_df.to_csv("results/sarima_results.csv", index=False)
    
    sarima_model = SARIMAX(train_returns, order=best_order, seasonal_order=best_seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_forecast = sarima_model.forecast(steps=len(test_returns))

    sarima_mse = mean_squared_error(test_returns, sarima_forecast)
    sarima_r2 = r2_score(test_returns, sarima_forecast)
    print(f"SARIMA Test MSE: {sarima_mse:.4f}")
    print(f"SARIMA Test R2: {sarima_r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_returns, label="Train", color="blue")
    plt.plot(test_df["Date"], test_returns, label="Test", color="green")
    plt.plot(test_df["Date"], arima_forecast, label=f"SARIMA{best_order}{seasonal_order} Forecast", color="red")
    plt.title(f"SARIMA Model: Train, Test, and Forecast (Best Order: {best_order}{seasonal_order})")
    plt.xlabel("Date")
    plt.ylabel("Close Price Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/statisticalmodels/sarima_best_model_forecast.png", dpi=300)
    plt.show()





