from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from app.model import load_data, featurize_data, featurize_data_for_prediction, scale_data, plot_actual_vs_predicted, reconstruct_from_windows

import mlflow
import numpy as np
import pandas as pd
import pickle


def build_and_evaluate_linear_regression(
    ticker, 
    scale_bool,
    use_nonlinear_transformation_bool, 
    use_baseline_data_bool,
    use_macro_data_bool, 
    use_extra_data_bool,
    target_window
):
    df = load_data(ticker)

    X_train, X_test, y_train, y_test = featurize_data(
        df, 
        use_baseline_data=use_baseline_data_bool, 
        use_macro_data=use_macro_data_bool, 
        use_extra_data=use_extra_data_bool, 
        target_window=target_window)
    dates = df.loc[X_test.index, "Date"] 

    try: 
        X_test = X_test.drop(columns=["Date"])
        X_train = X_train.drop(columns=["Date"])
    except:
        pass
    
    # IMPORTANT: non-linear transformation should be applied before scaling
    if use_nonlinear_transformation_bool:
        degree = 2
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        
    # scale the data
    if scale_bool:
        X_train, X_test, scaler_X = scale_data(X_train, X_test)
    
    # get min-max date for training and testing    
    model = LinearRegression()
    
    print(X_train)
    
    model.fit(X_train, y_train)
    cachedir = "cachedir"
    model_save_path = f"{cachedir}/{ticker}_bs{use_baseline_data_bool}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale_bool}_nt{use_nonlinear_transformation_bool}_w{target_window}_linear_model.pkl"
    with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
    print(f"Model saved to {model_save_path}")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if target_window != 1:
        
        y_pred_train = reconstruct_from_windows(y_pred_train)
        y_pred_test = reconstruct_from_windows(y_pred_test)
        
        y_train = reconstruct_from_windows(y_train)
        y_test = reconstruct_from_windows(y_test)

        
    mse_train = round(float(mean_squared_error(y_train, y_pred_train)),7)
    mae_train = round(float(mean_absolute_error(y_train, y_pred_train)),7)
    r2_train = round(float(r2_score(y_train, y_pred_train)),7)
    
    mse_test = round(float(mean_squared_error(y_test, y_pred_test)),7)
    mae_test = round(float(mean_absolute_error(y_test, y_pred_test)),7)
    r2_test = round(float(r2_score(y_test, y_pred_test)),7)
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    mlflow.set_experiment("Linear Regression")
    
    with mlflow.start_run():
        
        params = {
            "ticker": ticker,
            "apply_scaling": scale_bool,
            "apply_nonlinear_transformation": use_nonlinear_transformation_bool,
            "use_baseline_data": use_baseline_data_bool,
            "use_macro_data": use_macro_data_bool,
            "use_extra_data": use_extra_data_bool,
            "target_window": target_window
        }
        
        mlflow.log_params(params)
        
        metrics = {
            "training accuracy": {"MSE": mse_train, "MAE": mae_train, "R2": r2_train},
            "test accuracy": {"MSE": mse_test, "MAE": mae_test, "R2": r2_test}
        }
        
        mlflow.log_metric("MSE_train", mse_train)
        mlflow.log_metric("MAE_train", mae_train)
        mlflow.log_metric("R2_train", r2_train)
        mlflow.log_metric("MSE_test", mse_test)
        mlflow.log_metric("MAE_test", mae_test)
        mlflow.log_metric("R2_test", r2_test)
        
        print(f"Evaluation training: {metrics['training accuracy']}")
        print(f"Evaluation testing: {metrics['test accuracy']}")
        
        
        if target_window != 1:
            y_test = y_test[:-(target_window - 1)]
            y_pred_test = y_pred_test[:-(target_window - 1)]
            
        fig1,fig2,fig3,fig4 = plot_actual_vs_predicted(y_test, y_pred_test, dates)
        
        mlflow.log_figure(fig1, f"{ticker}_actual_vs_predict_plot_all.png")
        mlflow.log_figure(fig2, f"{ticker}_actual_vs_predict_plot_last_year.png")
        mlflow.log_figure(fig3, f"{ticker}_actual_vs_predict_plot_last_month.png")
        mlflow.log_figure(fig4, f"{ticker}_actual_vs_predict_plot_last_week.png")
        
    
    mlflow.end_run()
    
    y_test = np.array(y_test)
    model_name = f"LinearRegression_{ticker}_{target_window}"
    
    results = {
        "model": model_name,
        "transformations": f"Scaled: {scale_bool}, Non-linear: {use_nonlinear_transformation_bool}",
        "features": f"Baseline: {use_baseline_data_bool}, Macro: {use_macro_data_bool}, Extra: {use_extra_data_bool}",
        "target": target_window,
        "train MSE": mse_train ,
        "test MSE": mse_test 
    }
    
    return model, y_test, y_pred_test, results

# predict with linear regression model

def predict_linear_model(
    model_path, 
    rawdf, 
    scaler_X=None, 
    use_nonlinear_transformation_bool=False, 
    use_baseline_data_bool=False, 
    use_macro_data_bool=False, 
    use_extra_data_bool=False, 
    target_window=1
    ):
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = featurize_data_for_prediction(
        rawdf, 
        use_baseline_data=use_baseline_data_bool, 
        use_macro_data=use_macro_data_bool, 
        use_extra_data=use_extra_data_bool
    )

    try:
        X = X.drop(columns=["Date"])
    except KeyError:
        pass

    if use_nonlinear_transformation_bool:
        degree = 2
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X = poly.fit_transform(X)

    if scaler_X:
        X, _, _ = scale_data(X,X)
        
    predictions = model.predict(X)

    if target_window != 1:
        predictions = reconstruct_from_windows(predictions)

    return predictions