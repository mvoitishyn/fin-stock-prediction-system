from model import load_data, featurize_data, featurize_data_for_prediction, scale_data, reconstruct_from_windows, plot_actual_vs_predicted
from model_classes import SimpleNN_v0, SimpleNN_v1, SimpleNN_v2

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import mlflow
import pickle

import torch
torch.classes.__path__ = []

def train_simple_nn(
    ticker,
    use_baseline_data_bool: bool,
    use_macro_data_bool: bool,
    use_extra_data_bool: bool,
    epochs, 
    learning_rate, 
    hidden_size, 
    scale,
    cachedir: str,
    target_window: int
    ):
    
    df = load_data(ticker)
    
    X_train, X_test, y_train, y_test = featurize_data(df, use_baseline_data=use_baseline_data_bool, use_macro_data=use_macro_data_bool, use_extra_data=use_extra_data_bool, target_window=target_window)

    dates = df.loc[X_test.index, "Date"] 
    
    try:
        X_test = X_test.drop(columns=["Date"])
        X_train = X_train.drop(columns=["Date"])
    except:
        pass

    # scale the data
    if scale:
        X_train, X_test, scaler_X = scale_data(X_train, X_test)
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        if target_window != 1:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        else:
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        if target_window != 1:
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        else:
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        
    else:
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        try:
            y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        except:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
   
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        try:
            y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
        except:
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
    input_size = X_train.shape[1]
    output_size = target_window
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("SNN")

    # define the model acritechture
    model = SimpleNN_v1(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    params = {
        "ticker": ticker,
        "n_features": input_size,
        "n_targets": output_size,
        "n_hidden": hidden_size,
        "trainepochs": epochs,
        "learning_rate": learning_rate,
        "apply_scaling": scale,
        "use_baseline_data": use_baseline_data_bool,
        "use_macro_data": use_macro_data_bool,
        "use_extra_data": use_extra_data_bool,
        "target_window": target_window 
    }
    with mlflow.start_run():
    
        mlflow.log_params(params)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred_train = model(X_train_tensor)
            loss = criterion(y_pred_train, y_train_tensor)
            
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
                
        model_save_path = f"{cachedir}/{ticker}_bs{use_baseline_data_bool}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_w{target_window}_neural_network.pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_save_path}")


        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor)
            y_pred_train = model(X_train_tensor)
            
            y_test = np.array(y_test)
            y_train = np.array(y_train)

            y_pred_test = np.array(y_pred_test)
            y_pred_train = np.array(y_pred_train)
        
        if target_window != 1:
            
            y_pred_train = reconstruct_from_windows(y_pred_train)
            y_pred_test = reconstruct_from_windows(y_pred_test)
            
            y_train = reconstruct_from_windows(y_train)
            y_test = reconstruct_from_windows(y_test)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)


        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"Train MSE: {mse_train}, Train MAE: {mae_train}, Train R2: {r2_train}")
        print(f"Test MSE: {mse_test}, Test MAE: {mae_test}, Test R2: {r2_test}")
        
        mlflow.log_metric("Train MSE", mse_train)
        mlflow.log_metric("Train MAE", mae_train)
        mlflow.log_metric("Train R2", r2_train)
        mlflow.log_metric("Test MSE", mse_test)
        mlflow.log_metric("Test MAE", mae_test)
        mlflow.log_metric("Test R2", r2_test)
        
        if target_window != 1:
            y_test = y_test[:-(target_window - 1)]
            y_pred_test = y_pred_test[:-(target_window - 1)]
        
        fig1,fig2,fig3,fig4 = plot_actual_vs_predicted(y_test, y_pred_test, dates)
        
        mlflow.log_figure(fig1, f"{ticker}_actual_vs_predict_plot_all.png")
        mlflow.log_figure(fig2, f"{ticker}_actual_vs_predict_plot_last_year.png")
        mlflow.log_figure(fig3, f"{ticker}_actual_vs_predict_plot_last_month.png")
        mlflow.log_figure(fig4, f"{ticker}_actual_vs_predict_plot_last_week.png")

    mlflow.end_run()
    
    
    results = {
        "model": model_save_path,
        "transformations": f"Scaled: {scale}",
        "features": f"Baseline: {use_baseline_data_bool}, Macro: {use_macro_data_bool}, Extra: {use_extra_data_bool}",
        "target": target_window,
        "train MSE": mse_train,
        "test MSE": mse_test 
    }

    return model, y_test, y_pred_test, results


def predict_neural_network(
    model_path: str, 
    rawdf, 
    scaler_X=False, 
    use_baseline_data_bool=True, 
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
        use_extra_data=use_extra_data_bool)
    
    
    X = X.drop(columns=["Date"])
       
    if scaler_X:
        X, _, _ = scale_data(X, X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred = y_pred.numpy()
    
    if target_window != 1:
        y_pred = reconstruct_from_windows(y_pred)
    
    return y_pred