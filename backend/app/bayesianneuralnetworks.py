from app.model import load_data, featurize_data, scale_data, plot_actual_vs_predicted_bayesian, reconstruct_from_windows, featurize_data_for_prediction
from app.model_classes import BayesianNeuralNet_v0, BayesianNeuralNet_v1, BayesianNeuralNet_v2, BayesianNeuralNet_v3, BayesianNeuralNet_v4, BayesianNeuralNet_v5
from app.bnn_experiments_classes import bnn_normal_relu_loc_scale, bnn_normal_relu_loc, bnn_normal_tanh_loc, bnn_studentt_relu_loc_scale, bnn_normal_tanh_loc_scale, bnn_studentt_tanh_loc_scale, bnn_studentt_relu_loc, bnn_studentt_tanh_loc

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import borch
from torch.nn.utils import clip_grad_norm_

import numpy as np

import mlflow
import pickle

def train_and_eval_bnn_model(
    ticker: str,
    trainepochs: int,
    learning_rate: float,
    clip_grad_value: float,
    hidden_neurons: int, 
    scale: bool,
    predsamples: int,
    use_baseline_data_bool: bool,
    use_macro_data_bool: bool,
    use_extra_data_bool: bool,
    target_window: int,
    cachedir: str
    ):
    df = load_data(ticker)
    X_train, X_test, y_train, y_test = featurize_data(df, use_baseline_data = use_baseline_data_bool, use_macro_data = use_macro_data_bool, use_extra_data = use_extra_data_bool, target_window=target_window)
        
    x_train_dates = df.loc[X_train.index, "Date"] 
    x_test_dates = df.loc[X_test.index, "Date"] 
    
    try:
        X_test = X_test.drop(columns=["Date"])
        X_train = X_train.drop(columns=["Date"])
    except:
        pass
    
    n_features = X_train.shape[1]
    n_targets = target_window

    mean_price_return = y_train.mean()
    std_price_return = y_train.std()
    
    
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
    
    model = bnn_studentt_tanh_loc(
        ticker, 
        n_features, 
        n_targets,
        hidden_neurons, 
        X_train_tensor, 
        y_train_tensor,
        X_test_tensor, 
        y_test_tensor, 
        mean_price_return, 
        std_price_return
    )
    
    opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    opt.zero_grad()
    loss = 0
    
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("BNN")
    
    with mlflow.start_run():
    
        for epoch in range(trainepochs):
            
            borch.sample(model)
            model.observe(target=y_train_tensor)
            _ = model.forward(X_train_tensor)

            loss = borch.infer.vi_loss(**borch.pq_to_infer(model))
            
            print(loss)
            
            mlflow.log_metric("loss", loss, step=epoch)
            
            loss.backward()
            
            clip_grad_norm_(model.parameters(),clip_grad_value)
            opt.step()
            
        params = {
            "ticker": ticker,
            "n_features": n_features,
            "n_targets": n_targets,
            "n_hidden": hidden_neurons,
            "trainepochs": trainepochs,
            "learning_rate": learning_rate,
            "clip_grad_value": clip_grad_value,
            "hidden_neurons": hidden_neurons,
            "scale": scale,
            "use_baseline_data": use_baseline_data_bool,
            "use_macro_data": use_macro_data_bool,
            "use_extra_data": use_extra_data_bool,
            "target_window": target_window
        }
        
        mlflow.log_params(params)
        
        model_save_path = f"{cachedir}/{ticker}_bs{use_baseline_data_bool}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_w{target_window}_bayesian_model.pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_save_path}")

        
        model.observe(None)

        # predict on training data
        preds_train = np.zeros((predsamples, model.y_train.shape[0], 1))
        
        for i in range(predsamples):
            pred = model.forward(X_train_tensor)
            pred = pred.detach().numpy()
            if target_window != 1:
                pred = pred[:-(target_window - 1)]
                pred = reconstruct_from_windows(pred)
            preds_train[i] = pred
            
        mean_preds_train = preds_train.mean(axis=0)
        
        # predict on testing data
        preds_test = np.zeros((predsamples, model.y_test.shape[0], 1))
        
        for i in range(predsamples):
            pred = model.forward(X_test_tensor)
            pred = pred.detach().numpy()
            if target_window != 1:
                pred = pred[:-(target_window - 1)]
                pred = reconstruct_from_windows(pred)
            preds_test[i] = pred
            
        mean_preds_train = preds_train.mean(axis=0)
        mean_preds_test = preds_test.mean(axis=0)

        if target_window != 1:
            y_train = reconstruct_from_windows(y_train)
            y_test = reconstruct_from_windows(y_test)
            
            y_test = y_test[:-(target_window - 1)]            
            y_train = y_train[:-(target_window - 1)]

        mean_preds_train = preds_train.mean(axis=0)
        mean_preds_test = preds_test.mean(axis=0)

        std_preds_train = preds_train.std(axis=0)
        std_preds_test= preds_test.std(axis=0)
            
        # evaluate the model on test data
        mse_train = mean_squared_error(y_train, mean_preds_train)
        mae_train = mean_absolute_error(y_train, mean_preds_train)
        r2_train = r2_score(y_train, mean_preds_train)

        # evaluate the model on test data
        mse_test = mean_squared_error(y_test, mean_preds_test)
        mae_test = mean_absolute_error(y_test, mean_preds_test)
        r2_test = r2_score(y_test, mean_preds_test)

        print(f"Train MSE: {mse_train}, Train MAE: {mae_train}, Train R2: {r2_train}")
        print(f"Test MSE: {mse_test}, Test MAE: {mae_test}, Test R2: {r2_test}")
        
        fig1, fig2, fig3, fig4 = plot_actual_vs_predicted_bayesian(y_train, y_test, mean_preds_train, mean_preds_test, std_preds_train, std_preds_test, x_train_dates, x_test_dates, ticker)
        
        mlflow.log_figure(fig1, f"{ticker}_actual_vs_predict_plot_all.png")
        mlflow.log_figure(fig2, f"{ticker}_actual_vs_predict_plot_last_year.png")
        mlflow.log_figure(fig3, f"{ticker}_actual_vs_predict_plot_last_month.png")
        mlflow.log_figure(fig4, f"{ticker}_actual_vs_predict_plot_week.png")


    mlflow.end_run()
    
    target_distribution = "StudenT"
    
    results = {
        "model": model_save_path,
        "transformations": f"Scaled: {scale}, Target Distribution: {target_distribution}",
        "targets": "location",
        "features": f"Baseline: {use_baseline_data_bool}, Macro: {use_macro_data_bool}, Extra: {use_extra_data_bool}",
        "activation_function": "TanH",
        "hidden_neurons": hidden_neurons,
        "target": target_window,
        "train MSE": mse_train,
        "test MSE": mse_test 
    }
    
    return model, y_test, mean_preds_test, results

def averaged_prediction(model, X_tensor, n_runs):
    preds = [model.forward(X_tensor) for _ in range(n_runs)]
    stacked = torch.stack(preds, dim=0)
    avg_pred = stacked.mean(dim=0)
    var_pred  = stacked.var(dim=0, unbiased=False)
    return avg_pred, var_pred

def predict_bayesian_model(
    processedf: str,
    target_window: int,
    model,
    predsamples: int,
    use_baseline_data_bool: bool,
    use_macro_data_bool: bool,
    use_extra_data_bool: bool,
    scale: bool
):
    
    X = featurize_data_for_prediction(
        processedf, 
        use_baseline_data=use_baseline_data_bool, 
        use_macro_data=use_macro_data_bool, 
        use_extra_data=use_extra_data_bool)
    
    print(X)
    print(X.shape)
    
    X = X.drop(columns=["Date"])
    
    # if scale:
    #     # X, _, _ = scale_data(X, X)
    #     X_tensor = torch.tensor(X, dtype=torch.float32)
    # else:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
       
    print(X_tensor)
    mean_preds, var_preds = averaged_prediction(model, X_tensor, n_runs=predsamples)

    return mean_preds, var_preds

def predict_all_bayesian_model(
    processedf: str,
    target_window: int,
    model,
    predsamples: int,
    use_baseline_data_bool: bool,
    use_macro_data_bool: bool,
    use_extra_data_bool: bool,
    scale: bool
):
    
    X_train, X_test, y_train, y_test = featurize_data(
        processedf, 
        use_baseline_data=use_baseline_data_bool, 
        use_macro_data=use_macro_data_bool, 
        use_extra_data=use_extra_data_bool)
    
    dates = X_test["Date"]
    
    X_test = X_test.drop(columns=["Date"])
    
    if scale:
        X_test, _, _ = scale_data(X_test, X_test)
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
       
    mean_preds, var_preds = averaged_prediction(model, X_tensor, n_runs=predsamples)

    return mean_preds, var_preds, dates


