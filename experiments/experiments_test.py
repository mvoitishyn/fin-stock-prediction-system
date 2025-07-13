from app.model_classes import BayesianNeuralNet_v0, BayesianNeuralNet_v1, BayesianNeuralNet_v2, BayesianNeuralNet_v3, BayesianNeuralNet_v4, BayesianNeuralNet_v5
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from app.model import load_data, featurize_data, scale_data, plot_actual_vs_predicted
from torch.nn.utils import clip_grad_norm_
import torch

import borch

import numpy as np

ticker = "INTC"
trainepochs = 1000
learningrate = 0.01
clip_grad_value = 0.1
hidden_neurons = 1
scale = True
predsamples = 15
use_baseline_data_bool = True
use_macro_data_bool = True
use_extra_data_bool = True
target_window = 1
cachadir = "cachedir"


rawdf = load_data(ticker)
X_train, X_test, y_train, y_test = featurize_data(
    rawdf, 
    use_baseline_data = use_baseline_data_bool, 
    use_macro_data = use_macro_data_bool, 
    use_extra_data = use_extra_data_bool, 
    target_window=target_window
)

x_train_dates = rawdf.loc[X_train.index, "Date"] 
x_test_dates = rawdf.loc[X_test.index, "Date"] 


X_test = X_test.drop(columns=["Date"])
X_train = X_train.drop(columns=["Date"])


if scale:
        X_train, X_test, _ = scale_data(X_train, X_test)
        
n_targets = target_window

mean_price_return = y_train.mean()
std_price_return = y_train.std()
    
n_features = X_train.shape[1]


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    
model = BayesianNeuralNet_v5(
    ticker=ticker,
    n_features=n_features,
    n_targets=n_targets,
    n_hidden=hidden_neurons,
    X_train = X_train_tensor,
    y_train = y_train_tensor,
    X_test = X_test_tensor,
    y_test = y_test_tensor,
    mean_price_return=mean_price_return,
    std_price_return=std_price_return
)

opt = torch.optim.AdamW(model.parameters(), lr = learningrate)
opt.zero_grad()
loss = 0
    
  
for epoch in range(trainepochs):
            
    borch.sample(model)
    model.observe(target=y_train_tensor)
    _ = model.forward(X_train_tensor)

    loss = borch.infer.vi_loss(**borch.pq_to_infer(model))
            
    print(loss)
            
            
    loss.backward()
            
    clip_grad_norm_(model.parameters(),clip_grad_value)
    opt.step()
    
    
    
model.observe(None)

# predict on training data
preds_train = np.zeros((predsamples, model.y_train.shape[0], 1))
preds_test = np.zeros((predsamples, model.y_test.shape[0], 1))

        
for i in range(predsamples):
    pred = model.forward(X_train_tensor)
    pred = pred.detach().numpy()
    preds_train[i] = pred

        
for i in range(predsamples):
    pred = model.forward(X_test_tensor)
    pred = pred.detach().numpy()
    preds_test[i] = pred

        
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
        
        
fig1 =  plot_actual_vs_predicted(y_train, y_test, mean_preds_train, mean_preds_test, std_preds_train, std_preds_test, x_train_dates, x_test_dates)
fig1.savefig(f"{ticker}_actual_vs_predict_plot_1.png", transparent=True, bbox_inches='tight')


