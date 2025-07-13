import borch
import torch
import torch.nn as nn

import numpy as np

import borch
import torch
import torch.nn as nn
import torch.nn.functional as F
import borch.distributions as dist

class SimpleNN_v0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_v0, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class SimpleNN_v1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_v1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x
    
class SimpleNN_v2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_v2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

class SimpleRNN_v0(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN_v0, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1) 
        out, _ = self.rnn(x) 
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class BayesianNeuralNet_v0(borch.Module):
    def __init__(
        self,
        ticker,
        n_features,
        n_targets,
        n_hidden, 
        X_train,
        y_train,
        X_test,
        y_test,
        mean_price_return,
        std_price_return
    ):
        super().__init__(posterior=borch.posterior.Automatic())
        
        self.linear1 = borch.nn.Linear(n_features, n_hidden)
         
        self.ticker = ticker
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    

    def forward(self, X):
   
        output_linear1 = self.linear1(X)
        
        self.target = borch.distributions.Normal(output_linear1, 0.02)
        
        return self.target

class BayesianNeuralNet_v1(borch.Module):
    def __init__(
        self,
        ticker,
        n_features,
        n_targets,
        n_hidden, 
        X_train,
        y_train,
        X_test,
        y_test,
        mean_price_return,
        std_price_return
    ):
        super().__init__(posterior=borch.posterior.Automatic())
        
        self.linear1 = borch.nn.Linear(n_features, n_hidden)
        self.relu1 = torch.nn.LeakyReLU()
        self.linear2 = borch.nn.Linear(n_hidden, n_targets)

       
        self.ticker = ticker
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    

    def forward(self, X):
        x = self.linear1(X)
        x = self.relu1(x)
        output = self.linear2(x)
        
        self.target = borch.distributions.Normal(output, 0.15)
        return self.target
    
class BayesianNeuralNet_v2(borch.Module):
    def __init__(
        self,
        ticker,
        n_features,
        n_targets,
        n_hidden, 
        X_train,
        y_train,
        X_test,
        y_test,
        mean_price_return,
        std_price_return
    ):
        super().__init__(posterior=borch.posterior.Automatic())
        
        self.linear1 = borch.nn.Linear(n_features, n_hidden) 
        self.relu = torch.nn.LeakyReLU()
        self.linear2 = borch.nn.Linear(n_hidden, n_targets)
       
        self.ticker = ticker
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
        self.sigma = borch.distributions.Normal(
            torch.tensor([np.abs(mean_price_return)], dtype=torch.float32), 
            torch.tensor([std_price_return], dtype=torch.float32)
        )
    def forward(self, X):
   
        output_linear1 = self.linear1(X)
        relu1 = self.relu(output_linear1)
        output_linear2 = self.linear2(relu1)
        
        self.target = borch.distributions.StudentT(
            df=3, 
            loc=output_linear2, 
            scale=abs(self.sigma) * 2
        )
        return self.target
    
class BayesianNeuralNet_v3(borch.Module):
    def __init__(
        self,
        ticker,
        n_features,
        n_targets,
        n_hidden, 
        X_train,
        y_train,
        X_test,
        y_test,
        mean_price_return,
        std_price_return
    ):
        super().__init__(posterior=borch.posterior.Automatic())
        
        self.linear1 = borch.nn.Linear(n_features, n_hidden)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = borch.nn.Linear(n_hidden, n_hidden)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = borch.nn.Linear(n_hidden, n_targets)

       
        self.ticker = ticker
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
        self.sigma = borch.distributions.LogNormal(
            torch.tensor([np.log(mean_price_return)], dtype=torch.float32), 
            torch.tensor([std_price_return], dtype=torch.float32)
        )

    def forward(self, X):
        x = self.relu1(self.linear1(X))
        x = self.relu2(self.linear2(x))
        output = self.linear3(x)
        self.target = borch.distributions.StudentT(
            df=3, 
            loc=output, 
            scale=0.2
        )
        return self.target
    
    
class BayesianNeuralNet_v4(borch.Module):
    def __init__(
        self,
        ticker,
        n_features,
        n_targets,
        n_hidden, 
        X_train,
        y_train,
        X_test,
        y_test,
        mean_price_return = 0.02,
        std_price_return = 0.15

    ):
        super().__init__(posterior=borch.posterior.Automatic())
        
        self.linear1 = borch.nn.Linear(n_features, n_hidden)
        self.activation = nn.Tanh()
        
        self.mean_head = borch.nn.Linear(n_hidden, n_targets)
        self.logstd_head = borch.nn.Linear(n_hidden, n_targets)

        self.ticker = ticker
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    

    def forward(self, X):
        x = self.linear1(X)
        x_activated = self.activation(x)
        
        mean = self.mean_head(x_activated)
        std = self.logstd_head(x_activated)
        
        self.target = borch.distributions.Normal(loc=mean, scale=abs(std))
        return self.target
    
class BayesianNeuralNet_v5(borch.Module):
    def __init__(
        self,
        ticker,
        n_features,
        n_targets,
        n_hidden, 
        X_train,
        y_train,
        X_test,
        y_test,
        mean_price_return,
        std_price_return
    ):
        super().__init__(posterior=borch.posterior.Automatic())

        self.linear1 = borch.nn.Linear(
            n_features, n_hidden,

           
        )
        self.activation = nn.Tanh()

        self.mean_head = borch.nn.Linear(
            n_hidden, n_targets,
            bias=dist.Normal(0, 1),
        )

        self.logstd_head = borch.nn.Linear(
            n_hidden, n_targets,
            bias=dist.Normal(0, 2),
        )

        # Store inputs
        self.ticker = ticker
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def forward(self, X):
        x = self.linear1(X)
        x_activated = self.activation(x)

        mean = self.mean_head(x_activated)
        log_std = self.logstd_head(x_activated)
        std = F.softplus(log_std) + 1e-4  

        self.target = borch.distributions.StudentT(df=4, loc=mean, scale=std)
        return self.target