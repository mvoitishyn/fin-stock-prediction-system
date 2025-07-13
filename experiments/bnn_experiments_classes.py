import borch
import torch
import torch.nn as nn

import numpy as np

import borch
import torch
import torch.nn as nn
import torch.nn.functional as F
import borch.distributions as dist



# MODEL 1

class bnn_normal_tanh_loc_scale(borch.Module):
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

        self.target = borch.distributions.Normal(loc=mean, scale=std)
        return self.target

# MODEL 2
class bnn_normal_relu_loc_scale(borch.Module):
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
        self.activation = nn.ReLU()

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

        self.target = borch.distributions.Normal(loc=mean, scale=std)
        return self.target
    
# MODEL 3
class bnn_normal_tanh_loc(borch.Module):
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
        
        self.std_prcie_return = torch.tensor(std_price_return)
        
    def forward(self, X):
        x = self.linear1(X)
        x_activated = self.activation(x)

        mean = self.mean_head(x_activated)

        self.target = borch.distributions.Normal(loc=mean, scale=self.std_prcie_return)
        return self.target

# MODEL 4
class bnn_normal_relu_loc(borch.Module):
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
        self.activation = nn.ReLU()

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
        
        self.std_prcie_return = torch.tensor(std_price_return)
        
    def forward(self, X):
        x = self.linear1(X)
        x_activated = self.activation(x)

        mean = self.mean_head(x_activated)

        self.target = borch.distributions.Normal(loc=mean, scale=self.std_prcie_return)
        return self.target


# MODEL 5

class bnn_studentt_tanh_loc_scale(borch.Module):
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
    
    
# MODEL 6
class bnn_studentt_relu_loc_scale(borch.Module):
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
        self.activation = nn.ReLU()

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
    
    
    
# MODEL 7
class bnn_studentt_tanh_loc(borch.Module):
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
        
        self.std_price_return = torch.tensor(std_price_return)

    def forward(self, X):
        x = self.linear1(X)
        x_activated = self.activation(x)

        mean = self.mean_head(x_activated)

        self.target = borch.distributions.StudentT(df=4, loc=mean, scale=self.std_price_return)
        return self.target
    
    
    
    
# MODEL 8
class bnn_studentt_relu_loc(borch.Module):
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
        self.activation = nn.ReLU()

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
        
        self.std_price_return = torch.tensor(std_price_return)

    def forward(self, X):
        x = self.linear1(X)
        x_activated = self.activation(x)

        mean = self.mean_head(x_activated)

        self.target = borch.distributions.StudentT(df=4, loc=mean, scale=self.std_price_return)
        return self.target