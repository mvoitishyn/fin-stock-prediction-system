from trainmodel import run_linear_single_model, run_single_neural_network, run_bayesian_single_model
from model import load_data
from bayesianneuralnetworks import predict_bayesian_model

results = run_linear_single_model(
    ticker = "INTC",
    scale_bool=True,
    use_nonlinear_transformation_bool=False,
    use_macro_data_bool=True,
    use_extra_data_bool=True,
    target_window=1
)


results = run_single_neural_network(
    ticker = "INTC",
    epochs = 1000,
    learning_rate = 0.01,
    hidden_size =  1,
    scale = True,
    use_macro_data_bool = True,
    use_extra_data_bool = True,
    target_window = 1,
    cachedir ="cachedir"
)


results =  run_bayesian_single_model(
    ticker = "INTC",
    trainepochs = 1000,
    learning_rate = 0.01,
    clip_grad_value = 0.1,
    hidden_neurons =  1,
    scale = True,
    predsamples = 10,
    use_baseline_data_bool = True,
    use_macro_data_bool = True,
    use_extra_data_bool = True,
    target_window = 1,
    cachedir = "cachedir"
)

ticker = "INTC"
use_macro_data_bool = True
use_extra_data_bool = True
scale = True
target_window = 1

import pickle

model_path = f"cachedir/{ticker}_bs{True}_mac{use_macro_data_bool}_ex{use_extra_data_bool}_sc{scale}_w{target_window}_bayesian_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)
    
processedf = load_data("INTC")

mean_predictions, mean_variances = predict_bayesian_model(
        processedf=processedf,
        target_window=1,
        model = model,
        predsamples=15,
        use_baseline_data_bool=True,
        use_macro_data_bool=True,
        use_extra_data_bool=True,
        scale=True
)