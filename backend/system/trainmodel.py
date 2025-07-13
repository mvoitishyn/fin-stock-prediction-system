from linearmodels import build_and_evaluate_linear_regression
from neuralnetworks import train_simple_nn
from bayesianneuralnetworks import train_and_eval_bnn_model

import pandas as pd

def run_linear_single_model(
    ticker: str,
    scale_bool: bool,
    use_nonlinear_transformation_bool: bool,
    use_macro_data_bool: bool,
    use_extra_data_bool: bool,
    target_window: int
):
    output = build_and_evaluate_linear_regression(
        ticker,
        scale_bool=scale_bool,
        use_nonlinear_transformation_bool=use_nonlinear_transformation_bool,
        use_baseline_data_bool=True,
        use_macro_data_bool=use_macro_data_bool,
        use_extra_data_bool=use_extra_data_bool,
        target_window=target_window,
    )

    if output is None:
        return None

    model, actual, predicted, results = output

    results_df = pd.DataFrame([results])
    results_df.to_csv("../results/linear_single_model_result.csv", index=False)

    return results_df

def run_linear_all_models(ticker: str):
    options = [
        {"scale_bool": True, "use_nonlinear_transformation_bool": False},
        {"scale_bool": True, "use_nonlinear_transformation_bool": True},
    ]
    windows = [1, 3, 5]
    
    results_list = []

    for option in options:
        for window in windows:
            for use_macro_data, use_extra_data in [
                (False, False),
                (True, False),
                (True, True),
            ]:
                output = build_and_evaluate_linear_regression(
                    ticker,
                    scale_bool=option["scale_bool"],
                    use_nonlinear_transformation_bool=option["use_nonlinear_transformation_bool"],
                    use_baseline_data_bool=True,
                    use_macro_data_bool=use_macro_data,
                    use_extra_data_bool=use_extra_data,
                    target_window=window,
                )
                if output is None:
                    continue
                model, actual, predicted, results = output
                results_list.append(results)
                
    results_df = pd.DataFrame(results_list)  
    results_df = results_df.sort_values(by=["test MSE","target"], ascending=True)
    results_df.to_csv("results/linear_models_results_NEW.csv", index=False)  
    
    return results_df

def run_single_neural_network(
    ticker: str,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    hidden_size: int = 3,
    scale: bool = True,
    use_macro_data_bool: bool = True,
    use_extra_data_bool: bool = True,
    target_window: int = 5,
    cachedir: str = "cachedir"
):
    output = train_simple_nn(
        ticker,
        use_baseline_data_bool=True,
        use_macro_data_bool=use_macro_data_bool,
        use_extra_data_bool=use_extra_data_bool,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        scale=scale,
        cachedir=cachedir,
        target_window=target_window
    )

    if output is None:
        return None

    model, actual, predicted, results = output
    results_df = pd.DataFrame([results])
    results_df.to_csv("results/neural_network_single_result.csv", index=False)

    return results_df


def run_all_neural_networks(ticker: str):
    results_list = []
    windows = [1, 3, 5]

    for scale_bool in [True]:
        for window in windows:
            for use_macro_data, use_extra_data in [
                (False, False),
                (True, False),
                (True, True),
            ]:
                model, actual, predicted, results = train_simple_nn(
                    ticker,
                    use_baseline_data_bool=True,
                    use_macro_data_bool=use_macro_data,
                    use_extra_data_bool=use_extra_data,
                    epochs=1000,
                    learning_rate=0.01,
                    hidden_size=6,
                    scale=scale_bool,
                    cachedir="cachedir",
                    target_window=window,
                )            
                results_list.append(results)

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by=["test MSE","target"], ascending=True)
    results_df.to_csv("results/neural_networks_results_NEW.csv", index=False)

    return results_df

def run_bayesian_single_model(
    ticker: str,
    trainepochs: int = 1000,
    learning_rate: float = 0.01,
    clip_grad_value: float = 0.1,
    hidden_neurons: int = 1,
    scale: bool = True,
    predsamples: int = 15,
    use_baseline_data_bool: bool = True,
    use_macro_data_bool: bool = True,
    use_extra_data_bool: bool = True,
    target_window: int = 5,
    cachedir: str = "cachedir"
):
    output = train_and_eval_bnn_model(
        ticker,
        trainepochs=trainepochs,
        learning_rate=learning_rate,
        clip_grad_value=clip_grad_value,
        hidden_neurons=hidden_neurons,
        scale=scale,
        predsamples=predsamples,
        use_baseline_data_bool=use_baseline_data_bool,
        use_macro_data_bool=use_macro_data_bool,
        use_extra_data_bool=use_extra_data_bool,
        target_window=target_window,
        cachedir=cachedir
    )

    if output is None:
        return None

    model, actual, predicted, results = output
    results_df = pd.DataFrame([results])
    results_df.to_csv("../results/bayesian_neural_networks_results.csv", index=False)

    return model, actual, predicted, results_df

    
def run_bayesian_all_neural_networks(ticker: str):
    results_list = []
    windows = [1, 3, 5]

    for scale_bool in [True]:
        for window in windows:
            for use_macro_data, use_extra_data in [
                (False, False),
                (True, False),
                (True, True),
            ]:
                model, actual, predicted, results = train_and_eval_bnn_model(
                    ticker,
                    trainepochs=1000,
                    learning_rate=0.01,
                    clip_grad_value=0.1,
                    hidden_neurons=1,
                    scale=scale_bool,
                    predsamples=15,
                    use_baseline_data_bool=True,
                    use_macro_data_bool=use_macro_data,
                    use_extra_data_bool=use_extra_data,
                    target_window=window,
                    cachedir="cachedir"

                )
                
                results_list.append(results)
                
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by=["test MSE","target"], ascending=True)
    results_df.to_csv("results/bayesian_neural_networks_results_NEW.csv", index=False)

    return results_df