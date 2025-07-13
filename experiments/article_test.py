from app.bnn_experiments_classes import bnn_normal_relu_loc_scale, bnn_normal_relu_loc, bnn_normal_tanh_loc, bnn_studentt_relu_loc_scale, bnn_normal_tanh_loc_scale, bnn_studentt_tanh_loc_scale, bnn_studentt_relu_loc, bnn_studentt_tanh_loc


from app.trainmodel import run_bayesian_single_model
import pandas as pd

all_results = pd.DataFrame()

data_combinations = [
    {"baseline": True, "macro": False, "extra": False},
    {"baseline": True, "macro": True, "extra": False},
    {"baseline": True, "macro": True, "extra": True}
]

model_comlpexity_neurons = [1,3,9]
prediction_windows = [1,3,5]

for data_cfg in data_combinations:
    for neurons in model_comlpexity_neurons:
        for window in prediction_windows:
            model, actual, predicted, results_df = run_bayesian_single_model(
                ticker = "INTC",
                trainepochs=100,
                learning_rate=0.01,
                clip_grad_value=0.1,
                hidden_neurons=neurons,
                scale=True,
                predsamples=15,
                use_baseline_data_bool=data_cfg["baseline"],
                use_macro_data_bool=data_cfg["macro"],
                use_extra_data_bool=data_cfg["extra"],
                target_window=window,
                cachedir="cachedir"
            )
            all_results = pd.concat([all_results, results_df])
            
all_results = all_results.sort_values(by="test MSE", ascending=True).reset_index(drop=True)
all_results.to_csv("bnn_student_loc_tanh.csv")
