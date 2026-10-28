# Functions were made to account for multiple output variables (aside from just exam_score), however output data is
# always only one variable in these tests.

import pandas as pd
import fuzzySystem
import sys

def print_perfs(fuzzy_perf, bayes_perf, hybrid_perf):
    print("Fuzzy Expert System Performance:\n", fuzzy_perf)
    print("\n")
    print("Naïve Bayes Classification System Performance:\n", bayes_perf)
    print("\n")
    print("Hybrid System Performance:\n", hybrid_perf)

    # Determine highest MAE and RMSE of each antecedent
    min_mae = {}
    min_mae_sys = {}
    min_rmse = {}
    min_rmse_sys = {}
    for system, data in zip(["Fuzzy", "Bayes", "Hybrid"], [fuzzy_perf, bayes_perf, hybrid_perf]):
        for var, row in data.iterrows():
            if var not in min_mae or row.iloc[0] > min_mae[var]:
                min_mae[var] = row.iloc[0]
                min_mae_sys[var] = system
            if var not in min_rmse or row.iloc[1] > min_rmse[var]:
                min_rmse[var] = row.iloc[1]
                min_rmse_sys[var] = system

    print("\nSystem with best MAE for each variable:")
    for (var, mae), (_, system) in zip(min_mae.items(), min_mae_sys.items()):
        print(f"\t{system}: {var} = {mae}")

    print("\nSystem with best RMSE for each variable:")
    for (var, mae), (_, system) in zip(min_rmse.items(), min_rmse_sys.items()):
        print(f"\t{system}: {var} = {mae}")



def MAE(preds, resps):
    if preds.shape != resps.shape:
        raise ValueError("Shape of predictions and responses provided to MAE not equal.")

    mae_vals = {}
    for (col_name, pred_col), (_, resp_col) in zip(preds.items(), resps.items()):
        # Calculate mean for each column variable
        mae_vals[col_name] = sum(abs(pred_col[i] - resp_col[i]) for i in range(len(pred_col))) / len(pred_col)
    
    return pd.Series(mae_vals)

def RMSE(preds, resps):
    if preds.shape != resps.shape:
        raise ValueError("Shape of predictions and responses provided to MAE not equal.")

    rmse_vals = {}
    for (col_name, pred_col), (_, resp_col) in zip(preds.items(), resps.items()):
        # Calculate mean for each column variable
        rmse_vals[col_name] = (sum((pred_col[i] - resp_col[i])**2 for i in range(len(pred_col))) / len(pred_col))**0.5
    
    return pd.Series(rmse_vals)

# Generate fuzzy expert system predictions
def run_fuzzy_system(data_points):
    preds = []
    for _, row in data_points.iterrows():
        scores = fuzzySystem.run_rules(row)
        preds.append(fuzzySystem.weighted_final_score(scores))
    return pd.DataFrame(preds)

def main():
    if len(sys.argv) < 2:
        print("Script must be called with a dataset csv:")
        print("\tpython3 main.py <dataset>")
        return

    antecedants = list(fuzzySystem.DEFAULT_FUZZY_ANTECEDENTS.columns)
    consequences = list(fuzzySystem.DEFAULT_FUZZY_CONSEQUENCES.columns)

    dataset = sys.argv[1]
    
    # Gather data points
    df = pd.read_csv(
        dataset,
        usecols=antecedants + consequences
    )
    premis_vars = df[antecedants]
    response_vars = df[consequences]

    # Generate predictions
    fuzzy_preds = run_fuzzy_system(premis_vars)
    bayes_preds = []
    hybrid_preds = []

    # Measure performance
    fuzzy_perf = pd.DataFrame({
        "MAE": MAE(fuzzy_preds, response_vars),
        "RMSE": RMSE(fuzzy_preds, response_vars)
    })
    bayes_perf = pd.DataFrame({
        "MAE": {"exam_score": 0, "other_metric": 5},
        "RMSE": {"exam_score": 0, "other_metric": 2}
    })
    hybrid_perf = pd.DataFrame({
        "MAE": {"exam_score": 10, "other_metric": 3},
        "RMSE": {"exam_score": 0, "other_metric": 4}
    })

    print_perfs(fuzzy_perf, bayes_perf, hybrid_perf)

if __name__ == "__main__":
    main()