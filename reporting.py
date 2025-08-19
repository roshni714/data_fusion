import csv
import os


def save_predictions(state_estimates, prediction_path, startdate, enddate):
    # Save the state estimates to a file
    state_estimates.to_csv(prediction_path + f"/{startdate}_{enddate}.csv", index=False)


def save_evaluation(eval, evaluation_path):
    # Save the evaluation results to a file
    with open(evaluation_path, "a+", newline="") as csvfile:
        field_names = eval.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(evaluation_path).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(eval)
