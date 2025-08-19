import pandas as pd


def evaluate_predictions(
    state_estimates, state_gt_dataset, indicator, evaluation_group
):
    # Merge the estimates with the ground truth
    merged_df = pd.merge(
        state_estimates, state_gt_dataset.df, on=evaluation_group, how="left"
    )
    # Check if the indicator exists in both dataframes

    # Calculate the error
    error = (
        (merged_df[indicator + "_x"] - merged_df[indicator + "_y"]).abs()
        * merged_df[state_gt_dataset.weight]
    ).sum()

    return {
        "mae": error,
        "correlation": merged_df[indicator + "_x"].corr(merged_df[indicator + "_y"]),
    }
