#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from data_loader import get_target_samples, get_train_samples
from datetime import datetime
import ast
import os
import matplotlib.gridspec as gridspec
from data_loader import load_state_ground_truth_data


COLORS = {
    "national": "#785EF0",
    "state_unadjusted": "#DC267F",
    "state_adjusted_balancing_X": "#fe6100",
    "state_adjusted": "#fe6100",
    "model_free": "#648FFF",
}

LINESTYLES = {
    "ground_truth": "-",
    "national": "-",
    "state_unadjusted": "-",
    "state_adjusted": "--",
    "state_adjusted_balancing_X": "-",
}

LABELS = {
    "ground_truth": "Ground Truth",
    "national": "Agg Admin Only",
    "state_unadjusted": "Online Survey Only",
    "state_adjusted_balancing_X": "Data Fusion\n" + r"$\eta(x, y) = xy$",
    "state_adjusted": "Data Fusion\n" + r"$\eta(x, y) =y$",
    "model_free": "Model-Free",
}

TITLES = {
    "medicaid_ins": "Medicaid Insurance Enrollment",
    "snap": "SNAP Enrollment",
    "RECVDVACC": "COVID-19 Vaccination",
    "synthetic": "Synthetic Data",
}


def get_upper_and_lower_mae(indicator):
    file_names = os.listdir(f"predictions/{indicator}/ground_truth")
    start_dates = []
    mae_uppers = []
    mae_lowers = []
    for file_name in file_names:
        full_path = (
            f"predictions/{indicator}/"
            + f"state_adjusted_balancing_intercept_nregions=1/{file_name}"
        )
        predictions_df = pd.read_csv(full_path)
        startdate = file_name.split("_")[0]
        enddate = file_name.split("_")[1].replace(".csv", "")
        state_gt_dataset = load_state_ground_truth_data(
            indicator=indicator,
            evaluation_group=["state_name"],
            startdate=startdate,
            enddate=enddate,
        )
        gt_df = state_gt_dataset.df
        weight = state_gt_dataset.weight
        mae_upper = np.abs(predictions_df[f"{indicator}_upper"] - gt_df[indicator])
        mae_lower = np.abs(gt_df[indicator] - predictions_df[f"{indicator}_lower"])
        mae_max = np.maximum(mae_upper, mae_lower)
        mae_min = np.minimum(mae_upper, mae_lower)
        start_dates.append(datetime.strptime(startdate, "%m-%d-%Y"))
        max_mae_agg = (mae_max * gt_df[weight]).sum()
        min_mae_agg = (mae_min * gt_df[weight]).sum()
        mae_uppers.append(min_mae_agg)
        mae_lowers.append(max_mae_agg)

    df = pd.DataFrame(
        {"startdate": start_dates, "mae_upper": mae_uppers, "mae_lower": mae_lowers}
    )
    df.sort_values("startdate", inplace=True)
    return df


def main_prediction_plot(show_methods=[]):
    plots_dir = Path("plots/main_paper")
    plots_dir.mkdir(parents=True, exist_ok=True)
    indicators = ["RECVDVACC", "medicaid_ins", "snap"]
    date_for_indicator = [
        "06-09-2021_06-21-2021",
        "01-01-2021_12-31-2021",
        "06-09-2021_06-21-2021",
    ]

    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    gs.update(hspace=0.4)

    fontsize = 20

    for i, indicator in enumerate(indicators):
        if i == 0:
            ax = plt.subplot(gs[0, :2])
        elif i == 1:
            ax = plt.subplot(gs[0, 2:])
        elif i == 2:
            ax = plt.subplot(gs[1, 1:3])

        file_name = date_for_indicator[i] + ".csv"
        sorted_state_order = pd.read_csv(
            f"predictions/{indicator}/ground_truth/{file_name}"
        ).sort_values(indicator)["state_name"]

        ground_truth_df = pd.read_csv(
            f"predictions/{indicator}/ground_truth/{file_name}"
        )
        ground_truth_df = (
            ground_truth_df.set_index("state_name")
            .loc[sorted_state_order]
            .reset_index()
        )
        for method in show_methods:
            name = method
            if method == "state_adjusted":
                name = "state_adjusted_balancing_intercept"
            full_path = f"predictions/{indicator}/" + f"{name}_nregions=1/{file_name}"
            if not os.path.exists(full_path):
                continue
            predictions_df = pd.read_csv(full_path)
            predictions_df = (
                predictions_df.set_index("state_name")
                .loc[sorted_state_order]
                .reset_index()
            )

            if "state_adjusted" in method:
                # For state_adjusted, use the specific balancing feature
                ax.errorbar(
                    ground_truth_df[indicator],
                    predictions_df[indicator],
                    yerr=[
                        predictions_df[indicator]
                        - predictions_df[indicator + "_lower"],
                        predictions_df[indicator + "_upper"]
                        - predictions_df[indicator],
                    ],
                    fmt="o",
                    label=LABELS[method],
                    color=COLORS[method],
                )
            else:
                ax.scatter(
                    ground_truth_df[indicator],
                    predictions_df[indicator],
                    marker="o",
                    linestyle="-",
                    label=LABELS[method],
                    color=COLORS[method],
                )

        ax.set_xlabel("Ground Truth", fontsize=fontsize)
        ax.set_ylabel("Prediction", fontsize=fontsize)
        date = " - ".join(file_name.split(".")[0].split("_"))
        ax.set_title(f"{TITLES[indicator]} \n {date}", fontsize=fontsize)

        # Set xlim and ylim to be the same based on data
        min_val = ground_truth_df[indicator].min()
        max_val = ground_truth_df[indicator].max()
        x_vals = np.linspace(min_val, max_val, 2)
        ax.plot(x_vals, x_vals, linestyle="dashed", color="gray", label="y=x")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize * 0.75)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize * 0.75)

        if i == 0:
            # Get handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Sort labels and handles
            sorted_pairs = sorted(zip(labels, handles))
            sorted_labels, sorted_handles = zip(*sorted_pairs)

            # Create the legend with sorted labels and handles
            ax.legend(sorted_handles, sorted_labels, fontsize=fontsize * 0.75)
    n_show_methods = len(show_methods)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/pred_{n_show_methods}.pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def main_mae_plot(show_methods=[], nregions=1):
    plots_dir = Path("plots/main_paper")
    plots_dir.mkdir(parents=True, exist_ok=True)
    indicators = ["RECVDVACC", "medicaid_ins", "snap"]

    fontsize = 20
    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    gs.update(hspace=0.4)

    for i, indicator in enumerate(indicators):
        if i == 0:
            ax = plt.subplot(gs[0, :2])
        elif i == 1:
            ax = plt.subplot(gs[0, 2:])
        elif i == 2:
            ax = plt.subplot(gs[1, 1:3])

        upper_lower_mae_df = get_upper_and_lower_mae(indicator)

        df = pd.read_csv(f"results/{indicator}.csv")
        df["startdate"] = pd.to_datetime(
            df["startdate"]
        )  # Convert startdate to datetime for proper plotting

        for method in show_methods:
            method_df = df[df["method"] == method]
            # For state_adjusted, use the specific balancing feature
            ax.plot(
                method_df["startdate"],
                method_df["mae"],
                color=COLORS[method],
                label=LABELS[method],
                marker="o",
            )

            if "state_adjusted" in method:
                # For state_adjusted, use the specific balancing feature
                ax.fill_between(
                    upper_lower_mae_df["startdate"],
                    upper_lower_mae_df["mae_upper"],
                    upper_lower_mae_df["mae_lower"],
                    color=COLORS[method],
                    alpha=0.2,
                )

        ax.set_xlabel("Start Date", fontsize=fontsize)
        ax.set_ylabel("MAE", fontsize=fontsize)
        ax.set_title(f"{TITLES[indicator]}", fontsize=fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize * 0.75)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize * 0.75)
        # Get handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Sort labels and handles
        sorted_pairs = sorted(zip(labels, handles))
        sorted_labels, sorted_handles = zip(*sorted_pairs)

        # Create the legend with sorted labels and handles
        if i == 0:
            ax.legend(sorted_handles, sorted_labels, fontsize=fontsize * 0.75)
            # Set ylim to match the first subplot
    plt.tight_layout()
    n_show_methods = len(show_methods)
    plt.savefig(f"plots/main_paper/mae_{n_show_methods}.pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def plot_predictions(indicator, group_name="state_name"):
    # Create plots directory and subdirectory if they don't exist
    plots_dir = Path(f"plots/{indicator}/predictions")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get list of file names from the specified directory
    directory_path = f"predictions/{indicator}/"
    file_names = os.listdir(f"predictions/{indicator}/ground_truth")

    for file_name in file_names:
        plt.figure()
        sorted_state_order = pd.read_csv(
            f"predictions/{indicator}/ground_truth/{file_name}"
        ).sort_values(indicator)[group_name]

        methods = ["national", "state_unadjusted", "state_adjusted", "model_free"]
        ground_truth_df = pd.read_csv(
            f"predictions/{indicator}/ground_truth/{file_name}"
        )
        if group_name == "state_name":
            ground_truth_df = (
                ground_truth_df.set_index(group_name)
                .loc[sorted_state_order]
                .reset_index()
            )
        for method in methods:
            name = method
            if method == "state_adjusted":
                name = "state_adjusted_balancing_intercept"
            full_path = directory_path + f"{name}_nregions=1/{file_name}"
            if not os.path.exists(full_path):
                continue

            predictions_df = pd.read_csv(full_path)
            if group_name == "state_name":
                predictions_df = (
                    predictions_df.set_index(group_name)
                    .loc[sorted_state_order]
                    .reset_index()
                )

            if method in COLORS and method in LABELS:
                if "state_adjusted" in method:
                    # For state_adjusted, use the specific balancing feature
                    plt.errorbar(
                        ground_truth_df[indicator],
                        predictions_df[indicator],
                        yerr=[
                            predictions_df[indicator]
                            - predictions_df[indicator + "_lower"],
                            predictions_df[indicator + "_upper"]
                            - predictions_df[indicator],
                        ],
                        fmt="o",
                        label=LABELS[method],
                        color=COLORS[method],
                    )
                else:
                    plt.scatter(
                        ground_truth_df[indicator],
                        predictions_df[indicator],
                        marker="o",
                        linestyle="-",
                        label=LABELS[method],
                        color=COLORS[method],
                    )
            else:
                print(f"Warning: No color/label defined for method {method}")

        # Set xlim and ylim to be the same based on data
        min_val = ground_truth_df[indicator].min()
        max_val = ground_truth_df[indicator].max()
        x_vals = np.linspace(min_val, max_val, 2)
        plt.plot(x_vals, x_vals, linestyle="dashed", color="gray", label="y=x")

        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        date = " - ".join(file_name.split(".")[0].split("_"))
        plt.title(f"{TITLES[indicator]} \n {date}")
        plt.tight_layout()
        # Sort legend labels and handles alphabetically
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_pairs = sorted(zip(labels, handles))
        if sorted_pairs:
            sorted_labels, sorted_handles = zip(*sorted_pairs)
            plt.legend(sorted_handles, sorted_labels)
        plt.savefig(f"{plots_dir}/{date}.pdf", bbox_inches="tight")
        plt.close()  # Close the figure to free memory


def plot_mae_over_time(indicator, nregions=1):
    # Create plots directory and subdirectory if they don't exist
    plots_dir = Path(f"plots/{indicator}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f"results/{indicator}.csv")
    methods_and_balancing_features = df[
        ["method", "balancing_features"]
    ].drop_duplicates()
    df["startdate"] = pd.to_datetime(
        df["startdate"]
    )  # Convert startdate to datetime for proper plotting

    for index, row in methods_and_balancing_features.iterrows():
        method = row["method"]
        balancing_features = row["balancing_features"]
        if pd.isna(
            balancing_features
        ):  # Use pandas isna() instead of direct comparison
            method_df = df[
                (df["method"] == method) & (df["balancing_features"].isnull())
            ]
            name = method  # For methods without balancing features, just use the method name
        else:
            method_df = df[
                (df["method"] == method)
                & (df["balancing_features"] == balancing_features)
            ]
            name = method + "_balancing"
            try:
                feature_list = ast.literal_eval(balancing_features)
                for feature in feature_list:
                    if feature:
                        name += "_" + feature
            except (ValueError, SyntaxError):
                print(
                    f"Warning: Could not parse balancing features: {balancing_features}"
                )
                continue

        if name in COLORS:  # Only plot if we have a color defined for this method
            plt.plot(
                method_df["startdate"],
                method_df["mae"],
                color=COLORS[name],
                label=LABELS[name],
                marker="o",
            )

    plt.xlabel("Start Date")
    plt.ylabel("MAE")
    plt.title(f"{TITLES[indicator]}")
    # Sort legend labels and handles alphabetically
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles))
    if sorted_pairs:
        sorted_labels, sorted_handles = zip(*sorted_pairs)
        plt.legend(sorted_handles, sorted_labels)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{indicator}/mae_over_time.pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def synthetic_data_plot():
    train_samples = get_train_samples(n_samples=2000)
    target_samples = get_target_samples(n_samples=2000)

    fig, ax = plt.subplots(
        1, 4, figsize=(24, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.3]}
    )
    legend_ax = ax[3]
    legend_ax.axis("off")
    fontsize = 20
    ax[1].scatter(train_samples["X"], train_samples["Y"], color="grey", alpha=0.1, s=5)
    ax[0].scatter(
        target_samples["X"], target_samples["Y"], color="grey", alpha=0.1, s=5
    )

    directory_path = f"predictions/synthetic/"
    file_name = "None_None.csv"
    methods = [
        "state_adjusted_balancing_X_nregions=1",
        "state_unadjusted_nregions=1",
        "national_nregions=1",
        "model_free_nregions=1",
    ]
    indicator = "synthetic"

    ground_truth_df = pd.read_csv(f"predictions/synthetic/ground_truth/{file_name}")
    for method in methods:
        full_path = directory_path + f"{method}/{file_name}"
        if not os.path.exists(full_path):
            continue

        predictions_df = pd.read_csv(full_path)
        # Strip _nregions=N suffix for color and label lookup
        base_method = method.split("_nregions=")[0]
        if base_method == "state_adjusted_balancing_intercept":
            base_method = "state_adjusted"
        if base_method in COLORS and base_method in LABELS:
            if "state_adjusted" in method:
                # For state_adjusted, use the specific balancing feature
                ax[2].scatter(
                    ground_truth_df[indicator],
                    predictions_df[indicator],
                    # yerr=[
                    #    predictions_df[indicator]
                    #    - predictions_df[indicator + "_lower"],
                    #    predictions_df[indicator + "_upper"]
                    #    - predictions_df[indicator],
                    # ],
                    # fmt="o",
                    marker="o",
                    label=LABELS[base_method],
                    color=COLORS[base_method],
                )
            else:
                ax[2].scatter(
                    ground_truth_df[indicator],
                    predictions_df[indicator],
                    marker="o",
                    linestyle="-",
                    label=LABELS[base_method],
                    color=COLORS[base_method],
                )
        else:
            print(f"Warning: No color/label defined for method {base_method}")

    # Set xlim and ylim to be the same based on data

    y_lim_0 = ax[0].get_ylim()
    y_lim_1 = ax[1].get_ylim()
    x_lim_0 = ax[0].get_xlim()
    x_lim_1 = ax[1].get_xlim()
    min_x = min(x_lim_0[0], x_lim_1[0])
    max_x = max(x_lim_0[1], x_lim_1[1])
    min_y = min(y_lim_0[0], y_lim_1[0])
    max_y = max(y_lim_0[1], y_lim_1[1])

    for i in range(len(ax)):
        ax[i].set_xlabel("X", fontsize=fontsize)
        ax[i].set_ylabel("Y", fontsize=fontsize)
        ax[i].tick_params(labelsize=fontsize * 0.75)
        if i < 2:
            ax[i].set_ylim(min_y, max_y)
            ax[i].set_xlim(min_x, max_x)

    xlim = ax[2].get_xlim()
    ylim = ax[2].get_ylim()
    min_2 = min(xlim[0], ylim[0])
    max_2 = max(xlim[1], ylim[1])
    ax[2].set_xlim(min_2, max_2)
    ax[2].set_ylim(min_2, max_2)

    x_vals = np.linspace(min_2, max_2, 2)
    ax[2].plot(x_vals, x_vals, linestyle="dashed", color="gray", label="y=x")

    ax[2].set_xlabel("Ground Truth", fontsize=fontsize)
    ax[2].set_ylabel("Prediction", fontsize=fontsize)
    ax[2].set_title("Prediction vs. Ground Truth", fontsize=fontsize)
    ax[1].set_title("Survey Distribution " + r"$S$", fontsize=fontsize)
    ax[0].set_title("Population Distribution " + r"$P$", fontsize=fontsize)

    # Move legend from ax[2] to ax[3]
    handles, labels = ax[2].get_legend_handles_labels()
    if handles:
        sorted_pairs = sorted(zip(labels, handles))
        sorted_labels, sorted_handles = zip(*sorted_pairs)
        legend_ax.legend(sorted_handles, sorted_labels, fontsize=fontsize, loc="best")

    # todo make axes the same
    plt.savefig("plots/main_paper/train_target_samples.pdf", bbox_inches="tight")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate prediction plots for survey indicators"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        choices=["medicaid_ins", "snap", "RECVDVACC", "synthetic", "all"],
        default="all",
        help="Which indicator to plot (default: all)",
    )
    args = parser.parse_args()

    # Create figures directory if it doesn't exist
    Path("figures").mkdir(exist_ok=True)

    # Define indicators and titles
    all_indicators = ["medicaid_ins", "snap", "RECVDVACC"]

    # Determine which indicators to process
    indicators = all_indicators if args.indicator == "all" else [args.indicator]

    if args.indicator in ["medicaid_ins", "snap", "RECVDVACC", "all"]:
        group_name = "state_name"
    elif args.indicator == "synthetic":
        group_name = "X"

    for indicator in indicators:
        print(f"\nProcessing {indicator}...")
        try:
            plot_mae_over_time(indicator)
            plot_predictions(indicator, group_name=group_name)

        except Exception as e:
            print(f"Error processing {indicator}: {str(e)}")
            import traceback

            traceback.print_exc()

    SHOW_METHODS = [
        [],
        ["national"],
        ["national", "state_unadjusted"],
        ["national", "state_unadjusted", "state_adjusted"],
        ["national", "state_unadjusted", "state_adjusted", "model_free"],
    ]

    for set_of_methods in SHOW_METHODS:
        main_prediction_plot(show_methods=set_of_methods)
    main_mae_plot(show_methods=SHOW_METHODS[-1], nregions=1)
    synthetic_data_plot()


if __name__ == "__main__":
    main()
