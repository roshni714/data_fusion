import pandas as pd
import os
import yaml
import argparse
from typing import List, Optional


def generate_config(
    method,
    indicator,
    n_constraints,
    evaluation_group,
    moment_group,
    slack,
    balancing_features=None,
):
    # Use same date ranges as in HPS data
    date_ranges = []
    if indicator == "medicaid_ins":
        date_ranges = [
            {"startdate": "01-01-2021", "enddate": "12-31-2021"},  # 2021
            {"startdate": "01-01-2022", "enddate": "12-31-2022"},  # 2022
        ]
    elif indicator == "synthetic":
        date_ranges = [
            {"startdate": None, "enddate": None}
        ]  # No specific date range for synthetic data
    else:
        if indicator == "snap":
            start_week = 23
            end_week = 51
        elif indicator == "RECVDVACC":
            start_week = 25
            end_week = 40

        df = pd.read_excel("data/HPS/hps_weeks_lookup.xlsx")
        for index, row in df.iterrows():
            if row["Week"] > start_week and row["Week"] < end_week:
                date_ranges.append(
                    {
                        "startdate": row["Start"].strftime("%m-%d-%Y"),
                        "enddate": row["End"].strftime("%m-%d-%Y"),
                    }
                )

    base_config = {}
    if method == "ground_truth":
        base_config["prediction_path"] = f"predictions/{indicator}/{method}"
    elif balancing_features:
        feats = "balancing"
        for feat in balancing_features:
            feats += f"_{feat}"
        base_config["balancing_features"] = balancing_features
        base_config["prediction_path"] = (
            f"predictions/{indicator}/{method}_{feats.replace(' ', '_')}_nregions={n_constraints}"
        )
    else:
        base_config["prediction_path"] = (
            f"predictions/{indicator}/{method}_nregions={n_constraints}"
        )

    if not os.path.exists("configs"):
        os.makedirs("configs")
    if not os.path.exists(f"configs/{indicator}"):
        os.makedirs(f"configs/{indicator}")
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    if not os.path.exists(f"predictions/{indicator}"):
        os.makedirs(f"predictions/{indicator}")
    if not os.path.exists(base_config["prediction_path"]):
        os.makedirs(base_config["prediction_path"])
    if not os.path.exists("results"):
        os.makedirs("results")

    base_config["evaluation_path"] = f"results/{indicator}.csv"
    base_config["indicator"] = indicator
    base_config["method"] = method
    base_config["date_ranges"] = date_ranges
    base_config["n_constraints"] = n_constraints
    base_config["slack"] = slack
    base_config["evaluation_group"] = evaluation_group
    base_config["moment_group"] = moment_group  # Default moment group

    if method in ["ground_truth", "state_unadjusted"]:
        file_name = f"configs/{indicator}/{method}"
    else:
        file_name = f"configs/{indicator}/{method}_moment={moment_group}_nconstraints={n_constraints}"
        if balancing_features:
            file_name += "_balancing"
            for feature in balancing_features:
                file_name += f"_{feature.replace(' ', '_')}"

    file_name += ".yaml"
    with open(file_name, "w") as file:
        yaml.dump(base_config, file, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate configuration files for survey analysis"
    )
    parser.add_argument(
        "--indicator",
        type=str,
        required=True,
        choices=["snap", "medicaid_ins", "RECVDVACC", "synthetic"],
    )  # add onto this later?
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            "state_unadjusted",
            "national",
            "ground_truth",
            "state_adjusted",
            "model_free",
        ],
    )
    parser.add_argument("--balancing-features", type=str, nargs="+", action="append")
    parser.add_argument("--n-constraints", type=int, nargs="+")
    parser.add_argument("--slack", type=float, default=0.001)
    parser.add_argument("--evaluation-group", type=str, nargs="+", action="append")
    parser.add_argument("--moment-group", type=str, default="state_name")

    args = parser.parse_args()

    # Set default balancing features if not provided
    if args.balancing_features is None:
        args.balancing_features = []
        for method in args.methods:
            if method != "state_adjusted":
                args.balancing_features.append(None)
            else:
                args.balancing_features.append(["intercept"])
    elif len(args.balancing_features) < len(args.methods):
        args.balancing_features.extend(
            [None for _ in range(len(args.methods) - len(args.balancing_features))]
        )

    # Set default n_constraints if not provided
    if args.n_constraints is None:
        args.n_constraints = [1 for _ in args.methods]
    elif len(args.n_constraints) < len(args.methods):
        # Fill remaining methods with n_constraints=1
        args.n_constraints.extend(
            [1 for _ in range(len(args.methods) - len(args.n_constraints))]
        )

    print(f"\nGenerating configs for {args.indicator}")
    print(f"Methods: {args.methods}")
    print(f"Balancing features: {args.balancing_features}")
    print(f"Number of regions: {args.n_constraints}")
    print(f"Slack: {args.slack}\n")
    print(f"Evaluation group: {args.evaluation_group}")
    print(f"Moment group: {args.moment_group}\n")

    for method, balancing_features, n_constraints in zip(
        args.methods, args.balancing_features, args.n_constraints
    ):
        # Convert 'None' strings to None
        if balancing_features and balancing_features[0] == "None":
            balancing_features = None

        generate_config(
            method=method,
            indicator=args.indicator,
            n_constraints=n_constraints,
            evaluation_group=args.evaluation_group,
            moment_group=args.moment_group,
            slack=args.slack,
            balancing_features=balancing_features,
        )


if __name__ == "__main__":
    main()
