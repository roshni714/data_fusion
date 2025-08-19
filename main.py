import argh
from data_loader import (
    load_online_survey_data,
    load_census_data,
    load_national_ground_truth_data,
    load_state_ground_truth_data,
)
from data_utils import split
from regressor import (
    get_binary_conditional_predictor,
    get_cond_density_estimator,
    get_continuous_conditional_predictor,
)
from estimation import get_state_estimates, get_exponential_family_state_estimates
from model_free_estimation import get_model_free_exponential_family_state_estimates
from reporting import save_predictions, save_evaluation
from evaluate import evaluate_predictions
from covariate_balance import get_covariate_balance
import yaml


@argh.arg(
    "--config",
    help="which config file to use",
    default="configs/RECVDVACC/online_unadjusted.yaml",
)
def main(config="configs/RECVDVACC/online_unadjusted.yaml"):
    # Load the configuration file
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    method = config["method"]
    indicator = config["indicator"]
    date_ranges = config["date_ranges"]
    balancing_features = config.get("balancing_features", None)
    moment_group = config.get("moment_group", "state_name")
    prediction_path = config["prediction_path"]
    evaluation_path = config["evaluation_path"]
    n_constraints = config["n_constraints"]
    evaluation_group = config["evaluation_group"][0]

    census_dataset = load_census_data(indicator)

    for date_range in date_ranges:
        startdate = date_range["startdate"]
        enddate = date_range["enddate"]

        national_gt_dataset, regions = load_national_ground_truth_data(
            indicator=indicator,
            moment_group=moment_group,
            evaluation_group=evaluation_group,
            startdate=startdate,
            enddate=enddate,
            n_constraints=n_constraints,
        )

        state_gt_dataset = load_state_ground_truth_data(
            indicator=indicator,
            evaluation_group=evaluation_group,
            startdate=startdate,
            enddate=enddate,
        )

        if method == "national":
            attrs = evaluation_group + [indicator]
            state_estimates = national_gt_dataset.df[attrs]

        elif method == "state_unadjusted":
            online_survey_dataset = load_online_survey_data(
                indicator=indicator, startdate=startdate, enddate=enddate
            )
            train_dataset, validation_dataset = split(
                online_survey_dataset, frac=0.7, seed=144379
            )
            if indicator != "synthetic":

                _, covariate_ratio = get_covariate_balance(
                    online_survey_dataset, census_dataset
                )
            else:
                covariate_ratio = None

            if train_dataset.binary_indicator:
                predictor, _ = get_binary_conditional_predictor(
                    train_dataset, validation_dataset, covariate_ratio=covariate_ratio
                )
            else:
                predictor = get_continuous_conditional_predictor(
                    train_dataset, validation_dataset, covariate_ratio=covariate_ratio
                )

            state_estimates = get_state_estimates(
                indicator,
                predictor,
                census_dataset=census_dataset,
                evaluation_group=evaluation_group,
            )
        elif method == "state_adjusted":
            online_survey_dataset = load_online_survey_data(
                indicator=indicator, startdate=startdate, enddate=enddate
            )

            train_dataset, validation_dataset = split(
                online_survey_dataset, frac=0.7, seed=144379
            )

            if train_dataset.binary_indicator:
                predictor, predictor_metadata = get_binary_conditional_predictor(
                    train_dataset, validation_dataset, covariate_ratio=None
                )

            else:
                predictor = get_cond_density_estimator(
                    train_dataset, validation_dataset
                )
                predictor_metadata = None

            state_estimates = get_exponential_family_state_estimates(
                indicator=indicator,
                predictor=predictor,
                predictor_metadata=predictor_metadata,
                regions=regions,
                census_dataset=census_dataset,
                online_survey_dataset=online_survey_dataset,
                national_gt_dataset=national_gt_dataset,
                balancing_features=balancing_features,
                moment_group=moment_group,
                evaluation_group=evaluation_group,
                slack=config.get("slack", 0.01),
            )
        elif method == "ground_truth":
            attrs = evaluation_group + [indicator]
            state_estimates = state_gt_dataset.df[attrs]

        elif method == "model_free":
            online_survey_dataset = load_online_survey_data(
                indicator=indicator, startdate=startdate, enddate=enddate
            )

            train_dataset, validation_dataset = split(
                online_survey_dataset, frac=0.7, seed=144379
            )

            if indicator != "synthetic":
                _, covariate_ratio = get_covariate_balance(
                    online_survey_dataset, census_dataset
                )
            else:
                covariate_ratio = None

            if train_dataset.binary_indicator:
                predictor, _ = get_binary_conditional_predictor(
                    train_dataset, validation_dataset, covariate_ratio=covariate_ratio
                )
            else:
                predictor = get_cond_density_estimator(
                    train_dataset, validation_dataset
                )

            state_estimates = get_model_free_exponential_family_state_estimates(
                indicator=indicator,
                predictor=predictor,
                census_dataset=census_dataset,
                national_gt_dataset=national_gt_dataset,
                regions=regions,
                evaluation_group=evaluation_group,
                moment_group=moment_group,
            )

        save_predictions(
            state_estimates, prediction_path, startdate=startdate, enddate=enddate
        )
        if method in ["state_unadjusted", "state_adjusted", "national", "model_free"]:
            eval = evaluate_predictions(
                state_estimates, state_gt_dataset, indicator, evaluation_group
            )
            eval.update(
                {
                    "indicator": indicator,
                    "method": method,
                    "startdate": startdate,
                    "enddate": enddate,
                    "balancing_features": balancing_features,
                    "n_constraints": n_constraints,
                    "moment_group": moment_group,
                }
            )
            save_evaluation(eval, evaluation_path=evaluation_path)
            print(eval)


if __name__ == "__main__":

    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
