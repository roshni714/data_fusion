import pandas as pd
import datetime
import numpy as np
from data_utils import Dataset
import os
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import interp1d


THREE_REGION = {
    "East": [
        "Alabama",
        "Connecticut",
        "Delaware",
        "District of Columbia",
        "Florida",
        "Georgia",
        "Indiana",
        "Kentucky",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "New Hampshire",
        "New Jersey",
        "New York",
        "North Carolina",
        "Ohio",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "Tennessee",
        "Vermont",
        "Virginia",
        "West Virginia",
    ],
    "West": [
        "Alaska",
        "Arizona",
        "California",
        "Hawaii",
        "Idaho",
        "Montana",
        "Nevada",
        "Oregon",
        "Utah",
        "Washington",
    ],
    "Central": [
        "Arkansas",
        "Colorado",
        "Illinois",
        "Iowa",
        "Kansas",
        "Louisiana",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Nebraska",
        "New Mexico",
        "North Dakota",
        "Oklahoma",
        "South Dakota",
        "Texas",
        "Wisconsin",
        "Wyoming",
    ],
}
ONE_REGION = {"East": THREE_REGION["East"]}
TWO_REGION = {"Central": THREE_REGION["Central"], "East": THREE_REGION["East"]}

REGIONS_DIC = {
    "state_name": {1: ONE_REGION, 2: TWO_REGION, 3: THREE_REGION},
}

STATES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

MONTH_ORDER = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def month_num(month):
    return MONTH_ORDER[month]


def acs_dataloader():
    acs_weights = pd.read_csv("data/ACS/acs_weights_states_hps.csv")
    # Merge additional state-level characteristics
    acs_characteristics = pd.read_csv("data/ACS/acs_state_characteristics.csv")
    merged_df = pd.merge(acs_weights, acs_characteristics, on="state_name", how="left")

    # One-hot encode categorical variables
    categorical_columns = [
        "age_cat",
        "income_detailed",
        "race_grp",
        "education",
        "num_hh_cat",
    ]
    dummies_df = pd.get_dummies(
        merged_df[categorical_columns], columns=categorical_columns
    )
    # Replace "Male" and "Female" in the "sex" column with 0 and 1
    merged_df["sex"] = merged_df["sex"].map({"Male": 0, "Female": 1})
    # merged_df["age_cat"] = merged_df["age_cat"].map(AGE_MAP)
    # merged_df["income_detailed"] = merged_df["income_detailed"].map(INCOME_MAP)
    # Coarsen categories by summing relevant columns
    # for new_col, old_cols in COARSEN_CATEGORIES.items():
    #     dummies_df[new_col] = dummies_df[old_cols].sum(axis=1)
    #     dummies_df = dummies_df.drop(columns=old_cols)
    merged_df = pd.merge(merged_df, dummies_df, left_index=True, right_index=True)

    columns = dummies_df.columns.tolist() + [
        "sex",
        "avg_hh_size",
        "edu_hs_or_less",
        "english_only",
        "female_never_married",
        "fertility_rate",
        "food_stamps",
        "graduate_degree",
        "hh_computer",
        "hh_internet",
        "male_never_married",
        "mean_income",
        "median_house_value",
        "median_income",
        "median_rent",
        "poverty",
        "some_college_or_2yr",
        "unemployment_rate",
        "us_born",
        "veterans",
        "republican_pct",
        "tot_pop",
    ]

    dataset = Dataset(
        df=merged_df,
        covs=columns,
        indicator=None,
        weight="tot_weight",
        binary_indicator=None,
    )
    return dataset


def hps_dataloader(indicator, startdate, enddate):

    df = pd.read_csv("data/HPS/hps_prepped.csv", low_memory=False)
    # if indicator == "snap":
    #     df = pd.read_csv("data/HPS/hps_prepped_snap_amy.csv")
    date_df = pd.read_excel("data/HPS/hps_weeks_lookup.xlsx")

    print(f"\nRequested date range: {startdate} to {enddate}\n")

    # Restrict HPS data to appropriate date range
    startdate = datetime.datetime.strptime(startdate, "%m-%d-%Y")
    enddate = datetime.datetime.strptime(enddate, "%m-%d-%Y")

    # Find any weeks that overlap with the requested date range
    overlapping_weeks = date_df.loc[
        # Week starts before end date AND ends after start date
        (date_df["Start"] <= enddate)
        & (date_df["End"] >= startdate)
    ]

    if overlapping_weeks.empty:
        print(
            f"Warning: No HPS weeks found overlapping with the date range {startdate} to {enddate}"
        )
        # Return empty dataset with same structure
        empty_df = pd.DataFrame(columns=["state_name", indicator, "PWEIGHT"])
        return Dataset(
            df=empty_df,
            covs=[],
            indicator=indicator,
            weight="PWEIGHT",
            binary_indicator=True,
        )

    week_start = overlapping_weeks["Week"].min()
    week_end = overlapping_weeks["Week"].max()

    print(f"\nUsing HPS weeks {week_start} to {week_end}\n")

    sub_df = df.loc[(df["WEEK"] >= week_start) & (df["WEEK"] <= week_end)].reset_index(
        drop=True
    )

    # Drop rows with missing values in indicator column
    sub_df.dropna(subset=[indicator], inplace=True)

    if len(sub_df) == 0:
        print(
            f"Warning: No valid data found for indicator {indicator} in weeks {week_start}-{week_end}"
        )
        empty_df = pd.DataFrame(columns=["state_name", indicator, "PWEIGHT"])
        return Dataset(
            df=empty_df,
            covs=[],
            indicator=indicator,
            weight="PWEIGHT",
            binary_indicator=True,
        )

        # Merge ACS weights
    acs_weights = pd.read_csv("data/ACS/acs_weights_states_hps.csv")
    merged_df = pd.merge(
        sub_df,
        acs_weights,
        on=[
            "race_grp",
            "education",
            "age_cat",
            "sex",
            "num_hh_cat",
            "any_ins",
            "income_detailed",
            "state_name",
        ],
        how="left",
    )

    # Merge additional state-level characteristics
    acs_characteristics = pd.read_csv("data/ACS/acs_state_characteristics.csv")
    merged_df = pd.merge(merged_df, acs_characteristics, on="state_name", how="left")
    merged_df["income_detailed"] = merged_df["income_detailed"].astype("Int64")

    # One-hot encode categorical variables
    categorical_columns = [
        "age_cat",
        "race_grp",
        "income_detailed",
        "education",
        "num_hh_cat",
    ]
    dummies_df = pd.get_dummies(
        merged_df[categorical_columns], columns=categorical_columns
    )
    merged_df["sex"] = merged_df["sex"].map({"Male": 0, "Female": 1})
    # merged_df["age_cat"] = merged_df["age_cat"].map(AGE_MAP)
    # merged_df["income_detailed"] = merged_df["income_detailed"].map(INCOME_MAP)

    # Coarsen categories by summing relevant columns
    # for new_col, old_cols in COARSEN_CATEGORIES.items():
    #     dummies_df[new_col] = dummies_df[old_cols].sum(axis=1)
    #     dummies_df = dummies_df.drop(columns=old_cols)
    merged_df = pd.merge(merged_df, dummies_df, left_index=True, right_index=True)

    assert merged_df[indicator].isna().sum() == 0

    # Fill in NaN values with mean
    columns = dummies_df.columns.tolist() + [
        "sex",
        "avg_hh_size",
        "edu_hs_or_less",
        "english_only",
        "female_never_married",
        "fertility_rate",
        "food_stamps",
        "graduate_degree",
        "hh_computer",
        "hh_internet",
        "male_never_married",
        "mean_income",
        "median_house_value",
        "median_income",
        "median_rent",
        "poverty",
        "some_college_or_2yr",
        "unemployment_rate",
        "us_born",
        "veterans",
        "republican_pct",
        "tot_pop",
    ]
    covs = sorted(columns)
    selected_columns = [
        col for col in merged_df.columns if any(col.startswith(var) for var in covs)
    ]

    mean = merged_df[selected_columns + ["PWEIGHT", "tot_weight"]].mean()
    merged_df[selected_columns + ["PWEIGHT", "tot_weight"]] = merged_df[
        selected_columns + ["PWEIGHT", "tot_weight"]
    ].fillna(mean)

    dataset = Dataset(
        merged_df,
        columns,
        indicator,
        weight="PWEIGHT",
        binary_indicator=True,
    )
    return dataset


# doesn't use indicator variable
def cdc_dataloader(indicator, startdate, enddate):
    df = pd.read_csv("data/CDC/cdc_state_targets.csv")

    # Restrict CDC data to appropriate date range
    startdate = datetime.datetime.strptime(startdate, "%m-%d-%Y")
    enddate = datetime.datetime.strptime(enddate, "%m-%d-%Y")
    df["Date"] = pd.to_datetime(df["Date"])
    sub_df = df.loc[(df["Date"] >= startdate) & (df["Date"] <= enddate)].reset_index(
        drop=True
    )

    # Drop states that we don't care about in target distribution
    acs_characteristics = pd.read_csv("data/ACS/acs_state_characteristics.csv")
    states = acs_characteristics["state_name"].unique()
    sub_df = sub_df.loc[df["state_name"].isin(states)].reset_index(drop=True)

    # Aggregate over time period
    sub_df = (
        sub_df.groupby("state_name")
        .agg({"Administered_Dose1_Recip_18PlusPop_Pct": "mean"})
        .reset_index()
    )
    sub_df.rename(
        columns={"Administered_Dose1_Recip_18PlusPop_Pct": "RECVDVACC"}, inplace=True
    )

    # Get state proportions
    acs_weights = pd.read_csv("data/ACS/acs_weights_states_hps.csv")
    acs_weights = (
        acs_weights.groupby("state_name").agg({"tot_weight": "sum"}).reset_index()
    )
    acs_weights["tot_weight"] /= acs_weights["tot_weight"].sum()
    new_df = pd.merge(acs_weights, sub_df, on="state_name", how="left")

    dataset = Dataset(
        df=new_df,
        covs=[],
        indicator="RECVDVACC",
        weight="tot_weight",
        binary_indicator=True,
    )
    return dataset


"""
Extract monthly truths (proportion of households enrolled in SNAP)
per month, per state from 2019 to 2023.

Saves into data/SNAP/monthly_truths.csv.
"""


def load_snap_monthly_truths():
    sheets = ["NERO", "MARO", "SERO", "MWRO", "SWRO", "MPRO", "WRO"]
    excels = [20, 21, 22, 23]

    all_states = pd.DataFrame()

    for y in excels:
        for sheet in sheets:
            df = pd.read_excel(f"data/USDA/FY{y}.xlsx", sheet_name=sheet)
            df = (
                df[6:]
                .rename(
                    columns={
                        "National Data Bank Version 8.2 PUBLIC - Supplemental Nutrition Assistance Program": "period",
                        "Unnamed: 1": "households",
                    }
                )
                .reset_index()
            )
            df = df.loc[:, ["period", "households"]]
            df.dropna(subset=["period", "households"])

            df["state"] = np.nan
            df["state"] = df["state"].astype("object")
            for i in range(len(df)):
                if df.loc[i, "period"] in STATES:
                    df.loc[i, "state"] = df.loc[i, "period"]
                elif (
                    df.loc[i, "period"] == sheet or df.loc[i, "households"] == "--"
                ):  # delete agg data, delete missing data
                    df.loc[i, "period"] = np.nan
                elif i > 0:
                    df.loc[i, "state"] = df.loc[i - 1, "state"]

            month_year_df = df["period"].str.extract(
                r"(?P<month>[A-Za-z]+) (?P<year>\d{4})"
            )
            df["month"] = month_year_df["month"]
            df["year"] = month_year_df["year"]

            df = df.dropna(subset=["state", "period", "month", "year", "households"])
            df["month"] = df["month"].apply(
                month_num
            )  # avoid na issues by doing it here
            df = df.drop(columns=["period"])
            all_states = pd.concat([all_states, df])

    ### TODO: verify that 2020 is fine to use
    acs_weights = pd.read_csv("data/ACS/acs_weights_states_hps.csv")
    state_households = (
        acs_weights.groupby("state_name")
        .agg({"tot_weight_household": "sum"})
        .reset_index()
    )

    # convert SNAP households to percentage for each month
    all_states = pd.merge(
        all_states,
        state_households,
        left_on="state",
        right_on="state_name",
        how="right",
    )
    all_states["snap"] = all_states["households"] / all_states["tot_weight_household"]
    all_states = all_states.drop(
        columns=["households", "tot_weight_household", "state"]
    )

    all_states.to_csv(f"data/USDA/monthly_truths.csv", index=False)

    return all_states


"""
Load SNAP data for a given time period, weighted by days in each month.
startdate enddate in format "mm-dd-yyyy"
"""


def snap_gt_dataloader(indicator, startdate, enddate):
    startdate = datetime.datetime.strptime(startdate, "%m-%d-%Y")
    enddate = datetime.datetime.strptime(enddate, "%m-%d-%Y")

    # load SNAP data
    if os.path.exists("data/USDA/monthly_truths.csv"):
        df = pd.read_csv("data/USDA/monthly_truths.csv")
    else:
        df = load_snap_monthly_truths()

    # create start and end dates for each month
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df["end_date"] = df["date"] + pd.offsets.MonthEnd(0)

    result = {"state_name": [], "snap": []}

    # calc weighted average for each state
    for state in df["state_name"].unique():
        df_state = df[df["state_name"] == state]
        weight_sum = 0
        snap_weighted = 0

        # weights based on days overlap
        for _, month_row in df_state.iterrows():
            latest_start = max(startdate, month_row["date"])
            earliest_end = min(enddate, month_row["end_date"])
            delta = (earliest_end - latest_start).days + 1

            if delta > 0:
                snap_weighted += delta * month_row["snap"]
                weight_sum += delta

        if weight_sum > 0:
            weighted_snap = snap_weighted / weight_sum
            result["state_name"].append(state)
            result["snap"].append(weighted_snap)

    result_df = pd.DataFrame(result)

    # get state weights
    acs_weights = pd.read_csv("data/ACS/acs_weights_states_hps.csv")
    acs_weights = (
        acs_weights.groupby("state_name").agg({"tot_weight": "sum"}).reset_index()
    )
    acs_weights["tot_weight"] /= acs_weights["tot_weight"].sum()

    new_df = pd.merge(acs_weights, result_df, on="state_name", how="left")

    dataset = Dataset(
        df=new_df, covs=[], indicator="snap", weight="tot_weight", binary_indicator=True
    )
    return dataset


def ins_gt_dataloader(indicator, evaluation_group, startdate, enddate):
    """
    Load insurance (Medicaid) data for a given time period.

    Args:
        indicator: Not used, kept for consistency with other loaders
        startdate: Start date in format "mm-dd-yyyy"
        enddate: End date in format "mm-dd-yyyy"
    Returns:
        Dataset with Medicaid insurance participation as proportion, averaged across years if multiple years
    """
    startdate = datetime.datetime.strptime(startdate, "%m-%d-%Y")
    enddate = datetime.datetime.strptime(enddate, "%m-%d-%Y")

    # Get years that overlap with the date range
    ### TODO: check if we should do this weighted thing, or if we should restrict to single years
    ### + if restricting to single year, should I also modify the HPS dataloader?
    years = list(range(startdate.year, enddate.year + 1))
    if len(years) > 1:
        print("Multiple years detected. Insurance rates will be averaged across years.")

    dfs = []
    for year in years:
        df = pd.read_csv(f"data/ACS/acs_insurance_state_race_{year}.csv")
        df = df[df["state_name"].isin(STATES)]
        # Calc days overlap for this year
        year_start = max(startdate, datetime.datetime(year, 1, 1))
        year_end = min(enddate, datetime.datetime(year, 12, 31))
        df["weight"] = (year_end - year_start).days + 1
        dfs.append(df)

    # Combine all years and calc weighted average
    df = pd.concat(dfs, ignore_index=True)
    df_avg = (
        df.groupby(evaluation_group)
        .apply(lambda x: np.average(x["acs_medicaid_insurance"], weights=x["weight"]))
        .reset_index(name="acs_medicaid_insurance")
    )

    # Get state proportions from ACS weights
    acs_weights = pd.read_csv("data/ACS/acs_weights_states_hps.csv")
    acs_weights = (
        acs_weights.groupby(evaluation_group).agg({"tot_weight": "sum"}).reset_index()
    )
    acs_weights["tot_weight"] /= acs_weights["tot_weight"].sum()

    new_df = pd.merge(
        acs_weights,
        df_avg[evaluation_group + ["acs_medicaid_insurance"]],
        on=evaluation_group,
        how="left",
    ).rename(columns={"acs_medicaid_insurance": "medicaid_ins"})

    dataset = Dataset(
        df=new_df,
        covs=[],
        indicator="medicaid_ins",
        weight="tot_weight",
        binary_indicator=True,
    )
    return dataset


def get_target_samples(n_samples):
    """
    Generate synthetic training samples for demonstration purposes.
    This function simulates a dataset with covariates and a target variable.
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.uniform(0, 2, n_samples)
    Y = X**2 + np.random.normal(0, 0.5, n_samples)  # Simulated target variable
    return {"X": X, "Y": Y, "W": np.ones(n_samples) / n_samples}


def get_train_conditional_pdf(x):
    bin_start = -1
    bin_end = 6
    theta = 2.0
    scale = 0.5

    def actual_log_partition_function(x):
        log_partition_function = np.log(
            quad(
                lambda z: norm.pdf(z, loc=x**2, scale=scale)
                * np.exp(theta * (x**2) * z),
                bin_start,
                bin_end,
            )[0]
        )
        return log_partition_function

    log_partition_function_value = actual_log_partition_function(x)
    pdf = lambda z: norm.pdf(z, loc=x**2, scale=scale) * np.exp(
        theta * (x**2) * z - log_partition_function_value
    )
    return pdf


def get_train_samples(n_samples):
    np.random.seed(153)

    X_min = 0.0
    X_max = 2.0
    X = np.random.uniform(X_min, X_max, n_samples)
    U = np.random.uniform(0, 1, n_samples)
    scale = 0.5
    theta = 2.0
    bin_start = -1
    bin_end = 6

    def actual_log_partition_function(x):
        log_partition_function = np.log(
            quad(
                lambda z: norm.pdf(z, loc=x**2, scale=scale)
                * np.exp(theta * (x**2) * z),
                bin_start,
                bin_end,
            )[0]
        )
        return log_partition_function

    X_lin = np.linspace(X_min, X_max, 500)
    log_partition_vals = np.array([actual_log_partition_function(x) for x in X_lin])
    interp_log_partition = interp1d(
        X_lin, log_partition_vals, bounds_error=False, fill_value="extrapolate"
    )

    def get_inverse_cdf(x):
        log_partition_function_value = interp_log_partition(x)
        pdf = lambda z: norm.pdf(z, loc=x**2, scale=scale) * np.exp(
            theta * (x**2) * z - log_partition_function_value
        )
        zs = np.linspace(bin_start, bin_end, 200)
        pdf_vals = pdf(zs)
        cdf_vals = np.cumsum(pdf_vals * (zs[1] - zs[0]))  # Numerical integration
        inverse_cdf = interp1d(
            cdf_vals, zs, bounds_error=False, fill_value="extrapolate"
        )
        return inverse_cdf

    inverse_cdfs = []

    for i in range(n_samples):
        inverse_cdfs.append(get_inverse_cdf(X[i]))
    Y = np.array([inverse_cdfs[i](U[i]) for i in range(n_samples)])
    return {"X": X, "Y": Y, "W": np.ones(n_samples) / n_samples}


def load_census_data(indicator):
    if indicator in ["medicaid_ins", "RECVDVACC", "snap"]:
        census_dataset = acs_dataloader()
    elif indicator == "synthetic":
        census_dataset = sim_target_covariates_dataloader()
    return census_dataset


def sim_target_covariates_dataloader():
    n_samples = 50
    X = np.linspace(0, 2, n_samples)
    W = np.ones(n_samples) / n_samples  # Uniform weights for synthetic data
    df = pd.DataFrame({"X": X, "W": W})
    dataset = Dataset(
        df=df,
        covs=["X"],
        indicator=None,
        weight="W",
        binary_indicator=None,
    )
    return dataset


def sim_dataloader(indicator):
    """
    Load synthetic survey data for a given time period.
    """

    train_samples = get_train_samples(20000)
    train_samples["synthetic"] = train_samples.pop("Y")

    df = pd.DataFrame(train_samples)
    dataset = Dataset(
        df=df,
        covs=["X"],
        indicator=indicator,
        weight="W",
        binary_indicator=False,
    )
    return dataset


def sim_gt_dataloader(indicator):
    """
    Load synthetic ground truth data for a given time period.
    This is a placeholder function that simulates loading ground truth data.
    """

    X = np.linspace(0, 2, 50)

    cond_mean_Y = X**2

    df = pd.DataFrame({"X": X, indicator: cond_mean_Y, "W": np.ones_like(X) / len(X)})

    dataset = Dataset(
        df=df,
        covs=[],
        indicator=indicator,
        weight="W",
        binary_indicator=False,
    )
    return dataset


INDICATOR_TO_SURVEY_LOADER = {
    "RECVDVACC": hps_dataloader,
    "snap": hps_dataloader,
    "medicaid_ins": hps_dataloader,
    "synthetic": sim_dataloader,
}
INDICATOR_TO_GT_LOADER = {
    "RECVDVACC": cdc_dataloader,
    "snap": snap_gt_dataloader,
    "medicaid_ins": ins_gt_dataloader,
    "synthetic": sim_gt_dataloader,
}


def load_state_ground_truth_data(
    evaluation_group, indicator="RECVDVACC", startdate="2-17-2021", enddate="3-1-2021"
):
    dataloader = INDICATOR_TO_GT_LOADER[indicator]
    if indicator in ["RECVDVACC", "snap"]:
        dataset = dataloader(indicator=indicator, startdate=startdate, enddate=enddate)
    elif indicator == "medicaid_ins":
        dataset = dataloader(
            indicator=indicator,
            evaluation_group=evaluation_group,
            startdate=startdate,
            enddate=enddate,
        )
    else:
        dataset = dataloader(indicator=indicator)
    return dataset


def load_national_ground_truth_data(
    indicator="RECVDVACC",
    moment_group="state_name",
    evaluation_group=["state_name"],
    startdate="2-17-2021",
    enddate="3-1-2021",
    n_constraints=3,
):
    attrs = list(set(evaluation_group.copy() + [moment_group]))
    dataset = load_state_ground_truth_data(
        indicator=indicator,
        evaluation_group=attrs,
        startdate=startdate,
        enddate=enddate,
    )

    national_df = dataset.df.copy(deep=True)

    if indicator in ["RECVDVACC", "snap", "medicaid_ins"]:

        all_states_in_regions_dic = []
        for REGION in REGIONS_DIC[moment_group][n_constraints].keys():
            all_states_in_regions_dic.extend(
                REGIONS_DIC[moment_group][n_constraints][REGION]
            )
        data_indicator = dataset.df[moment_group].isin(all_states_in_regions_dic)
        y_all_region = dataset.df.loc[data_indicator, indicator]
        r_all_region = dataset.df.loc[data_indicator, dataset.weight]
        national_df[indicator] = (
            y_all_region * r_all_region
        ).sum() / r_all_region.sum()

        regions = REGIONS_DIC[moment_group][n_constraints]
        if moment_group in evaluation_group:
            for region in regions.keys():
                region_indicator = dataset.df[moment_group].isin(regions[region])
                r_region = dataset.df.loc[region_indicator][dataset.weight]
                y_region = dataset.df.loc[region_indicator][indicator]
                mean_region = (y_region * r_region).sum() / r_region.sum()
                x = national_df[moment_group].isin(regions[region])
                national_df.loc[x, indicator] = mean_region

        national_dataset = Dataset(
            df=national_df,
            covs=[],
            indicator=indicator,
            weight=dataset.weight,
            binary_indicator=dataset.binary_indicator,
        )

        return national_dataset, regions
    else:
        national_df[indicator] = 4 / 3
        national_dataset = Dataset(
            df=national_df,
            covs=[],
            indicator=indicator,
            weight=dataset.weight,
            binary_indicator=False,
        )
        regions = {"All": list(national_df["X"])}
        return national_dataset, regions


def load_online_survey_data(
    indicator="RECVDVACC", startdate="2-17-2021", enddate="3-1-2021"
):
    dataloader = INDICATOR_TO_SURVEY_LOADER[indicator]
    if indicator in ["RECVDVACC", "snap", "medicaid_ins"]:
        dataset = dataloader(indicator=indicator, startdate=startdate, enddate=enddate)
    else:
        # synthetic data doesn't use startdate and enddate
        dataset = dataloader(indicator=indicator)
    return dataset
