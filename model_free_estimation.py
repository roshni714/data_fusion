from estimation import (
    ConstraintSpline,
    get_constraint_vector,
)
import numpy as np
import torch
import pandas as pd


def get_log_partition_function_binary_outcome_constraint_spline(
    X, theta, predictor, constraint_spline, state_col
):
    predictions = torch.tensor(predictor(X))
    basis_vectors_1 = torch.tensor(constraint_spline.get_spline(state_col, np.ones(1)))
    basis_vectors_0 = torch.tensor(constraint_spline.get_spline(state_col, np.zeros(1)))
    log_partition_function = torch.log(
        torch.exp(torch.transpose(basis_vectors_1, 2, 1) @ theta)
        * predictions.reshape(-1, 1, 1)
        + torch.exp(torch.transpose(basis_vectors_0, 2, 1) @ theta)
        * (1 - predictions.reshape(-1, 1, 1))
    )
    return log_partition_function


def get_model_free_exponential_family_state_estimates(
    indicator,
    predictor,
    census_dataset,
    national_gt_dataset,
    regions,
    evaluation_group,
    moment_group,
):
    """
    Get model-free state estimates using an exponential family distribution.
    """
    if indicator in ["RECVDVACC", "medicaid_ins", "snap"]:
        return get_model_free_exponential_family_state_estimates_binary(
            indicator,
            predictor,
            census_dataset,
            national_gt_dataset,
            regions,
            evaluation_group,
            moment_group,
        )
    else:
        return get_model_free_exponential_family_state_estimates_continuous(
            indicator,
            predictor,
            census_dataset,
            national_gt_dataset,
            regions,
            evaluation_group,
            moment_group,
        )


def get_log_partition_function_continuous_outcome_constraint_spline(
    theta, pdf_grid, y_grid, constraint_spline, state_col
):
    """
    X is n x d
    theta is d_tilting x 1
    pdf_grid is n x 1 x m
    y_grid is m
    Returns a vector of shape n x 1
    """
    eta_vectors = torch.tensor(
        constraint_spline.get_spline(state_col, y_grid)
    )  # n x d_tilting x m
    transposed_eta_vectors = torch.tensor(
        np.transpose(eta_vectors, (0, 2, 1))[:, :, None, :]
    )  # n x m x 1 x d_tilting
    exponential = torch.exp(torch.matmul(transposed_eta_vectors, theta)).squeeze(
        -1
    )  # n x m x 1
    log_partition_function = torch.log(
        torch.trapezoid(
            (exponential * torch.tensor(pdf_grid)).squeeze(-1),
            torch.tensor(y_grid.flatten()),
            dim=1,
        )
    )  # n,
    return log_partition_function.reshape(-1, 1)


def get_model_free_exponential_family_state_estimates_continuous(
    indicator,
    predictor,
    census_dataset,
    national_gt_dataset,
    regions,
    evaluation_group,
    moment_group,
    n_epochs=80,
    seed=12359038,
):
    """
    Get model-free state estimates using an exponential family distribution for continuous indicators.
    """
    X, r, state_col = census_dataset.get_data(
        normalize_weight=True, return_state_col=moment_group
    )
    # Initialize the constraint spline
    constraint_spline = ConstraintSpline(
        regions=regions,
        moment_group=moment_group,
    )
    constraint_m, _ = get_constraint_vector(national_gt_dataset, moment_group, regions)
    constraint_m = torch.tensor(constraint_m, dtype=torch.float64)

    theta_init = torch.nn.Parameter(
        torch.tensor(
            np.random.normal(size=(constraint_spline.d_constraints, 1), scale=0.25),
            requires_grad=True,
        )
    )

    densities = predictor(X)
    y_grid = np.linspace(
        densities[0].outcome_range[0].item(), densities[0].outcome_range[1].item(), 100
    )
    pdf_grid = np.array([density.pdf(y_grid) for density in densities]).reshape(
        X.shape[0], len(y_grid), 1
    )  # n x m x 1

    def loss_function(theta):
        log_partition_function = (
            get_log_partition_function_continuous_outcome_constraint_spline(
                theta, pdf_grid, y_grid, constraint_spline, state_col
            )
        )
        avg_log_partition_function = (
            log_partition_function
            * torch.tensor(r).reshape(log_partition_function.shape)
        ).sum() / torch.tensor(r).sum()
        avg_log_partition_function = avg_log_partition_function.reshape((1, 1))
        loss = avg_log_partition_function - torch.matmul(constraint_m.T, theta)
        return loss

    optimizer = torch.optim.Adam([theta_init], lr=0.1)
    torch.manual_seed(seed)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_function(theta_init)
        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss.item(),
            "Theta:",
            theta_init.detach().numpy().flatten(),
        )
        loss.backward()
        optimizer.step()

    final_theta = theta_init.detach()
    print("Final theta:", final_theta)

    log_partition_function = (
        get_log_partition_function_continuous_outcome_constraint_spline(
            final_theta, pdf_grid, y_grid, constraint_spline, state_col
        ).numpy()
    )  # nx 1
    eta_vectors = constraint_spline.get_spline(state_col, y_grid)  # n x 1 x m

    transposed_eta_vectors = np.transpose(eta_vectors, (0, 2, 1))[
        :, :, None, :
    ]  # n x m x 1 x d_constraints
    exponential = np.exp(
        np.matmul(transposed_eta_vectors, final_theta.numpy())
        - log_partition_function[:, :, None, None]
    ).squeeze(
        -1
    )  # n x m x 1
    probs = pdf_grid * exponential

    conditional_mean = np.trapezoid(
        y_grid * probs.squeeze(-1), y_grid.flatten(), axis=1
    )
    df = pd.DataFrame({"X": X.flatten(), indicator: conditional_mean})
    return df


def get_model_free_exponential_family_state_estimates_binary(
    indicator,
    predictor,
    census_dataset,
    national_gt_dataset,
    regions,
    evaluation_group,
    moment_group,
    n_epochs=80,
    seed=12359038,
):
    """
    Get model-free state estimates using an exponential family distribution.
    """

    X, r, state_col = census_dataset.get_data(
        normalize_weight=True, return_state_col=moment_group
    )
    # Initialize the constraint spline
    constraint_spline = ConstraintSpline(
        regions=regions,
        moment_group=moment_group,
    )
    constraint_m, _ = get_constraint_vector(national_gt_dataset, moment_group, regions)
    constraint_m = torch.tensor(constraint_m, dtype=torch.float64)

    theta_init = torch.nn.Parameter(
        torch.tensor(
            np.random.normal(size=(constraint_spline.d_constraints, 1), scale=0.25),
            requires_grad=True,
        )
    )

    def loss_function(theta):
        log_partition_function = (
            get_log_partition_function_binary_outcome_constraint_spline(
                X, theta, predictor, constraint_spline, state_col
            )
        )
        avg_log_partition_function = (
            log_partition_function
            * torch.tensor(r).reshape(log_partition_function.shape)
        ).sum() / torch.tensor(r).sum()
        avg_log_partition_function = avg_log_partition_function.reshape((1, 1))
        loss = avg_log_partition_function - torch.matmul(constraint_m.T, theta)
        return loss

    optimizer = torch.optim.Adam([theta_init], lr=0.1)
    torch.manual_seed(seed)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_function(theta_init)
        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss.item(),
            "Theta:",
            theta_init.detach().numpy().flatten(),
        )
        loss.backward()
        optimizer.step()

    final_theta = theta_init.detach()
    print("Final theta:", final_theta)

    eta_vectors_1 = constraint_spline.get_spline(state_col, np.ones(1))
    predictions = predictor(X)

    log_partition_function = (
        get_log_partition_function_binary_outcome_constraint_spline(
            X, final_theta, predictor, constraint_spline, state_col
        )
    )

    group_to_indexes = (
        census_dataset.df.groupby(evaluation_group)
        .apply(lambda x: x.index.tolist())
        .to_dict()
    )

    final_theta = final_theta.detach().numpy()

    estimates = []
    for group, indexes in group_to_indexes.items():
        res = {}
        for i, name in enumerate(evaluation_group):
            if type(group) is str:
                res[name] = group
            elif type(group) is tuple:
                res[name] = group[i]
        adjustment = np.exp(
            np.transpose(eta_vectors_1, (0, 2, 1)) @ final_theta
            - log_partition_function.numpy()
        ).flatten()
        res.update(
            {
                indicator: (
                    (
                        predictions[indexes].flatten()
                        * r[indexes].flatten()
                        * adjustment[indexes]
                    ).sum()
                    / r[indexes].sum()
                ).item(),
            }
        )
        estimates.append(res)
    df = pd.DataFrame(estimates)
    return df
