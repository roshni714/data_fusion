from splines import CovariateSpline
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_covariate_ratio_classifier(online_survey_dataset, census_dataset):
    """
    Computes the covariate ratio for the given online survey dataset, census dataset
    Args:
        online_survey_dataset: The online survey dataset.
        census_dataset: The census dataset.
    """
    X_census, r = census_dataset.get_data(normalize_weight=True)
    X_online, _, _ = online_survey_dataset.get_data()

    sample_idxs = np.random.choice(
        X_census.shape[0], size=X_online.shape[0], replace=True, p=r
    )
    X_census_samples = X_census[sample_idxs]

    classifier = LogisticRegression()
    X = np.vstack((X_online, X_census_samples))
    X_mean = np.mean(X, axis=0).reshape(1, -1)
    X_std = np.std(X, axis=0).reshape(1, -1) + 1e-6

    X = (X - X_mean) / X_std
    y = np.hstack((np.zeros(X_online.shape[0]), np.ones(X_census_samples.shape[0])))

    classifier.fit(X, y)

    def covariate_ratio(X):
        X = (X - X_mean) / X_std
        proba = classifier.predict_proba(X)[:, 0]
        ratio = (1 - proba) / proba
        return ratio

    return covariate_ratio


def get_log_partition_function_samples(basis_online, gamma):
    """
    Computes the log partition function for the given basis functions and gamma parameters.

    Args:
        basis_online (torch.Tensor): The basis functions for the online survey data.
        gamma (torch.nn.Parameter): The parameters to compute the log partition function.

    Returns:
        torch.Tensor: The log partition function values.
    """
    return torch.log(torch.exp(basis_online @ gamma).mean())


def get_covariate_balance(
    online_survey_dataset, census_dataset, n_epochs=100, seed=12359038
):
    X_online, _, _ = online_survey_dataset.get_data()
    X_census, r = census_dataset.get_data(normalize_weight=True)

    spline = CovariateSpline(census_dataset)

    # Get the basis functions for the online survey
    basis_online = spline.get_spline(X_online)
    basis_census = spline.get_spline(X_census)

    f_bar = np.sum(basis_census * r[:, None], axis=0)
    f_bar = torch.tensor(f_bar, dtype=torch.float64).reshape(-1, 1)

    basis_online = torch.tensor(basis_online, dtype=torch.float64)

    gamma_init = torch.nn.Parameter(
        torch.tensor(
            np.random.normal(size=(f_bar.shape[0], 1), scale=0.01),
            requires_grad=True,
        )
    )

    def loss_function(gamma):
        avg_log_partition_function = get_log_partition_function_samples(
            basis_online, gamma
        )
        avg_log_partition_function = avg_log_partition_function.reshape((1, 1))
        loss = avg_log_partition_function - torch.matmul(f_bar.T, gamma)
        return loss

    optimizer = torch.optim.Adam([gamma_init], lr=0.05)
    torch.manual_seed(seed)

    print("Starting optimization for covariate balance...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_function(gamma_init)
        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss.item(),
        )
        loss.backward()
        optimizer.step()

    beta = (
        torch.exp(get_log_partition_function_samples(basis_online, gamma_init))
        .detach()
        .numpy()
    )
    gamma = gamma_init.detach().numpy().reshape(1, spline.d_covariate)

    def covariate_ratio(X):
        f_X = spline.get_spline(X)
        gamma = gamma_init.detach().numpy().reshape(1, -1)
        r = np.exp(f_X @ gamma.T).flatten() / beta
        return r

    return gamma, covariate_ratio
