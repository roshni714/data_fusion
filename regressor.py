import torch
from data_utils import standardize
import tqdm
import copy
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import SplineTransformer
from statsmodels.nonparametric.kde import KDEUnivariate


class NonparametricConditionalDistribution:
    """
    Represents a nonparametric conditional distribution.
    """

    def __init__(self, pdf_function, outcome_range):
        """
        Initialize a ConditionalDistribution object.

        :param pdf_function: The probability density function.
        :type pdf_function: callable
        :param outcome_range: The range of possible outcomes.
        :type outcome_range: tuple
        """
        self.pdf_function = pdf_function
        self.outcome_range = outcome_range

    def pdf(self, z):
        """
        Probability density function (pdf) of conditional distribution.

        :param z: The input value.
        :type z: float or numpy.ndarray
        :return: The probability density at z.
        :rtype: float or numpy.ndarray
        """
        return self.pdf_function(z)


def fit_carrier_function(y):
    kde = KDEUnivariate(y)
    kde.fit(fft=False)
    return kde


def get_cond_density_estimator(
    train_dataset,
    validation_dataset,
    truncate_lower_bound=-2,
    truncate_upper_bound=8,
    n_bins=100,
    n_knots=4,
    degree=3,
    n_epochs=4000,
    seed=123456,
    device="cpu",
):
    """
    Apply the Lindsey's method for marginal density estimation (Efron & Tibshirani 1996).

    :param train_dataset: The training dataset for which to apply the Lindsey method.
    :type train_dataset: Dataset
    :param validation_dataset: The validation dataset for which to apply the Lindsey method.
    :type validation_dataset: Dataset
    :param n_bins: The number of bins to use for the outcome space.
    :type n_bins: int
    :param n_knots: The number of knots to use for the spline basis functions.
    :type n_knots: int
    :param degree: The degree of the spline basis functions.
    :type degree: int
    :param n_epochs: The number of epochs to train the density estimator.
                     Defaults to 300.
    :type n_epochs: int
    :return: A callable that maps a numpy array to a numpy array of NonparametricConditionalDistribution objects.
    :rtype: Callable[[np.ndarray], np.ndarray]
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, r_train = train_dataset.get_data(normalize_weight=True)
    X_val, y_val, r_val = validation_dataset.get_data(normalize_weight=True)

    y_train = np.clip(y_train, a_min=truncate_lower_bound, a_max=truncate_upper_bound)
    y_val = np.clip(y_val, a_min=truncate_lower_bound, a_max=truncate_upper_bound)

    # Get data. Standardize Y.
    X_train, X_mean, X_std = standardize(X_train)
    # y_train, y_mean, y_std = standardize(y_train)
    X_val = (X_val - X_mean) / X_std
    # y_val = (y_val - y_mean) / y_std

    # Add intercept to X
    # X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    # X_val = np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis=1)

    # Bin the outcome space.
    bin_ends = np.linspace(truncate_lower_bound, truncate_upper_bound, n_bins)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    # Fit carrier density (Y marginal) and evaluate on bin boundaries.
    kde = fit_carrier_function(y_train)
    front = kde.evaluate(bin_ends)

    # def spline_transform(y):
    #     zs = np.linspace(
    #         truncate_lower_bound, truncate_upper_bound, 5)
    #     splines = []
    #     for z in zs:
    #         splines.append((y <= z).astype(float))  # n x 1
    #     return np.concatenate(splines, axis=1) #n x k

    # # # Get B-spline basis functions.
    spline = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=True)
    spline.fit(y_train.reshape(-1, 1))

    # Get basis representation of bin boundaries.
    bin_basis_elements = spline.transform(bin_ends.reshape(-1, 1))  # n_bins x k
    k = bin_basis_elements.shape[1]
    bin_basis_elements = torch.tensor(
        bin_basis_elements[:, None, :], dtype=torch.float64
    ).to(
        device
    )  # n_bins x 1 x k

    # Get basis representation of sampled Y values.
    basis_matrix_train = torch.tensor(
        spline.transform(y_train.reshape(-1, 1))[:, None, :],
        dtype=torch.float64,
    )  # n x 1 x k

    basis_matrix_val = torch.tensor(
        spline.transform(y_val.reshape(-1, 1))[:, None, :], dtype=torch.float64
    )  # n x 1 x k

    # # Add intercept to X_train and X_val
    X_train = np.concatenate([X_train, np.ones((n_train, 1))], axis=1)
    X_val = np.concatenate([X_val, np.ones((n_val, 1))], axis=1)

    X_train = torch.tensor(X_train, dtype=torch.float64).reshape(
        X_train.shape[0], X_train.shape[1], 1
    )
    r_train = torch.tensor(r_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)

    X_val = torch.tensor(X_val, dtype=torch.float64).reshape(
        X_val.shape[0], X_val.shape[1], 1
    )
    r_val = torch.tensor(r_val, dtype=torch.float64)
    y_val = torch.tensor(y_val, dtype=torch.float64)

    bin_ends = torch.tensor(bin_ends, dtype=torch.float64).to(device)  # n_bins
    front = torch.tensor(front, dtype=torch.float64).to(device)  # n_bins

    theta = torch.nn.Parameter(
        torch.tensor(
            np.random.uniform(-1.0, 1.0, k * X_train.shape[1]).reshape(
                k, X_train.shape[1]
            ),
            dtype=torch.float64,
            device=device,
        )
    )

    # unscaled_bin_ends = bin_ends * y_std + y_mean

    def glm_nll(theta, X, basis_matrix):
        params = torch.matmul(theta, X.to(device)).reshape(
            X.shape[0], k, 1
        )  # n x k x 1
        tiled_params = torch.tile(
            params[:, None, :, :], dims=(1, n_bins, 1, 1)
        )  # n x n_bins x k x 1
        tiled_basis = torch.tile(
            bin_basis_elements[None, :, :, :].to(device), dims=(X.shape[0], 1, 1, 1)
        )  # n x n_bins x 1 x k
        res1 = torch.matmul(tiled_basis, tiled_params).squeeze(-1)  # n x n_bins x 1
        tilting = torch.exp(res1)

        tiled_front = torch.tile(
            front[None, :, None].to(device), dims=(X.shape[0], 1, 1)
        )  # n x n_bins x 1
        final_matrix = tiled_front * tilting  # n x n_bins x 1
        log_norm_constant = torch.log(
            torch.trapezoid(y=final_matrix, x=bin_ends.to(device), dim=1)
        )  # n x 1

        actual_res1 = torch.matmul(basis_matrix.to(device), params)  # n x 1 x 1
        nll = -actual_res1.squeeze(-1) + log_norm_constant  # n x 1
        return nll.flatten()

    optimizer = torch.optim.Adam([theta], lr=1e-2)
    batch_size = int(n_train / 3)
    print("Fitting conditional densities vs glm spline method...")
    pbar = tqdm.tqdm(list(range(n_epochs)))

    thetas = []
    val_losses = []
    for epoch in pbar:
        if epoch % 25 == 0:
            val_loss = torch.sum(
                glm_nll(theta, X_val, basis_matrix_val) * r_val.to(device)
            )

            val_losses.append(val_loss.detach().item())
            thetas.append(theta.detach().clone().cpu())

        idx = np.random.choice(n_train, size=batch_size, replace=True)
        optimizer.zero_grad()
        loss = torch.sum(
            glm_nll(theta, X_train[idx, :], basis_matrix_train[idx, :])
            * r_train[idx].to(device)
        )
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})

    best_model_idx = np.argmin(val_losses)
    final_theta = thetas[best_model_idx]
    print("Final Theta: {}".format(final_theta))

    bin_basis_elements = bin_basis_elements.cpu()
    front = front.cpu()
    bin_ends = bin_ends.cpu()
    final_theta = final_theta.cpu()

    def helper(X_test):
        X_test = (X_test - X_mean) / X_std
        if X_test.shape[1] == final_theta.shape[1] - 1:
            X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)

        nat_param = torch.matmul(
            final_theta,
            torch.tensor(X_test, dtype=torch.float64).reshape(
                X_test.shape[0], X_test.shape[1], 1
            ),
        )  # n x k x 1

        tiled_params = torch.tile(
            nat_param[:, None, :, :], dims=(1, n_bins, 1, 1)
        )  # n x n_bins x k x 1
        tiled_basis = torch.tile(
            bin_basis_elements[None, :, :, :], dims=(X_test.shape[0], 1, 1, 1)
        )  # n x n_bins x 1 x k
        res1 = torch.matmul(tiled_basis, tiled_params)  # n x n_bins x 1 x1
        tilting = torch.exp(res1).squeeze(-1)
        tiled_front = torch.tile(
            front[None, :, None], dims=(X_test.shape[0], 1, 1)
        )  # n x n_bins x 1
        final_matrix = tiled_front * tilting  # n x n_bins x 1
        norm_constant = torch.trapezoid(y=final_matrix, x=bin_ends, dim=1)
        tiled_norm_constant = torch.tile(
            norm_constant[:, None, :].to(device), dims=(1, n_bins, 1)
        )

        pdf_matrix = final_matrix / tiled_norm_constant
        pdf_matrix = pdf_matrix.detach()

        cond_dists = []

        for i in range(len(X_test)):
            pdf_function = interp1d(
                bin_ends,
                pdf_matrix[i].flatten(),
                bounds_error=False,
                fill_value=(0.0, 1e-6),
            )

            cond_dists.append(
                NonparametricConditionalDistribution(
                    pdf_function,
                    outcome_range=(bin_ends[0], bin_ends[-1]),
                )
            )
        return np.array(cond_dists)

    return helper


def get_continuous_conditional_predictor(
    train_dataset,
    validation_dataset,
    covariate_ratio=None,
    n_hidden_units=128,
    n_layers=1,
    n_epochs=250,
    lr=0.005,
    device="cpu",
    seed=0,
    verbose=False,
):
    # Get data first to check if empty
    X_train, y_train, _ = train_dataset.get_data(normalize_weight=False)
    r_train = (
        np.ones_like(y_train) / y_train.shape[0]
    )  # Default weights if not provided
    if covariate_ratio is not None:
        # Apply covariate ratio if provided
        r_train = covariate_ratio(X_train)

    # Check if we have enough data to train
    if len(X_train) == 0:
        if verbose:
            print(
                f"Warning: Insufficient data (n={len(X_train)}) for training. Using mean prediction."
            )
        # Return a simple predictor that always predicts the weighted mean
        mean_pred = np.average(y_train, weights=r_train) if len(y_train) > 0 else 0.5
        return lambda x: np.full(len(x), mean_pred)

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_val, y_val, _ = validation_dataset.get_data(normalize_weight=False)
    r_val = np.ones_like(y_val) / y_val.shape[0]
    if covariate_ratio is not None:
        # Apply covariate ratio if provided
        r_val = covariate_ratio(X_val)
    X_train, X_mean, X_std = standardize(X_train)
    X_val = (X_val - X_mean) / X_std
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train, y_mean, y_std = standardize(y_train)
    y_val = (y_val - y_mean) / y_std
    y_val = torch.tensor(y_val, dtype=torch.float32)
    r_val = torch.tensor(r_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    r_train = torch.tensor(r_train, dtype=torch.float32)

    d = X_train.shape[1]

    model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
    for _ in range(n_layers - 1):
        model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
        model_list.append(torch.nn.ReLU())
    model_list.append(torch.nn.Linear(n_hidden_units, 1))
    predictor = torch.nn.Sequential(*model_list).to(device)
    loss_f = torch.nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    batch_size = 2048
    pbar = tqdm.tqdm(list(range(n_epochs)))
    val_losses = []
    models = []
    predictor = predictor.to(device)

    for epoch in pbar:
        if epoch % 10 == 0:
            predictor.eval()
            predictions = predictor(X_val.to(device)).squeeze()
            unweighted_loss = loss_f(predictions, y_val.to(device))
            weights = r_val.to(device)
            val_loss = torch.mean(unweighted_loss * weights)
            val_losses.append(val_loss.detach().item())
            # ideally do torch.save to save checkpoints. Avoid saving so many models
            # in memory. And more correct.
            models.append(copy.deepcopy(predictor.cpu()))

        predictor.train()
        idx = np.random.choice(len(X_train), size=batch_size, replace=True)
        optimizer.zero_grad()
        predictions = predictor(X_train[idx, :].to(device)).squeeze()
        unweighted_loss = loss_f(predictions, y_train[idx].to(device))
        weights = r_train[idx].to(device)
        loss = torch.mean(unweighted_loss * weights)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"val loss": val_losses[-1]})

    best_model_idx = np.argmin(val_losses)
    print("best model", best_model_idx)
    final_predictor = models[best_model_idx]
    final_predictor.eval()
    final_predictor = final_predictor.cpu()

    def estimator(X_test):
        X_test = (X_test - X_mean) / X_std
        predictions = (
            (final_predictor(torch.Tensor(X_test)).reshape(X_test.shape[0], 1))
            .detach()
            .numpy()
            .flatten()
        ) * y_std + y_mean

        return predictions

    return estimator


def get_binary_conditional_predictor(
    train_dataset,
    validation_dataset,
    covariate_ratio=None,
    n_hidden_units=128,
    n_layers=1,
    n_epochs=250,
    lr=0.005,
    device="cpu",
    seed=0,
    verbose=False,
):
    # Get data first to check if empty
    X_train, y_train, r_train = train_dataset.get_data()
    if covariate_ratio is not None:
        # Apply covariate ratio if provided
        r_train = covariate_ratio(X_train)

    # Check if we have enough data to train
    if len(X_train) == 0:
        if verbose:
            print(
                f"Warning: Insufficient data (n={len(X_train)}) for training. Using mean prediction."
            )
        # Return a simple predictor that always predicts the weighted mean
        mean_pred = np.average(y_train, weights=r_train) if len(y_train) > 0 else 0.5
        return lambda x: np.full(len(x), mean_pred)

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_val, y_val, r_val = validation_dataset.get_data()
    if covariate_ratio is not None:
        # Apply covariate ratio if provided
        r_val = covariate_ratio(X_val)
    X_train, X_mean, X_std = standardize(X_train)
    X_val = (X_val - X_mean) / X_std
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    assert train_dataset.binary_indicator
    assert validation_dataset.binary_indicator
    y_val = torch.tensor(y_val, dtype=torch.float32)
    r_val = torch.tensor(r_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    r_train = torch.tensor(r_train, dtype=torch.float32)

    d = X_train.shape[1]

    model_list = [torch.nn.Linear(d, n_hidden_units), torch.nn.ReLU()]
    for _ in range(n_layers - 1):
        model_list.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
        model_list.append(torch.nn.ReLU())
    model_list.append(torch.nn.Linear(n_hidden_units, 1))
    model_list.append(torch.nn.Sigmoid())
    predictor = torch.nn.Sequential(*model_list).to(device)
    loss_f = torch.nn.BCELoss(reduction="none")

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    batch_size = 2048
    pbar = tqdm.tqdm(list(range(n_epochs)))
    val_losses = []
    models = []
    predictor = predictor.to(device)

    for epoch in pbar:
        if epoch % 10 == 0:
            predictor.eval()
            predictions = predictor(X_val.to(device)).squeeze()
            unweighted_loss = loss_f(predictions, y_val.to(device))
            weights = r_val.to(device)
            val_loss = torch.mean(unweighted_loss * weights)
            val_losses.append(val_loss.detach().item())
            # ideally do torch.save to save checkpoints. Avoid saving so many models
            # in memory. And more correct.
            models.append(copy.deepcopy(predictor.cpu()))

        predictor.train()
        idx = np.random.choice(len(X_train), size=batch_size, replace=True)
        optimizer.zero_grad()
        predictions = predictor(X_train[idx, :].to(device)).squeeze()
        unweighted_loss = loss_f(predictions, y_train[idx].to(device))
        weights = r_train[idx].to(device)
        loss = torch.mean(unweighted_loss * weights)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"val loss": val_losses[-1]})

    best_model_idx = np.argmin(val_losses)
    print("best model", best_model_idx)
    final_predictor = models[best_model_idx]
    final_predictor.eval()
    final_predictor = final_predictor.cpu()

    def estimator(X_test):
        X_test = (X_test - X_mean) / X_std
        predictions = (
            (final_predictor(torch.Tensor(X_test)).reshape(X_test.shape[0], 1))
            .detach()
            .numpy()
            .flatten()
        )

        return predictions

    metadata = {"X_mean": X_mean, "X_std": X_std, "final_predictor": final_predictor}

    return estimator, metadata
