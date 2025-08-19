import numpy as np
import pandas as pd
from data_utils import standardize


class CovariateSpline:
    def __init__(self, census_dataset):
        self.census_dataset = census_dataset
        X, r = census_dataset.get_data(normalize_weight=True)
        X, X_mean, X_std = standardize(X)
        self.X_mean = X_mean
        self.X_std = X_std
        self.d_covariate = X.shape[1]

    def get_spline(self, X):
        return (X - self.X_mean) / self.X_std


class TiltingSpline:
    def __init__(self, census_dataset, balancing_features):
        self.census_dataset = census_dataset
        # self.X_mean = X_mean
        # self.X_std = X_std
        # if y_mean is None:
        #     self.y_mean = 0.
        # else:
        #     self.y_mean = y_mean
        # if y_std is None:
        #     self.y_std = 1.
        # else:
        #     self.y_std = y_std
        self.d_tilting = 0
        if "intercept" in balancing_features:
            self.d_tilting = 1
            self.intercept = True
        else:
            self.d_tilting = 0
            self.intercept = False
        self.balancing_features = balancing_features

        if balancing_features is not None:
            remaining_features = balancing_features.copy()
            if "intercept" in remaining_features:
                remaining_features.remove("intercept")
            else:
                pass

            print(len(remaining_features))
            print(remaining_features)

            if len(remaining_features) > 0:
                self.idxs = census_dataset.get_covariates_idx(remaining_features)
                self.d_tilting += len(self.idxs)
            else:
                self.idxs = None
        else:
            self.idxs = None

    def get_spline(self, X, y):
        """
        X  is n x d
        y is m
        Returns a tensor of the shape n x d_tilting x m
        """
        # new_X = (X - self.X_mean) / self.X_std
        # new_y = (y - self.y_mean) / self.y_std
        new_X = X
        max_X = np.max(new_X, axis=0)
        min_X = np.min(new_X, axis=0)
        new_X = (new_X - min_X) / (max_X - min_X)
        basis_vectors = []
        new_y = y
        if self.idxs is not None:
            main_X = new_X[:, self.idxs]  # n x d_tilting-1
            main_X_2 = np.tile(
                main_X[:, :, None], (1, 1, new_y.shape[0])
            )  # n x d_tilting-1 x m
            y_copy = np.tile(
                y[None, None, :], (main_X.shape[0], main_X.shape[1], 1)
            )  # n x d_tilting-1  x m
            prod = main_X_2 * y_copy  # n x d_tilting-1 x m
            basis_vectors.append(prod)
        if self.intercept:
            intercept_vector = new_y[None, :] * np.ones((new_X.shape[0], 1, 1))
            basis_vectors.append(intercept_vector)
        basis = np.concatenate(basis_vectors, axis=1)  # n x d_tilting x m
        return basis

    def get_spline_indiv(self, X, y):
        new_X = X
        max_X = np.max(new_X, axis=0)
        min_X = np.min(new_X, axis=0)
        new_X = (new_X - min_X) / (max_X - min_X + 1e-8)
        basis_vectors = []
        new_y = y
        if self.idxs is not None:
            main_X = new_X[:, self.idxs]  # n x d_tilting-
            prod = main_X * new_y.reshape(len(new_y), 1)  # n x d_tilting-1
            basis_vectors.append(prod)
        if self.intercept:
            intercept_vector = new_y.reshape(len(new_y), 1) * np.ones((len(new_y), 1))
            basis_vectors.append(intercept_vector)
        basis = np.concatenate(basis_vectors, axis=1)  # n x d_tilting
        return basis


class ConstraintSpline:
    def __init__(self, regions, moment_group):
        self.d_constraints = len(regions)
        self.regions = regions
        self.moment_group = moment_group

    def get_spline(self, state_col, y):
        """
        y is a vector of length m
        state_col is a vector of length n
        Returns a tensor of shape n x d_constraints x m
        """
        basis_vectors = []

        for region in self.regions.keys():
            prod = (
                np.isin(state_col, self.regions[region]).reshape(len(state_col), 1)
                * y[None, :]
            )  # n x m
            prod = prod.reshape(prod.shape[0], 1, prod.shape[1])  # n x 1 x m
            basis_vectors.append(prod)
        res = np.concatenate(basis_vectors, axis=1)

        return res

    def get_spline_indiv(self, state_col, y):
        """
        y is a vector of length m
        state_col is a vector of length n
        Returns a tensor of shape n x d_constraints x m
        """
        basis_vectors = []

        for region in self.regions.keys():
            prod = np.isin(state_col, self.regions[region]).reshape(
                len(state_col), 1
            ) * y.reshape(len(state_col), 1)
            basis_vectors.append(prod)
        res = np.concatenate(basis_vectors, axis=1)
        return res


def get_log_partition_function_binary_outcome(X, theta, predictor, tilting_spline):
    predictions = predictor(X)
    basis_vectors_1 = tilting_spline.get_spline(X, np.ones(1))
    basis_vectors_0 = tilting_spline.get_spline(X, np.zeros(1))
    log_partition_function = np.log(
        np.exp(np.transpose(basis_vectors_1, (0, 2, 1)) @ theta)
        * predictions.reshape(-1, 1, 1)
        + np.exp(np.transpose(basis_vectors_0, (0, 2, 1)) @ theta)
        * (1 - predictions.reshape(-1, 1, 1))
    )
    return log_partition_function


def get_log_partition_function_continuous_outcome(
    X, theta, pdf_grid, y_grid, tilting_spline
):
    """
    X is n x d
    theta is d_tilting x 1
    pdf_grid is n x 1 x m
    y_grid is m
    Returns a vector of shape n x 1
    """
    eta_vectors = tilting_spline.get_spline(X, y_grid)  # n x d_tilting x m
    transposed_eta_vectors = np.transpose(eta_vectors, (0, 2, 1))[
        :, :, None, :
    ]  # n x m x 1 x d_tilting
    exponential = np.exp(np.matmul(transposed_eta_vectors, theta)).squeeze(
        -1
    )  # n x m x 1
    log_partition_function = np.log(
        np.trapezoid((exponential * pdf_grid).squeeze(-1), y_grid.flatten(), axis=1)
    )  # n,
    return log_partition_function.reshape(-1, 1)
