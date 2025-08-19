import numpy as np
import pandas as pd


def standardize(z):
    """
    Standardize a numpy array.

    :param z: The input array to be standardized.
    :type z: numpy.ndarray
    :return: A tuple containing the standardized array, mean array, and standard deviation array.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)

    if isinstance(z_std, np.ndarray):
        z_std[np.where(z_std == 0.0)[0]] = 1.0
    elif z_std == 0.0:
        z_std = 1.0

    data = (z - z_mean) / z_std
    return data, z_mean, z_std


class Dataset:
    def __init__(
        self,
        df,
        covs,
        indicator,
        weight,
        binary_indicator,
    ):
        self.df = df
        self.covs = sorted(covs)
        self.indicator = indicator
        self.weight = weight
        self.binary_indicator = binary_indicator

    def get_covariates(self, covariates_list):
        covs_list = sorted(covariates_list)
        return self.df[sorted(covs_list)].values.astype(np.float32)

    def get_covariates_idx(self, covariates_list):
        idxs = []
        for cov2 in covariates_list:
            cov2_idxs = []
            for i, cov1 in enumerate(self.covs):
                if cov1.startswith(cov2):
                    cov2_idxs.append(i)
            if len(cov2_idxs) > 1:
                cov2_idxs = cov2_idxs[:-1]
            idxs += cov2_idxs
        return idxs

    def get_data(self, normalize_weight=False, return_state_col=None):
        if self.covs is None:
            X = self.df[self.covs].values
        else:
            X = self.df[sorted(self.covs)].values.astype(np.float32)

        r = self.df[self.weight].values.astype(np.float32)
        if normalize_weight:
            r = r / r.sum()

        if self.indicator is None and return_state_col is None:
            return X, r
        elif self.indicator is None and return_state_col is not None:
            return X, r, self.df[return_state_col].values
        elif self.indicator is not None and return_state_col is None:
            y = self.df[self.indicator].values.astype(np.float32)
            return X, y, r
        elif self.indicator is not None and return_state_col is not None:
            y = self.df[self.indicator].values.astype(np.float32)
            return X, y, r, self.df[return_state_col].values

    def sample_minibatch(
        self, n_samples, normalize_weight=False, return_state_col=None
    ):
        weights = self.df[self.weight].values
        weights = weights / weights.sum()
        idx = np.random.choice(len(self.df), size=n_samples, replace=True, p=weights)

        if (
            self.covs is not None
            and self.indicator is not None
            and return_state_col is not None
        ):
            X, y, r = self.get_data(normalize_weight)
            return X[idx], y[idx], r[idx], self.df[return_state_col].values[idx]
        elif (
            self.covs is not None
            and self.indicator is None
            and return_state_col is not None
        ):
            X, r = self.get_data(normalize_weight)
            return X[idx], r[idx], self.df[return_state_col].values[idx]
        elif (
            self.covs is not None
            and self.indicator is not None
            and not return_state_col
        ):
            X, y, r = self.get_data(normalize_weight)
            return X[idx], y[idx], r[idx]
        else:
            X, r = self.get_data(normalize_weight)
            return X[idx], r[idx]

    def __len__(self):
        return len(self.df)


def split(dataset, frac=0.6, seed=0):
    """
    Split a dataset into two parts.

    :param dataset: The input dataset.
    :type dataset: Dataset
    :param frac: The fraction of the dataset to be used for training.
    :type frac: float
    """
    n = len(dataset)
    np.random.seed(seed)
    idx = np.random.permutation(n)
    train_idx = idx[: int(frac * n)]
    test_idx = idx[int(frac * n) :]

    train_dataset = Dataset(
        dataset.df.iloc[train_idx].copy(deep=True).reset_index(drop=True),
        indicator=dataset.indicator,
        covs=dataset.covs,
        weight=dataset.weight,
        binary_indicator=dataset.binary_indicator,
    )

    test_dataset = Dataset(
        dataset.df.iloc[test_idx].copy(deep=True).reset_index(drop=True),
        indicator=dataset.indicator,
        covs=dataset.covs,
        weight=dataset.weight,
        binary_indicator=dataset.binary_indicator,
    )

    return train_dataset, test_dataset
