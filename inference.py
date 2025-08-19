import numpy as np
from scipy.stats import norm
import torch
from torch.autograd.functional import jacobian


class BinaryStatisticalInference:
    def __init__(
        self,
        theta,
        predictor_metadata,
        lamb,
        mu_eta,
        moment_group,
        constraint_m,
        tilting_spline,
        constraint_spline,
        online_survey_dataset,
        census_dataset,
    ):
        self.theta = theta
        self.predictor_metadata = predictor_metadata
        self.moment_group = moment_group
        self.constraint_m = torch.tensor(constraint_m, dtype=torch.float64).flatten()
        self.tilting_spline = tilting_spline
        self.constraint_spline = constraint_spline
        self.online_survey_dataset = online_survey_dataset
        self.census_dataset = census_dataset
        model = self.predictor_metadata["final_predictor"]

        self.f = torch.nn.Sequential(*list(model.children())[:-2])
        self.X_mean = self.predictor_metadata["X_mean"]
        self.X_std = self.predictor_metadata["X_std"]
        gamma = list(model.children())[-2].weight.data.flatten()
        self.gamma = gamma.double()

        nu = torch.cat(
            [
                torch.tensor(self.theta.flatten()),
                self.gamma,
                torch.tensor(lamb.flatten()),
            ]
        )
        self.nu = torch.nn.Parameter(nu, requires_grad=True)

        self.theta_idx = np.arange(len(self.theta))
        self.gamma_idx = np.arange(len(self.theta), len(self.theta) + len(gamma))
        self.lamb_idx = np.arange(
            len(self.theta) + len(gamma), len(self.theta) + len(gamma) + len(lamb)
        )
        self.mu_eta_idx = np.arange(
            len(self.theta) + len(gamma) + len(lamb),
            len(self.theta) + len(gamma) + len(lamb) + len(mu_eta),
        )

    def _get_units(self, X, state_col):

        eta1s = torch.tensor(
            self.tilting_spline.get_spline_indiv(X, np.ones(X.shape[0])),
            dtype=torch.float64,
        )  # n x d_tilting
        eta0s = torch.tensor(
            self.tilting_spline.get_spline_indiv(X, np.zeros(X.shape[0])),
            dtype=torch.float64,
        )
        psi1s = torch.tensor(
            self.constraint_spline.get_spline_indiv(
                state_col, np.ones(state_col.shape[0])
            ),
            dtype=torch.float64,
        )
        psi0s = torch.tensor(
            self.constraint_spline.get_spline_indiv(
                state_col, np.zeros(state_col.shape[0])
            ),
            dtype=torch.float64,
        )
        fs = self.f(torch.tensor((X - self.X_mean) / self.X_std)).double()
        return (eta1s, eta0s), (psi1s, psi0s), fs

    def _get_outer_products(self, X, state_col):
        (eta1s, eta0s), (psi1s, psi0s), _ = self._get_units(X, state_col)

        outer_prod_eta1s = torch.einsum(
            "bi,bj->bij", eta1s, eta1s
        )  # n x d_tilting x d_tilting
        outer_prod_eta1_psi1 = torch.einsum(
            "bi,bj->bij", eta1s, psi1s
        )  # n x d_tilting x d_constraint
        outer_prod_eta0s = torch.einsum(
            "bi,bj->bij", eta0s, eta0s
        )  # n x d_tilting x d_tilting
        outer_prod_eta0_psi0 = torch.einsum("bi,bj->bij", eta0s, psi0s)

        return (
            outer_prod_eta1s,
            outer_prod_eta1_psi1,
            outer_prod_eta0s,
            outer_prod_eta0_psi0,
        )

    def get_z_mean_helper(self):
        X_census, r, census_state_col = self.census_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )
        X_online, y, _, online_state_col = self.online_survey_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )

        y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
        (eta1s, eta0s), (psi1s, psi0s), fs = self._get_units(X_census, census_state_col)
        (
            outer_prod_eta1s,
            outer_prod_eta1_psi1,
            outer_prod_eta0s,
            outer_prod_eta0_psi0,
        ) = self._get_outer_products(X_census, census_state_col)
        r = torch.tensor(r, dtype=torch.float64).reshape(-1, 1)

        def helper_z_mean(nu):
            theta = nu[self.theta_idx]
            gamma = nu[self.gamma_idx]
            lamb = nu[self.lamb_idx]

            eta1_theta = torch.einsum("ij,j->i", eta1s, theta).reshape(-1, 1)
            eta0_theta = torch.einsum("ij,j->i", eta0s, theta).reshape(-1, 1)
            census_pred = torch.sigmoid(torch.einsum("ij,j->i", fs, gamma)).reshape(
                -1, 1
            )
            weights = (torch.exp(eta1_theta) * census_pred) / (
                torch.exp(eta1_theta) * census_pred
                + torch.exp(eta0_theta) * (1 - census_pred)
            )

            expectation_eta = eta1s * weights + eta0s * (1 - weights)
            expectation_psi = psi1s * weights + psi0s * (1 - weights)

            batch_size = X_online.shape[0] // 100
            first_index_mean = 0.0
            total_samples = 0.0
            for i in range(0, y.shape[0], batch_size):
                print("batch", i)
                X_online_batch = X_online[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                _, _, online_fs_batch = self._get_units(
                    X_online_batch, online_state_col
                )
                online_pred_batch = torch.sigmoid(
                    torch.einsum("ij,j->i", online_fs_batch, gamma)
                ).reshape(-1, 1)
                first_index_batch = (-y_batch + online_pred_batch) * online_fs_batch
                total_samples += first_index_batch.shape[0]
                first_index_mean = (
                    first_index_mean * (total_samples - first_index_batch.shape[0])
                    + first_index_batch.sum(dim=0)
                ) / total_samples

            first_index = first_index_mean[None, :]
            second_index = (r * (psi1s * weights + psi0s * (1 - weights))).sum(
                dim=0
            ) - self.constraint_m
            second_index = second_index[None, :]

            mat_eta1_outer_prod_theta = torch.einsum(
                "bij,j->bi", outer_prod_eta1s, theta
            )
            mat_eta1_psi1_outer_prod_lamb = torch.einsum(
                "bij,j->bi", outer_prod_eta1_psi1, lamb
            )
            mat_eta0_outer_prod_theta = torch.einsum(
                "bij,j->bi", outer_prod_eta0s, theta
            )
            mat_eta0_psi0_outer_prod_lamb = torch.einsum(
                "bij,j->bi", outer_prod_eta0_psi0, lamb
            )

            mat_mu_eta_outer_prod = torch.einsum(
                "bi,bj->bij", expectation_eta, expectation_eta
            )

            mat_mu_eta_theta = torch.einsum(
                "bij, j -> bi", mat_mu_eta_outer_prod, theta
            )

            mat_mu_eta_psi_outer_prod = torch.einsum(
                "bi,bj->bij", expectation_eta, expectation_psi
            )

            mat_mu_eta_psi_outer_prod_lamb = torch.einsum(
                "bij, j -> bi", mat_mu_eta_psi_outer_prod, lamb
            )

            third_index_first_part = (
                mat_eta1_outer_prod_theta + mat_eta1_psi1_outer_prod_lamb
            ) * weights + (
                mat_eta0_outer_prod_theta + mat_eta0_psi0_outer_prod_lamb
            ) * (
                1 - weights
            )
            third_index_second_part = mat_mu_eta_theta + mat_mu_eta_psi_outer_prod_lamb
            third_index = torch.sum(
                r * (third_index_first_part - third_index_second_part), dim=0
            )
            third_index = third_index[None, :]

            z = torch.cat([first_index, second_index, third_index], dim=1).squeeze(0)
            return z

        return helper_z_mean

    def get_z_outerproduct_mean(self):

        X_census, r, census_state_col = self.census_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )
        X_online, y, _, online_state_col = self.online_survey_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )

        y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
        (eta1s, eta0s), (psi1s, psi0s), fs = self._get_units(X_census, census_state_col)
        (
            outer_prod_eta1s,
            outer_prod_eta1_psi1,
            outer_prod_eta0s,
            outer_prod_eta0_psi0,
        ) = self._get_outer_products(X_census, census_state_col)
        r = torch.tensor(r, dtype=torch.float64).reshape(-1, 1)

        theta = self.nu[self.theta_idx]
        gamma = self.nu[self.gamma_idx]
        lamb = self.nu[self.lamb_idx]

        eta1_theta = torch.einsum("ij,j->i", eta1s, theta).reshape(-1, 1)
        eta0_theta = torch.einsum("ij,j->i", eta0s, theta).reshape(-1, 1)
        census_pred = torch.sigmoid(torch.einsum("ij,j->i", fs, gamma)).reshape(-1, 1)
        weights = (torch.exp(eta1_theta) * census_pred) / (
            torch.exp(eta1_theta) * census_pred
            + torch.exp(eta0_theta) * (1 - census_pred)
        )

        expectation_eta = eta1s * weights + eta0s * (1 - weights)
        expectation_psi = psi1s * weights + psi0s * (1 - weights)

        second_index = (r * (psi1s * weights + psi0s * (1 - weights))).sum(
            dim=0
        ) - self.constraint_m
        second_index = second_index[None, :]

        mat_eta1_outer_prod_theta = torch.einsum("bij,j->bi", outer_prod_eta1s, theta)
        mat_eta1_psi1_outer_prod_lamb = torch.einsum(
            "bij,j->bi", outer_prod_eta1_psi1, lamb
        )
        mat_eta0_outer_prod_theta = torch.einsum("bij,j->bi", outer_prod_eta0s, theta)
        mat_eta0_psi0_outer_prod_lamb = torch.einsum(
            "bij,j->bi", outer_prod_eta0_psi0, lamb
        )

        mat_mu_eta_outer_prod = torch.einsum(
            "bi,bj->bij", expectation_eta, expectation_eta
        )

        mat_mu_eta_theta = torch.einsum("bij, j -> bi", mat_mu_eta_outer_prod, theta)

        mat_mu_eta_psi_outer_prod = torch.einsum(
            "bi,bj->bij", expectation_eta, expectation_psi
        )

        mat_mu_eta_psi_outer_prod_lamb = torch.einsum(
            "bij, j -> bi", mat_mu_eta_psi_outer_prod, lamb
        )

        third_index_first_part = (
            mat_eta1_outer_prod_theta + mat_eta1_psi1_outer_prod_lamb
        ) * weights + (mat_eta0_outer_prod_theta + mat_eta0_psi0_outer_prod_lamb) * (
            1 - weights
        )
        third_index_second_part = mat_mu_eta_theta + mat_mu_eta_psi_outer_prod_lamb
        third_index = torch.sum(
            r * (third_index_first_part - third_index_second_part), dim=0
        )
        third_index = third_index[None, :]

        batch_size = X_online.shape[0] // 100
        total_samples = 0.0
        z_outer_prod_mean = 0.0
        for i in range(0, y.shape[0], batch_size):
            X_online_batch = X_online[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            _, _, online_fs_batch = self._get_units(X_online_batch, online_state_col)
            online_pred_batch = torch.sigmoid(
                torch.einsum("ij,j->i", online_fs_batch, gamma)
            ).reshape(-1, 1)
            first_index_batch = (-y_batch + online_pred_batch) * online_fs_batch
            second_index_batch = torch.tile(
                second_index, (first_index_batch.shape[0], 1)
            )
            third_index_batch = torch.tile(third_index, (first_index_batch.shape[0], 1))
            z_batch = torch.cat(
                [first_index_batch, second_index_batch, third_index_batch], dim=1
            )
            outer_prod = torch.einsum("bi,bj->bij", z_batch, z_batch)  # n x nu_d x nu_d
            z_outer_prod_mean = (
                z_outer_prod_mean * total_samples + outer_prod.sum(dim=0)
            ) / (total_samples + z_batch.shape[0])
            total_samples += z_batch.shape[0]
        return z_outer_prod_mean

    def get_jacobian(self):
        helper_z_mean = self.get_z_mean_helper()
        jacobian_mean_z = jacobian(helper_z_mean, self.nu)
        return jacobian_mean_z

    def get_asymptotic_variance(self):
        helper_z_mean = self.get_z_mean_helper()
        z_mean = helper_z_mean(self.nu)
        n = len(self.online_survey_dataset)
        sigma = self.get_z_outerproduct_mean()  # nu_d x nu_d
        jac = self.get_jacobian()
        asymptotic_variance = torch.linalg.inv(jac) @ sigma @ torch.linalg.inv(jac).T
        one_step_nu = self.nu - torch.linalg.inv(jac) @ z_mean

        return one_step_nu, asymptotic_variance, n

    def get_ci(self, one_step_nu, asymptotic_variance, n, indexes, beta=0.05):
        X_census, r, census_state_col = self.census_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )
        (eta1s, eta0s), _, fs = self._get_units(X_census, census_state_col)
        eta1s_idxs = eta1s[indexes]
        eta0s_idxs = eta0s[indexes]
        fs_idxs = fs[indexes]
        r_idxs = torch.tensor(
            r[indexes] / r[indexes].sum(), dtype=torch.float64
        ).reshape(-1, 1)

        def helper_state_estimate(nu):
            gamma = nu[self.gamma_idx]
            theta = nu[self.theta_idx]
            pred = torch.sigmoid(torch.einsum("ij,j->i", fs_idxs, gamma)).reshape(-1, 1)
            eta1_theta = torch.einsum("ij,j->i", eta1s_idxs, theta).reshape(-1, 1)
            eta0_theta = torch.einsum("ij,j->i", eta0s_idxs, theta).reshape(-1, 1)
            state_estimate = torch.sum(
                r_idxs
                * pred
                * torch.exp(eta1_theta)
                / (pred * torch.exp(eta1_theta) + (1 - pred) * torch.exp(eta0_theta))
            )
            return state_estimate

        jacobian_state_estimate = jacobian(
            helper_state_estimate, one_step_nu, create_graph=True
        ).reshape(-1, 1)

        state_estimate_variance = (
            jacobian_state_estimate.T @ asymptotic_variance @ jacobian_state_estimate
        )
        state_estimate_variance /= n
        val = norm.ppf(1 - beta / 2)
        state_estimate = helper_state_estimate(one_step_nu)
        state_estimate_upper = state_estimate + val * torch.sqrt(
            state_estimate_variance
        )
        state_estimate_lower = state_estimate - val * torch.sqrt(
            state_estimate_variance
        )
        return state_estimate, (state_estimate_lower, state_estimate_upper)
