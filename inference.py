import numpy as np
from scipy.stats import norm
import torch
from torch.autograd.functional import jacobian
from tqdm import tqdm


class BinaryCrossFitInference:
    def __init__(
        self,
        theta,
        lamb,
        predictor,
        covariate_ratio,
        moment_group,
        constraint_m,
        tilting_spline,
        constraint_spline,
        online_survey_dataset,
        census_dataset,
    ):
        self.predictor = predictor
        self.covariate_ratio = covariate_ratio
        self.moment_group = moment_group
        self.constraint_m = torch.tensor(constraint_m, dtype=torch.float64).flatten()
        self.tilting_spline = tilting_spline
        self.constraint_spline = constraint_spline
        self.online_survey_dataset = online_survey_dataset
        self.census_dataset = census_dataset

        nu = torch.cat(
            [
                torch.tensor(theta.flatten()),
                torch.tensor(lamb.flatten()),
            ]
        )
        self.nu = torch.nn.Parameter(nu, requires_grad=True)
        self.theta_idx = np.arange(len(theta))
        self.lamb_idx = np.arange(len(theta), len(theta) + len(lamb))

    def _get_units(self, X, state_col):
        eta1s = torch.tensor(
            self.tilting_spline.get_spline_indiv(X, np.ones(X.shape[0])),
            dtype=torch.float64,
        )  # n x d_tilting
        eta0s = torch.tensor(
            self.tilting_spline.get_spline_indiv(X, np.zeros(X.shape[0])),
            dtype=torch.float64,
        )
        gamma1s = torch.tensor(
            self.constraint_spline.get_spline_indiv(
                state_col, np.ones(state_col.shape[0])
            ),
            dtype=torch.float64,
        )
        gamma0s = torch.tensor(
            self.constraint_spline.get_spline_indiv(
                state_col, np.zeros(state_col.shape[0])
            ),
            dtype=torch.float64,
        )

        return (eta1s, eta0s), (gamma1s, gamma0s)

    def _get_outer_products(self, X, state_col):
        (eta1s, eta0s), (gamma1s, gamma0s) = self._get_units(X, state_col)

        outer_prod_eta1s = torch.einsum(
            "bi,bj->bij", eta1s, eta1s
        )  # n x d_tilting x d_tilting
        outer_prod_eta1_gamma1 = torch.einsum(
            "bi,bj->bij", eta1s, gamma1s
        )  # n x d_tilting x d_constraint
        outer_prod_eta0s = torch.einsum(
            "bi,bj->bij", eta0s, eta0s
        )  # n x d_tilting x d_tilting
        outer_prod_eta0_gamma0 = torch.einsum("bi,bj->bij", eta0s, gamma0s)

        return (
            outer_prod_eta1s,
            outer_prod_eta1_gamma1,
            outer_prod_eta0s,
            outer_prod_eta0_gamma0,
        )

    def _get_probs(self, X, state_col):
        (eta1s, eta0s), _ = self._get_units(X, state_col)
        theta = self.nu[self.theta_idx]

        eta1_theta = torch.einsum("ij,j->i", eta1s, theta)
        eta0_theta = torch.einsum("ij,j->i", eta0s, theta)
        mus = torch.tensor(self.predictor(X), dtype=torch.float64).flatten()

        prob1 = (mus * torch.exp(eta1_theta)) / (
            mus * torch.exp(eta1_theta) + (1 - mus) * torch.exp(eta0_theta)
        )
        prob0 = 1 - prob1
        return prob1, prob0

    def _get_psi_nonorthogonal(self):
        theta = self.nu[self.theta_idx]
        lamb = self.nu[self.lamb_idx]

        X_census, r, census_state_col = self.census_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )
        r = torch.tensor(r, dtype=torch.float64).flatten()

        probs1, probs0 = self._get_probs(X_census, census_state_col)
        (eta1s, eta0s), (gamma1s, gamma0s) = self._get_units(X_census, census_state_col)
        (
            outer_prod_eta1s,
            outer_prod_eta1_gamma1,
            outer_prod_eta0s,
            outer_prod_eta0_gamma0,
        ) = self._get_outer_products(X_census, census_state_col)

        # Non orthogonal terms
        expectation_eta_outerprod = torch.einsum(
            "b,bij->bij", probs1, outer_prod_eta1s
        ) + torch.einsum("b,bij->bij", probs0, outer_prod_eta0s)
        expectation_eta = torch.einsum("b,bi->bi", probs1, eta1s) + torch.einsum(
            "b,bi->bi", probs0, eta0s
        )
        outerprod_expectation_eta = torch.einsum(
            "bi,bj->bij", expectation_eta, expectation_eta
        )
        covariance_eta = expectation_eta_outerprod - outerprod_expectation_eta
        final_covariance_eta = torch.einsum("b,bij->ij", r, covariance_eta)
        first_term = torch.einsum("ij,j->j", final_covariance_eta, theta)

        expectation_eta_gamma = torch.einsum(
            "b,bij->bij", probs1, outer_prod_eta1_gamma1
        ) + torch.einsum("b,bij->bij", probs0, outer_prod_eta0_gamma0)
        expectation_gamma = torch.einsum("b,bi->bi", probs1, gamma1s) + torch.einsum(
            "b,bi->bi", probs0, gamma0s
        )
        outerprod_expectation_eta_gamma = torch.einsum(
            "bi,bj->bij", expectation_eta, expectation_gamma
        )
        covariance_eta_gamma = expectation_eta_gamma - outerprod_expectation_eta_gamma
        final_covariance_eta_gamma = torch.einsum("b,bij->ij", r, covariance_eta_gamma)
        second_term = torch.einsum("ij,j->j", final_covariance_eta_gamma, lamb)

        d_l = first_term + second_term
        m = torch.einsum("bi,b->i", expectation_gamma, r) - self.constraint_m

        return d_l, m

    def _get_psi_orthogonal_adjustments(self):
        theta = self.nu[self.theta_idx]
        lamb = self.nu[self.lamb_idx]
        X_online, y, _, online_state_col = self.online_survey_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )
        y = torch.tensor(y, dtype=torch.float64).flatten()
        r_online = torch.tensor(
            self.covariate_ratio(X_online), dtype=torch.float64
        ).flatten()

        probs1, probs0 = self._get_probs(X_online, online_state_col)
        mus = torch.tensor(self.predictor(X_online), dtype=torch.float64).flatten()
        weights1 = probs1 / mus
        weights0 = probs0 / (1 - mus)
        (etas1, etas0), (gammas1, gammas0) = self._get_units(X_online, online_state_col)

        expectation_eta = torch.einsum("b,bi->bi", probs1, etas1) + torch.einsum(
            "b,bi->bi", probs0, etas0
        )
        expectation_gamma = torch.einsum("b,bi->bi", probs1, gammas1) + torch.einsum(
            "b,bi->bi", probs0, gammas0
        )

        delta_eta1 = etas1 - expectation_eta
        delta_gamma1 = gammas1 - expectation_gamma
        delta_eta0 = etas0 - expectation_eta
        delta_gamma0 = gammas0 - expectation_gamma

        delta_eta1_outer_prod = torch.einsum("bi,bj->bij", delta_eta1, delta_eta1)
        delta_eta1_gamma1_outer_prod = torch.einsum(
            "bi,bj->bij", delta_eta1, delta_gamma1
        )
        delta_eta0_outer_prod = torch.einsum("bi,bj->bij", delta_eta0, delta_eta0)
        delta_eta0_gamma0_outer_prod = torch.einsum(
            "bi,bj->bij", delta_eta0, delta_gamma0
        )

        rho1 = torch.einsum("bij,j->bi", delta_eta1_outer_prod, theta) + torch.einsum(
            "bij,j->bi", delta_eta1_gamma1_outer_prod, lamb
        )
        rho0 = torch.einsum("bij,j->bi", delta_eta0_outer_prod, theta) + torch.einsum(
            "bij,j->bi", delta_eta0_gamma0_outer_prod, lamb
        )

        adjustment_d_l = torch.einsum(
            "b,bi->bi",
            r_online.flatten() * weights1 * weights0 * (y.flatten() - mus),
            (rho1 - rho0),
        )

        adjustment_m = torch.einsum(
            "b,bi->bi",
            r_online.flatten() * weights1 * weights0 * (y.flatten() - mus),
            (gammas1 - gammas0),
        )
        return adjustment_d_l, adjustment_m

    def _get_psi(self):

        d_l, m = self._get_psi_nonorthogonal()
        adjustment_d_l, adjustment_m = self._get_psi_orthogonal_adjustments()
        final_d_l = d_l + adjustment_d_l
        final_m = m + adjustment_m

        psi = torch.cat([final_d_l, final_m], dim=1)
        return psi

    def _get_jacobian(self):

        def helper_psi(nu):
            self.nu = nu
            psi = self._get_psi().mean(dim=0)
            return psi

        jacobian_psi = jacobian(helper_psi, self.nu)
        return jacobian_psi

    def get_asymptotic_variance(self):
        psi = self._get_psi()
        psi_mean = psi.mean(dim=0)
        n = len(self.online_survey_dataset)
        sigma_mean = torch.einsum("bi,bj->bij", psi, psi).mean(dim=0)
        jacobian_psi = self._get_jacobian()
        asymptotic_variance = (
            torch.linalg.inv(jacobian_psi)
            @ sigma_mean
            @ torch.linalg.inv(jacobian_psi).T
        )
        one_step_nu = self.nu - torch.linalg.inv(jacobian_psi) @ psi_mean
        return one_step_nu, asymptotic_variance, n


def get_ci(
    census_dataset,
    predictor,
    moment_group,
    tilting_spline,
    one_step_nu,
    theta_idx,
    asymptotic_variance,
    n,
    indexes,
    beta=0.05,
):
    X_census, r, _ = census_dataset.get_data(
        normalize_weight=True, return_state_col=moment_group
    )
    eta1s = torch.tensor(
        tilting_spline.get_spline_indiv(X_census, np.ones(X_census.shape[0])),
        dtype=torch.float64,
    )  # n x d_tilting
    eta0s = torch.tensor(
        tilting_spline.get_spline_indiv(X_census, np.zeros(X_census.shape[0])),
        dtype=torch.float64,
    )
    eta1s_idxs = eta1s[indexes]
    eta0s_idxs = eta0s[indexes]
    mus_idxs = predictor(X_census[indexes])
    r_idxs = torch.tensor(r[indexes] / r[indexes].sum(), dtype=torch.float64).reshape(
        -1, 1
    )

    def helper_state_estimate(nu):
        theta = nu[theta_idx]
        pred = torch.tensor(mus_idxs).reshape(-1, 1)
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
    state_estimate_upper = state_estimate + val * torch.sqrt(state_estimate_variance)
    state_estimate_lower = state_estimate - val * torch.sqrt(state_estimate_variance)

    return state_estimate, (state_estimate_lower, state_estimate_upper)
