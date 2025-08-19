import pandas as pd
import numpy as np
import cvxpy as cp
from splines import (
    get_log_partition_function_continuous_outcome,
    get_log_partition_function_binary_outcome,
    TiltingSpline,
    ConstraintSpline,
)
from inference import BinaryStatisticalInference


def get_state_estimates(indicator, predictor, census_dataset, evaluation_group):
    X, r = census_dataset.get_data()
    predictions = predictor(X)

    group_to_indexes = (
        census_dataset.df.groupby(evaluation_group)
        .apply(lambda x: x.index.tolist())
        .to_dict()
    )
    estimates = []
    for group, indexes in group_to_indexes.items():
        res = {}
        for i, name in enumerate(evaluation_group):
            if type(group) is str or type(group) is float:
                res[name] = group
            elif type(group) is tuple:
                res[name] = group[i]
        res.update(
            {
                indicator: (
                    (predictions[indexes] * r[indexes]).sum() / r[indexes].sum()
                ).item(),
            }
        )
        estimates.append(res)
    df = pd.DataFrame(estimates)
    return df


class EstimatorsContinuousOutcome:

    def __init__(
        self,
        census_dataset,
        cond_density_estimator,
        group_name,
        tilting_spline,
        constraint_spline,
    ):
        self.tilting_spline = tilting_spline
        self.constraint_spline = constraint_spline
        self.group_name = group_name
        self.cond_density_estimator = cond_density_estimator

        X, r = census_dataset.get_data()
        density = self.cond_density_estimator(X[[0]])[0]
        self.y_grid = np.linspace(
            density.outcome_range[0].item(), density.outcome_range[1].item(), 100
        )
        self.census_dataset = census_dataset
        self.precomputation()

    def precomputation(self):
        X, r, state_col = self.census_dataset.get_data(return_state_col=self.group_name)
        self.weight = r.reshape(-1, 1) / r.sum()
        densities = self.cond_density_estimator(X)
        self.pdf_grid = np.array(
            [density.pdf(self.y_grid) for density in densities]
        ).reshape(
            X.shape[0], len(self.y_grid), 1
        )  # n x m x 1
        self.eta_vectors = self.tilting_spline.get_spline(
            X, self.y_grid
        )  # n x d_tilting x m
        self.psi_vectors = self.constraint_spline.get_spline(
            state_col, self.y_grid
        )  # n x d_constraints x m

    def set_theta(self, theta):
        self.theta = theta
        X, _ = self.census_dataset.get_data(return_state_col=None)

        self.log_partition_function = get_log_partition_function_continuous_outcome(
            X, self.theta, self.pdf_grid, self.y_grid, self.tilting_spline
        )
        transposed_eta_vectors = np.transpose(self.eta_vectors, (0, 2, 1))[
            :, :, None, :
        ]  # n x m x 1 x d_tilting
        exponential = np.exp(
            transposed_eta_vectors @ theta.flatten()
            - self.log_partition_function[:, :, None]
        )
        self.probs = self.pdf_grid * exponential  # n x m x 1

    def get_rho(self):
        # eta = n x d_tilting x m x 1
        # probs = n x 1 x m x 1
        probs_transpose = self.probs[:, None, :, :]  # n x 1 x m x 1
        eta_new = self.eta_vectors[:, :, :, None]  # n x d_tilting x m x 1
        probs_repeat_eta = np.tile(
            probs_transpose, (1, eta_new.shape[1], 1, 1)
        )  # n x d_constraints x m x 1
        eta = np.trapezoid(
            eta_new * probs_repeat_eta, self.y_grid.flatten(), axis=2
        )  # n x d_tilting x 1
        res = eta * self.weight[:, :, None]
        res = res.sum(axis=0)  # d_tilting x 1
        return res

    def get_m(self):
        probs_transpose = self.probs[:, None, :, :]  # n x 1 x m x 1
        psi_new = self.psi_vectors[:, :, :, None]  # n x d_constraints x m x 1
        probs_repeat_psi = np.tile(
            probs_transpose, (1, psi_new.shape[1], 1, 1)
        )  # n x d_constraints x m x 1
        psi = np.trapezoid(
            psi_new * probs_repeat_psi, self.y_grid.flatten(), axis=2
        )  # n x d_constraints x 1
        res = psi * self.weight[:, :, None]
        res = res.sum(axis=0)  # d_tilting x 1
        return res

    def get_B_matrix(self):
        psi_transposed = np.transpose(self.psi_vectors, (0, 2, 1))[
            :, :, :, None
        ]  # n x m x d_constraints x 1
        eta_transposed = np.transpose(self.eta_vectors, (0, 2, 1))[
            :, :, None, :
        ]  # n x m x 1 x d_tilting
        term1 = np.matmul(
            psi_transposed,
            eta_transposed,
        )  # n x m x d_constraints x d_tilting
        probs_repeat_term1 = np.tile(
            self.probs[:, :, :, None], (1, 1, term1.shape[-2], term1.shape[-1])
        )  # n x m x d_constraints x d_tilting
        term1_w_probs = np.trapezoid(
            term1 * probs_repeat_term1, self.y_grid.flatten(), axis=1
        )  # n x m x d_constraints x d_tilting
        term1_res = (
            term1_w_probs * self.weight[:, :, None]
        )  # n x d_constraints x d_tilting

        probs_transpose = self.probs[:, None, :, :]  # n x 1 x m x 1
        psi_new = self.psi_vectors[:, :, :, None]  # n x d_constraints x m x 1
        probs_repeat_psi = np.tile(
            probs_transpose, (1, psi_new.shape[1], 1, 1)
        )  # n x d_constraints x m x 1
        psi = np.trapezoid(
            psi_new * probs_repeat_psi, self.y_grid.flatten(), axis=2
        )  # n x d_constraints x 1

        eta_new = self.eta_vectors[:, :, :, None]  # n x d_tilting x m x 1
        probs_repeat_eta = np.tile(
            self.probs[:, None, :, :], (1, eta_new.shape[1], 1, 1)
        )  # n x d_tilting x m x 1
        eta = np.trapezoid(
            eta_new * probs_repeat_eta, self.y_grid.flatten(), axis=2
        )  # n x d_tilting x 1

        term2_res = (
            np.matmul(psi, np.transpose(eta, (0, 2, 1))) * self.weight[:, :, None]
        )  # n x d_constraints x d_tilting
        B = np.sum(term1_res - term2_res, axis=0)
        return np.vstack([B, -B])  # 2 * d_constraints x d_tilting

    def get_I_matrix(self):
        eta_new = np.transpose(self.eta_vectors, (0, 2, 1))[
            :, :, :, None
        ]  # n x m x d_tilting x 1
        eta_transposed = np.transpose(self.eta_vectors, (0, 2, 1))[
            :, :, None, :
        ]  # n x m x 1 x d_tilting
        term1 = np.matmul(eta_new, eta_transposed)  # n x m x d_tilting x d_tilting
        probs_repeat_term1 = np.tile(
            self.probs[:, :, :, None], (1, 1, term1.shape[-2], term1.shape[-1])
        )  # n x m x d_tilting x d_tilting
        term1_w_probs = np.trapezoid(
            term1 * probs_repeat_term1, self.y_grid.flatten(), axis=1
        )  # n x m x d_tilting x d_tilting
        term1_res = term1_w_probs * self.weight[:, :, None]  # n x d_tilting x d_tilting

        probs_transpose = self.probs[:, None, :, :]  # n x 1 x m x 1
        eta_new = self.eta_vectors[:, :, :, None]  # n x d_tilting x m x1
        probs_repeat_eta = np.tile(
            probs_transpose, (1, eta_new.shape[1], 1, 1)
        )  # n x d_tilting x m x 1
        eta = np.trapezoid(
            eta_new * probs_repeat_eta, self.y_grid.flatten(), axis=2
        )  # n x d_tilting x 1
        term2_res = (
            np.matmul(eta, np.transpose(eta, (0, 2, 1))) * self.weight[:, :, None]
        )  # n x d_tilting x d_tilting
        return np.sum(term1_res - term2_res, axis=0)

    def get_D_tensor(self):
        # n x d_tilting x m x 1
        d_tilting = self.eta_vectors.shape[1]
        probs_repeat = np.tile(self.probs[:, None, :, :], (1, d_tilting, 1, 1))
        eta_dot = (
            probs_repeat * self.eta_vectors[:, :, :, None]
        )  # n x d_tilting x m x 1
        expectation_eta = np.trapezoid(
            eta_dot, self.y_grid.flatten(), axis=2
        )  # n x d_tilting x 1
        z = self.eta_vectors - expectation_eta  # n x d_tilting x m
        tensor_partial = np.einsum(
            "bik,bjk->bijk", z, z
        )  # n x d_tilting x d_tilting x m
        tensor = np.einsum("bijk,blk->bijlk", tensor_partial, z)[
            :, :, :, :, :, None
        ]  # n x d_tilting x d_tilting x d_tilting x m x 1
        probs_repeat_2 = np.tile(
            self.probs[:, None, None, None, :, :],
            (1, d_tilting, d_tilting, d_tilting, 1, 1),
        )  # n x d_tilting x d_tilting x d_tilting x m x1
        tensor_w_probs = (
            tensor * probs_repeat_2
        )  # n x d_tilting x d_tilting x d_tilting x m x 1
        outer_expectation = np.trapezoid(
            tensor_w_probs, self.y_grid.flatten(), axis=4
        ).squeeze(
            -1
        )  # n x d_tilting x d_tilting x d_tilting
        D_cond_w_weight = (
            outer_expectation * self.weight[:, :, None, None]
        )  # n x d_tilting x d_tilting x d_tilting
        D = D_cond_w_weight.sum(axis=0)  # d_tilting x d_tilting x d_tilting
        return D
        # exp_tensor =

    def get_C_tensor(self):
        """
        Returns a tensor of shape d_constraints x d_tilting x d_tilting
        """
        first_part = np.einsum(
            "bkl,bjl->bkjl", self.psi_vectors, self.eta_vectors
        )  # n x d_constraints x d_tilting x m
        d_tilting = self.eta_vectors.shape[1]
        d_constraints = self.psi_vectors.shape[1]
        probs_repeat = np.tile(self.probs[:, None, :, :], (1, d_tilting, 1, 1))
        eta_dot = (
            probs_repeat * self.eta_vectors[:, :, :, None]
        )  # n x d_tilting x m x 1
        expectation_eta = np.trapezoid(
            eta_dot, self.y_grid.flatten(), axis=2
        )  # n x d_tilting x 1
        z = self.eta_vectors - expectation_eta  # n x d_tilting x m
        tensor1 = np.einsum(
            "bkjl,bil -> bkjil", first_part, z
        )  # n x d_constraints x d_tilting x d_tilting x m

        probs_repeat_2 = np.tile(
            self.probs[:, None, None, None, :, :],
            (1, d_constraints, d_tilting, d_tilting, 1, 1),
        ).squeeze(
            -1
        )  # n x d_constraints x d_tilting x d_tilting x m
        tensor1_w_probs = (
            tensor1 * probs_repeat_2
        )  # n x d_constraints x d_tilting x d_tilting x m

        outer_expectation1 = np.trapezoid(
            tensor1_w_probs, self.y_grid.flatten(), axis=4
        )  # n x d_constraints x d_tilting x d_tilting
        C1_w_weight = (
            outer_expectation1 * self.weight[:, :, None, None]
        )  # n x d_constraints x d_tilting x d_tilting
        C1 = C1_w_weight.sum(axis=0)  # d_constraints x d_tilting x d_tilting

        inner_cov_eta = np.einsum("bik,bjk-> bijk", z, z)
        probs_repeat_3 = np.tile(
            self.probs[:, None, None, :, :], (1, d_tilting, d_tilting, 1, 1)
        ).squeeze(-1)
        inner_cov_eta_w_probs = (
            inner_cov_eta * probs_repeat_3
        )  # n x d_tilting x d_tilting x m
        cov_eta = np.trapezoid(
            inner_cov_eta_w_probs, self.y_grid.flatten(), axis=3
        )  # n x d_tilting x d_tilting
        probs_repeat_4 = np.tile(self.probs[:, None, :, :], (1, d_constraints, 1, 1))
        psi_dot = (
            self.psi_vectors[:, :, :, None] * probs_repeat_4
        )  # n x d_constraints x m
        expectation_psi = np.trapezoid(psi_dot, self.y_grid.flatten(), axis=2).squeeze(
            -1
        )  # n x d_constraints x 1
        tensor2 = np.einsum(
            "bi, bkl -> bikl", expectation_psi, cov_eta
        )  # n x d_constraints x d_tilting x d_tilting
        C2_w_weight = (
            tensor2 * self.weight[:, :, None, None]
        )  # n x d_constraints x d_tilting x d_tilting
        C2 = C2_w_weight.sum(axis=0)  # d_constraints x d_tilting x d_tilting

        inner_cov_eta_psi = first_part
        probs_repeat_5 = np.tile(
            self.probs[:, None, None, :, :], (1, d_constraints, d_tilting, 1, 1)
        )
        inner_cov_eta_psi_w_probs = (
            inner_cov_eta_psi[:, :, :, :, None] * probs_repeat_5
        )  # n x d_constraints x d_tilting x m x1
        cov_eta_psi = np.trapezoid(
            inner_cov_eta_psi_w_probs, self.y_grid.flatten(), axis=3
        ).squeeze(
            -1
        )  # n x d_constraints x d_tilting

        tensor3 = np.einsum(
            "bij,bk->bijk", cov_eta_psi, expectation_eta.squeeze(-1)
        )  # n x d_constraints x d_tilting x 1
        C3_w_weight = (
            tensor3 * self.weight[:, :, None, None]
        )  # n x d_constraints x d_tilting
        C3 = C3_w_weight.sum(axis=0)  # d_constraints x d_tilting x d_tilting

        C = C1 - C2 - C3

        final = np.vstack([C, -C])  # 2 * d_constraints x d_tilting x d_tilting
        return final

    def get_kl_divergence(self):
        return (
            self.theta.T @ self.get_rho()
            - (self.log_partition_function * self.weight).sum()
        )


class EstimatorsBinaryOutcome:
    def __init__(
        self, census_dataset, predictor, moment_group, tilting_spline, constraint_spline
    ):
        self.tilting_spline = tilting_spline
        self.constraint_spline = constraint_spline
        self.moment_group = moment_group
        self.predictor = predictor
        self.census_dataset = census_dataset

        self.precomputation()

    def precomputation(self):
        X, r, state_col = self.census_dataset.get_data(
            normalize_weight=True, return_state_col=self.moment_group
        )
        self.predictions = self.predictor(X)

        self.weight = r.reshape(-1, 1, 1) / r.sum()
        self.eta_vectors_1 = self.tilting_spline.get_spline(
            X, np.ones(1)
        )  # n x d_tilting x 1
        self.eta_vectors_0 = self.tilting_spline.get_spline(X, np.zeros(1))
        self.psi_vectors_1 = self.constraint_spline.get_spline(
            state_col, np.ones(1)
        )  # n x d_constraints x 1
        self.psi_vectors_0 = self.constraint_spline.get_spline(state_col, np.zeros(1))

    def set_theta(self, theta):
        self.theta = theta
        X, _ = self.census_dataset.get_data(return_state_col=None)

        if np.all(self.theta == 0):
            self.log_partition_function = 0.0
        else:
            self.log_partition_function = get_log_partition_function_binary_outcome(
                X,
                self.theta,
                self.predictor,
                self.tilting_spline,
            )
        self.prob1 = np.exp(
            np.transpose(self.eta_vectors_1, (0, 2, 1)) @ self.theta
            - self.log_partition_function
        ) * self.predictions.reshape(-1, 1, 1)

        self.prob0 = 1 - self.prob1

    def get_rho(self):
        term1 = self.eta_vectors_1 * self.prob1
        term0 = self.eta_vectors_0 * self.prob0
        term = (term1 + term0) * self.weight
        res = term.sum(axis=0)
        return res

    def get_m(self):
        term1 = self.psi_vectors_1 * self.prob1
        term0 = self.psi_vectors_0 * self.prob0
        term = (term1 + term0) * self.weight
        res = term.sum(axis=0)
        return res

    def get_B_matrix(self):
        """
        psi = n x d_constraints x m
        eta = n x d_tilting x m

        to

        psi n x
        """
        expectation_eta = (
            self.eta_vectors_1 * self.prob1 + self.eta_vectors_0 * self.prob0
        )
        expecatation_psi = (
            self.psi_vectors_1 * self.prob1 + self.psi_vectors_0 * self.prob0
        )
        z1 = (self.eta_vectors_1 - expectation_eta).squeeze(-1)
        w1 = (self.psi_vectors_1 - expecatation_psi).squeeze(-1)
        term1 = np.einsum("bi,bj->bij", w1, z1)  # n x d_constraints x d_tilting
        z0 = (self.eta_vectors_0 - expectation_eta).squeeze(-1)
        w0 = (self.psi_vectors_0 - expecatation_psi).squeeze(-1)
        term0 = np.einsum("bi,bj->bij", w0, z0)  # n x d_constraints x d_tilting
        term = (term1 * self.prob1 + term0 * self.prob0) * self.weight
        res = term.sum(axis=0)
        final = np.vstack([res, -res])  # 2 * d_constraints x d_tilting
        return final

    def get_I_matrix(self):

        expectation_eta = (
            self.eta_vectors_1 * self.prob1 + self.eta_vectors_0 * self.prob0
        )

        z1 = (self.eta_vectors_1 - expectation_eta).squeeze(-1)
        term1 = np.einsum("bi,bj->bij", z1, z1)  # n x d_tilting x d_tilting
        z0 = (self.eta_vectors_0 - expectation_eta).squeeze(-1)
        term0 = np.einsum("bi,bj->bij", z0, z0)  # n x d_tilting x d_tilting
        term = (term1 * self.prob1 + term0 * self.prob0) * self.weight
        res = term.sum(axis=0)  # d_tilting x d_tilting
        return res

    def get_D_tensor(self):
        """
        Returns a tensor of shape d_tilting x d_tilting
        """

        expectation_eta = (
            self.eta_vectors_1 * self.prob1 + self.eta_vectors_0 * self.prob0
        )

        z1 = (self.eta_vectors_1 - expectation_eta).squeeze(-1)
        z0 = (self.eta_vectors_0 - expectation_eta).squeeze(-1)

        tensor1_partial = np.einsum("bi,bj->bij", z1, z1)  # n x d_tilting x d_tilting
        tensor1 = np.einsum(
            "bij,bk->bijk", tensor1_partial, z1
        )  # n x d_tilting x d_tilting x d_tilting

        tensor0_partial = np.einsum("bi,bj->bij", z0, z0)  # n x d_tilting x d_tilting
        tensor0 = np.einsum(
            "bij,bk->bijk", tensor0_partial, z0
        )  # n x d_tilting x d_tilting x d_tilting

        tensor = (
            tensor1 * self.prob1[:, :, :, None] + tensor0 * self.prob0[:, :, :, None]
        ) * self.weight[:, :, :, None]
        return tensor.sum(axis=0)  # d_tilting x d_tilting x d_tilting

    def get_C_tensor(self):
        """
        Returns a tensor of shape d_constraints  x d_tilting x d_tilting
        """

        # TERM1
        part1 = np.matmul(
            self.psi_vectors_1, np.transpose(self.eta_vectors_1, (0, 2, 1))
        )  # n x d_constraints x d_tilting
        expectation_eta = (
            self.eta_vectors_1 * self.prob1
            + self.eta_vectors_0 * self.prob0  # n x d_tilting x 1
        )

        z1 = (self.eta_vectors_1 - expectation_eta).squeeze(-1)  # n x d_tilting
        tensor1 = np.einsum(
            "bij,bk ->bijk", part1, z1
        )  # n x d_constraints x d_tilting x d_tilting

        part0 = np.matmul(
            self.psi_vectors_0, np.transpose(self.eta_vectors_0, (0, 2, 1))
        )  # n x d_constraints x d_tilting
        z0 = (self.eta_vectors_0 - expectation_eta).squeeze(-1)  # n x d_tilting
        tensor0 = np.einsum(
            "bij,bk ->bijk", part0, z0
        )  # n x d_constraints x d_tilting x d_tilting

        tensor_term1 = (
            tensor1 * self.prob1[:, :, :, None] + tensor0 * self.prob0[:, :, :, None]
        )  # n x d_constraints x d_tilting x d_tilting
        tensor_term1_w_weight = (
            tensor_term1 * self.weight[:, :, :, None]
        )  # n x d_constraints x d_tilting x d_tilting
        tensor_term1_mean = tensor_term1_w_weight.sum(
            axis=0
        )  # d_constraints x d_tilting x d_tilting

        # TERM2
        expectation_psi = (
            self.psi_vectors_1 * self.prob1 + self.psi_vectors_0 * self.prob0
        ).squeeze(
            -1
        )  # n x d_constraints
        cov_z1 = np.einsum("bi,bj->bij", z1, z1)  # n x d_tilting x d_tilting
        cov_z0 = np.einsum("bi,bj->bij", z0, z0)  # n x d_tilting x d_tilting
        term2_part2 = (
            cov_z1 * self.prob1 + cov_z0 * self.prob0
        )  # n x d_tilting x d_tilting
        tensor_term2 = np.einsum(
            "bi,bjk->bijk", expectation_psi, term2_part2
        )  # n x d_constraints x d_tilting x d_tilting
        tensor_term2_w_weight = (
            tensor_term2 * self.weight[:, :, :, None]
        )  # n x d_constraints x d_tilting x d_tilting
        tensor_term2_mean = tensor_term2_w_weight.sum(
            axis=0
        )  # d_constraints x d_tilting x d_tilting

        # TERM3
        cov = part1 * self.prob1 + part0 * self.prob0  # n x d_constraints x d_tilting
        tensor_term3 = np.einsum(
            "bij,bk->bijk", cov, expectation_eta.squeeze(-1)
        )  # n x d_constraints x d_tilting x d_tilting
        tensor_term3_w_weight = (
            tensor_term3 * self.weight[:, :, :, None]
        )  # n x d_constraints x d_tilting x d_tilting
        tensor_term3_mean = tensor_term3_w_weight.sum(
            axis=0
        )  # d_constraints x d_tilting x d_tilting

        res_tensor = (
            tensor_term1_mean - tensor_term2_mean - tensor_term3_mean
        )  # d_constraints x d_tilting x d_tilting
        final = np.vstack([res_tensor, -res_tensor])
        return final

    def get_kl_divergence(self):
        return (
            self.theta.T @ self.get_rho()
            - (self.log_partition_function * self.weight).sum()
        )


def get_constraint_vector(national_gt_dataset, group_name, regions):
    m = np.zeros((len(regions), 1))
    prob_region = np.ones((len(regions), 1))

    _, y, nat_r, nat_state_col = national_gt_dataset.get_data(
        normalize_weight=True, return_state_col=group_name
    )
    for i, region in enumerate(regions.keys()):
        m[i] = (
            y[np.isin(nat_state_col, regions[region]).reshape(y.shape)][0]
            * nat_r[np.isin(nat_state_col, regions[region]).reshape(nat_r.shape)]
        ).sum()
        prob_region[i] = (
            nat_r[np.isin(nat_state_col, regions[region]).reshape(nat_r.shape)].sum()
            / nat_r.sum()
        )
    return m, prob_region


def solve_qp(hessian_approx, linear_term, B_matrix, m_diff, slack):
    """
    Solves the linear system of equations to find delta and lambda.
    """
    delta = cp.Variable((hessian_approx.shape[0], 1))
    n_constraints = int(B_matrix.shape[0] / 2)
    constraints = [
        B_matrix[:n_constraints, :] @ delta <= m_diff + slack,
        B_matrix[n_constraints:, :] @ delta <= -m_diff + slack,
        #    B_matrix @ delta == m_diff
    ]
    objective = cp.Minimize(
        0.5 * cp.quad_form(delta, hessian_approx) + (linear_term.T @ delta)
    )
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise ValueError("QP problem is not optimal")
    delta = delta.value
    lamb = np.concat((constraints[0].dual_value, constraints[1].dual_value), axis=0)
    return delta, lamb


def merit_function(theta, constraint_m, mu_inv, estimators, slack=1e-2):
    estimators.set_theta(theta)
    kl = estimators.get_kl_divergence().item()
    m = estimators.get_m()
    merit = (
        kl
        + mu_inv * np.maximum(m - constraint_m - slack, np.zeros(m.shape)).sum()
        + mu_inv * (np.maximum(constraint_m - slack - m, np.zeros(m.shape))).sum()
    )
    return merit


def learn_theta(
    estimators,
    theta_init,
    lamb_init,
    constraint_m,
    slack=1e-2,
    n_epochs=100,
):
    estimators.set_theta(theta_init)
    theta = theta_init
    tol = 1e-2
    alpha = 1
    lamb = lamb_init
    mu_inv = 100
    for i in range(n_epochs):
        B_matrix = estimators.get_B_matrix()
        I_matrix = estimators.get_I_matrix()
        D_tensor = estimators.get_D_tensor()
        C_tensor = estimators.get_C_tensor()
        lamb_C_matrix = np.einsum("k,kij->ij", lamb.flatten(), C_tensor)
        lamb_B_vector = np.einsum("k,ki->i", lamb.flatten(), B_matrix)  # d_tilting x 1
        D_theta_matrix = np.einsum("ijk,k->ij", D_tensor, theta.flatten())
        I_theta_vector = I_matrix @ theta.flatten()  # d_tilting x 1
        m = estimators.get_m()
        m_diff = constraint_m - m
        constraint_violation = np.abs(m_diff).max()

        hessian_approx = I_matrix + D_theta_matrix - lamb_C_matrix
        linear_term = I_theta_vector - lamb_B_vector

        print("Constraint Violation:", constraint_violation)

        try:
            delta, nu = solve_qp(
                hessian_approx, linear_term, B_matrix, m_diff, slack=slack
            )
        except cp.error.DCPError as e:
            import pdb

            pdb.set_trace()
            print("QP problem is not optimal, stopping")
        except ValueError as e:
            import pdb

            pdb.set_trace()
            print("QP problem is not optimal, stopping")
        decrement = np.linalg.norm(delta) / delta.shape[0]
        print("Decrement", decrement)
        if decrement < tol:
            estimators.set_theta(theta)
            print("Converged")
            break

        if 1 / mu_inv < np.max(np.abs(lamb)) + 1e-3:
            mu_inv = np.max(np.abs(lamb)) + 2e-3
        alphas = np.logspace(start=-3, stop=0, base=10, num=5)
        # alphas = np.logspace(start=-3, stop=-0.5, base=10, num=5)
        merits = []
        for alpha in alphas:
            new_merit = merit_function(
                theta + alpha * delta,
                constraint_m,
                mu_inv=mu_inv,
                estimators=estimators,
                slack=slack,
            )
            print("New Merit", new_merit, "Alpha", alpha)
            merits.append(new_merit)
        alpha_best = alphas[np.argmin(merits)]
        theta = theta + alpha_best * delta
        lamb = lamb + 1e-2 * (nu - lamb)
        estimators.set_theta(theta)
        print("Best Alpha", alpha_best)

        if alpha_best == 1e-3:
            print("Alpha is too small, stopping")
            break

        print(
            i,
            "Theta:",
            theta,
            "Lambda",
            lamb,
            "Delta:",
            delta,
            "Alpha:",
            alpha_best,
            "KL:",
            estimators.get_kl_divergence(),
        )

    return theta, nu


def get_smart_initialization(
    indicator,
    census_dataset,
    predictor,
    constraint_m,
    regions,
    moment_group,
    slack=0.01,
    n_epochs=100,
):
    """
    Returns a smart initialization for theta.
    """
    np.random.seed(12359038)
    X, r = census_dataset.get_data(return_state_col=None)
    new_regions = {"All": []}
    for region in regions.keys():
        new_regions["All"].extend(regions[region])
    constraint_spline = ConstraintSpline(new_regions, moment_group=moment_group)
    tilting_spline = TiltingSpline(
        census_dataset,
        balancing_features=["intercept"],
    )
    constraint_m = constraint_m.sum(axis=0).reshape(-1, 1)

    theta_init = np.random.normal(size=(tilting_spline.d_tilting, 1), scale=0.25)
    lamb_init = np.zeros(shape=(constraint_spline.d_constraints * 2, 1))

    if indicator in ["medicaid_ins", "snap", "RECVDVACC"]:
        estimators = EstimatorsBinaryOutcome(
            census_dataset=census_dataset,
            predictor=predictor,
            moment_group=moment_group,
            tilting_spline=tilting_spline,
            constraint_spline=constraint_spline,
        )
    else:
        estimators = EstimatorsContinuousOutcome(
            census_dataset=census_dataset,
            cond_density_estimator=predictor,
            group_name="X",
            tilting_spline=tilting_spline,
            constraint_spline=constraint_spline,
        )

    theta = learn_theta(
        estimators,
        theta_init,
        lamb_init,
        constraint_m,
        slack=slack,
        n_epochs=n_epochs,
    )
    return theta


def get_exponential_family_state_estimates(
    indicator,
    predictor,
    predictor_metadata,
    census_dataset,
    online_survey_dataset,
    national_gt_dataset,
    regions,
    evaluation_group,
    moment_group,
    slack=0.01,
    balancing_features=None,
    n_epochs=100,
    seed=12359038,
):

    group_name = moment_group

    np.random.seed(seed)
    X, r = census_dataset.get_data(return_state_col=None)

    constraint_spline = ConstraintSpline(regions, moment_group=moment_group)
    tilting_spline = TiltingSpline(
        census_dataset,
        balancing_features,
    )
    constraint_m, prob_region = get_constraint_vector(
        national_gt_dataset, moment_group, regions
    )
    theta_init = np.zeros(shape=(tilting_spline.d_tilting, 1))
    if tilting_spline.d_tilting > 1:
        theta_best_intercept = get_smart_initialization(
            indicator,
            census_dataset,
            predictor,
            constraint_m,
            regions,
            moment_group,
            slack=0.01,
            n_epochs=100,
        )
        theta_init[-1] = theta_best_intercept.item()
    lamb_init = np.zeros(shape=(constraint_spline.d_constraints * 2, 1))

    if online_survey_dataset.binary_indicator is True:
        estimators = EstimatorsBinaryOutcome(
            census_dataset=census_dataset,
            predictor=predictor,
            moment_group=moment_group,
            tilting_spline=tilting_spline,
            constraint_spline=constraint_spline,
        )
    else:
        estimators = EstimatorsContinuousOutcome(
            census_dataset=census_dataset,
            cond_density_estimator=predictor,
            group_name=group_name,
            tilting_spline=tilting_spline,
            constraint_spline=constraint_spline,
        )

    theta, nu = learn_theta(
        estimators,
        theta_init,
        lamb_init,
        constraint_m,
        slack=slack,
        n_epochs=n_epochs,
    )

    lamb = (
        nu[: constraint_spline.d_constraints] - nu[constraint_spline.d_constraints :]
    ).reshape(1, -1)
    mu_eta = estimators.get_rho().flatten()

    if online_survey_dataset.binary_indicator is True:
        stats_inference = BinaryStatisticalInference(
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
        )
        one_step_nu, asymptotic_variance, n = stats_inference.get_asymptotic_variance()
        group_to_indexes = (
            census_dataset.df.groupby(evaluation_group)
            .apply(lambda x: x.index.tolist())
            .to_dict()
        )
        estimates = []
        for group, indexes in group_to_indexes.items():
            res = {}
            for i, name in enumerate(evaluation_group):
                if type(group) is str:
                    res[name] = group
                elif type(group) is tuple:
                    res[name] = group[i]

                estimate, (lower, upper) = stats_inference.get_ci(
                    indexes=indexes,
                    one_step_nu=one_step_nu,
                    asymptotic_variance=asymptotic_variance,
                    n=n,
                )
                res.update(
                    {
                        indicator: estimate.item(),
                        indicator + "_upper": upper.item(),
                        indicator + "_lower": lower.item(),
                    }
                )
                print(res)
                estimates.append(res)
        final_df = pd.DataFrame.from_records(estimates)
        return final_df

    else:
        estimators.set_theta(theta)
        conditional_mean = np.trapezoid(
            estimators.y_grid * estimators.probs.squeeze(-1),
            estimators.y_grid.flatten(),
            axis=1,
        )
        final_df = pd.DataFrame({"X": X[:, 0].flatten(), indicator: conditional_mean})
        return final_df
