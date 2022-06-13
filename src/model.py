from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lars
from sklearn.base import BaseEstimator

from src.constants import Columns


MIN_EIG_VALUE = 1e-10
EPSILON = 1e-20


class TreeElastic(BaseEstimator):
    def __init__(self, mean_shrinkage: float = 0., ridge_lambda: float = 0.5,
                 k_min: int = 0, k_max: float = np.inf):
        self.mean_shrinkage = mean_shrinkage
        self.ridge_lambda = ridge_lambda
        self.base_model = Lars(normalize=False, fit_intercept=False, n_nonzero_coefs=k_max)
        self.k_min = k_min
        self.k_max = k_max
        self.feature_weights = None
        self.betas = None

    @staticmethod
    def decompose_covariance(feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposes input feature df into Eigen-values/vectors. Because the decomposition could return complex numbers,
        take only the real part of them.

        Parameters
        ----------
        feature_df:
            Input feature df of size (n_observation, n_features)

        Returns
        -------
            Tuple of Eigen Values and Eigen Vectors
        """
        sigma = feature_df.cov()
        eig_values, eig_vectors = np.linalg.eig(sigma)
        eig_values = eig_values.real
        eig_vectors = eig_vectors.real
        gamma = min(feature_df.shape[0], sum(eig_values > MIN_EIG_VALUE))
        return eig_values[:gamma], eig_vectors[:gamma]

    def process_input(self, feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        eig_values, eig_vectors = self.decompose_covariance(feature_df)
        sigma_tilde = eig_vectors.T @ np.diag(np.sqrt(eig_values)) @ eig_vectors

        mu_tilde = eig_vectors.T @  np.diag(1 / np.sqrt(eig_values)) @ eig_vectors
        mu_tilde = mu_tilde @ (
                feature_df.mean() + self.mean_shrinkage * feature_df.values.mean()
        ).values

        n_feats = sigma_tilde.shape[0]
        sigma_tilde = np.vstack([sigma_tilde, np.diag(np.array([np.sqrt(self.ridge_lambda)] * n_feats))])
        mu_tilde = np.hstack([mu_tilde, np.array([0] * n_feats)])

        return sigma_tilde, mu_tilde

    @staticmethod
    def get_feature_weights(feature_df: pd.DataFrame) -> np.ndarray:
        """
        Based on the position of a Node in the tree - calculate its weight as 1 / SQRT(2 ^ node-depth).

        Parameters
        ----------
        feature_df:
            Input feature df of size (n_observation, n_features)

        Returns
        -------
            Array with adjusting weight per feature
        """
        depth_per_col = np.array([
            int(c.replace(f"{Columns.port_col}{Columns.col_sep}", ""))
            for c in feature_df.columns.get_level_values(Columns.port_col)
        ])
        return 1 / np.sqrt(2 ** depth_per_col)

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs) -> 'TreeElastic':
        feature_df = X.copy()
        self.feature_weights = self.get_feature_weights(feature_df)

        feature_df = feature_df.multiply(self.feature_weights)

        # Transform the returns into the model inputs
        input_x, input_y = self.process_input(feature_df)

        self.base_model.fit(input_x, input_y)
        # Adjust coefficients and normalize
        self.betas = (self.base_model.coef_path_.T * self.feature_weights)

        self.betas = (self.betas.T / (np.abs(np.sum(self.betas, axis=1)) + EPSILON)).T

        num_not_zero = np.sum(self.betas != 0, axis=1)
        mask = (num_not_zero >= self.k_min) & (num_not_zero <= self.k_max)
        if not mask.any():
            self.betas = np.zeros(self.betas.shape)
        else:
            self.betas = self.betas[mask]
        return self

    def predict(self, X: pd.DataFrame, y=None, *args, **kwargs) -> np.ndarray:
        feature_df = X.copy()
        feature_df = feature_df.multiply(self.feature_weights)
        return feature_df @ (self.betas / self.feature_weights).T

    def score(self, X: pd.DataFrame, y=None, *args, **kwargs):
        sdf = self.predict(X)
        sharpe_values = np.mean(sdf, axis=0) / (np.std(sdf, axis=0) + EPSILON)
        return max(sharpe_values)
