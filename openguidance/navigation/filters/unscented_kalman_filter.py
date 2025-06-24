"""Unscented Kalman Filter implementation for OpenGuidance navigation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

from .base import Filter, FilterConfig
from openguidance.core.types import State

logger = logging.getLogger(__name__)


@dataclass
class UKFConfig(FilterConfig):
    """Configuration for Unscented Kalman Filter."""
    alpha: float = 1e-3  # Spread of sigma points
    beta: float = 2.0    # Prior knowledge parameter (Gaussian = 2)
    kappa: float = 0.0   # Secondary scaling parameter
    
    # Square-root filtering
    use_square_root: bool = True
    
    # Sigma point selection method
    sigma_point_method: str = "symmetric"  # "symmetric", "simplex", "spherical"


class UnscentedKalmanFilter(Filter):
    """Unscented Kalman Filter for nonlinear state estimation.
    
    The UKF uses the unscented transform to handle nonlinear dynamics
    and measurement models without linearization.
    """
    
    def __init__(self, config: UKFConfig):
        super().__init__(config)
        self.config = config
        
        # UKF parameters
        self.n_states = config.state_dim
        self.n_measurements = config.measurement_dim
        
        # Calculate lambda parameter
        self.lambda_param = config.alpha**2 * (self.n_states + config.kappa) - self.n_states
        
        # Number of sigma points
        self.n_sigma = 2 * self.n_states + 1
        
        # Weights for mean and covariance
        self.weights_mean = np.zeros(self.n_sigma)
        self.weights_cov = np.zeros(self.n_sigma)
        
        self._compute_weights()
        
        # Square-root matrices
        if config.use_square_root:
            self.S_x = None  # Square root of state covariance
            self.S_z = None  # Square root of measurement covariance
        
        logger.info(f"UKF initialized with {self.n_states} states, {self.n_measurements} measurements")
    
    def _compute_weights(self) -> None:
        """Compute sigma point weights."""
        # Weight for first sigma point (mean)
        self.weights_mean[0] = self.lambda_param / (self.n_states + self.lambda_param)
        self.weights_cov[0] = self.weights_mean[0] + (1 - self.config.alpha**2 + self.config.beta)
        
        # Weights for remaining sigma points
        weight = 1.0 / (2 * (self.n_states + self.lambda_param))
        self.weights_mean[1:] = weight
        self.weights_cov[1:] = weight
    
    def _generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate sigma points using the specified method."""
        sigma_points = np.zeros((self.n_sigma, self.n_states))
        
        if self.config.sigma_point_method == "symmetric":
            return self._generate_symmetric_sigma_points(mean, cov)
        elif self.config.sigma_point_method == "simplex":
            return self._generate_simplex_sigma_points(mean, cov)
        elif self.config.sigma_point_method == "spherical":
            return self._generate_spherical_sigma_points(mean, cov)
        else:
            raise ValueError(f"Unknown sigma point method: {self.config.sigma_point_method}")
    
    def _generate_symmetric_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate symmetric sigma points."""
        sigma_points = np.zeros((self.n_sigma, self.n_states))
        
        # First sigma point is the mean
        sigma_points[0] = mean
        
        # Compute matrix square root
        if self.config.use_square_root and self.S_x is not None:
            sqrt_matrix = self.S_x * np.sqrt(self.n_states + self.lambda_param)
        else:
            try:
                sqrt_matrix = np.linalg.cholesky((self.n_states + self.lambda_param) * cov)
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
                sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals * (self.n_states + self.lambda_param)))
        
        # Generate positive and negative sigma points
        for i in range(self.n_states):
            sigma_points[i + 1] = mean + sqrt_matrix[:, i]
            sigma_points[i + 1 + self.n_states] = mean - sqrt_matrix[:, i]
        
        return sigma_points
    
    def _generate_simplex_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate simplex sigma points (minimal set)."""
        # Simplified implementation - would need full simplex algorithm
        return self._generate_symmetric_sigma_points(mean, cov)
    
    def _generate_spherical_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate spherical sigma points."""
        # Simplified implementation - would need spherical algorithm
        return self._generate_symmetric_sigma_points(mean, cov)
    
    def predict(self, state: State, dt: float, control_input: Optional[np.ndarray] = None) -> None:
        """Prediction step of the UKF."""
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state_estimate, self.covariance)
        
        # Propagate sigma points through dynamics
        predicted_sigma_points = np.zeros_like(sigma_points)
        for i, sigma_point in enumerate(sigma_points):
            predicted_sigma_points[i] = self._dynamics_model(sigma_point, dt, control_input)
        
        # Compute predicted mean and covariance
        self.state_estimate = np.sum(self.weights_mean[:, np.newaxis] * predicted_sigma_points, axis=0)
        
        # Predicted covariance
        self.covariance = self.process_noise.copy()
        for i, sigma_point in enumerate(predicted_sigma_points):
            diff = sigma_point - self.state_estimate
            self.covariance += self.weights_cov[i] * np.outer(diff, diff)
        
        # Update square-root matrix if using square-root filtering
        if self.config.use_square_root:
            self._update_square_root_prediction(predicted_sigma_points)
        
        # Ensure numerical stability
        self._ensure_positive_definite()
        
        self.prediction_count += 1
        logger.debug(f"UKF prediction step {self.prediction_count} completed")
    
    def update(self, measurement: np.ndarray, measurement_noise: Optional[np.ndarray] = None) -> None:
        """Update step of the UKF."""
        if measurement_noise is None:
            measurement_noise = self.measurement_noise
        
        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points(self.state_estimate, self.covariance)
        
        # Transform sigma points through measurement model
        measurement_sigma_points = np.zeros((self.n_sigma, self.n_measurements))
        for i, sigma_point in enumerate(sigma_points):
            measurement_sigma_points[i] = self._measurement_model(sigma_point)
        
        # Predicted measurement mean
        predicted_measurement = np.sum(self.weights_mean[:, np.newaxis] * measurement_sigma_points, axis=0)
        
        # Innovation covariance
        innovation_cov = measurement_noise.copy()
        for i, meas_sigma in enumerate(measurement_sigma_points):
            diff = meas_sigma - predicted_measurement
            innovation_cov += self.weights_cov[i] * np.outer(diff, diff)
        
        # Cross-covariance
        cross_cov = np.zeros((self.n_states, self.n_measurements))
        for i in range(self.n_sigma):
            state_diff = sigma_points[i] - self.state_estimate
            meas_diff = measurement_sigma_points[i] - predicted_measurement
            cross_cov += self.weights_cov[i] * np.outer(state_diff, meas_diff)
        
        # Kalman gain
        try:
            kalman_gain = cross_cov @ np.linalg.inv(innovation_cov)
        except np.linalg.LinAlgError:
            kalman_gain = cross_cov @ np.linalg.pinv(innovation_cov)
            logger.warning("Using pseudo-inverse for Kalman gain computation")
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Innovation gating
        if self._innovation_gate(innovation, innovation_cov):
            # Update state and covariance
            self.state_estimate += kalman_gain @ innovation
            self.covariance -= kalman_gain @ innovation_cov @ kalman_gain.T
            
            # Joseph form update for numerical stability
            I_KH = np.eye(self.n_states) - kalman_gain @ np.eye(self.n_measurements)
            self.covariance = I_KH @ self.covariance @ I_KH.T + kalman_gain @ measurement_noise @ kalman_gain.T
            
            # Update square-root matrix if using square-root filtering
            if self.config.use_square_root:
                self._update_square_root_correction(kalman_gain, innovation_cov)
            
            # Store innovation statistics
            self.innovation_history.append(innovation)
            self.innovation_covariance_history.append(innovation_cov)
            
            # Update adaptive noise if enabled
            if self.config.adaptive_noise:
                self._update_adaptive_noise(innovation, innovation_cov)
            
            self.update_count += 1
            logger.debug(f"UKF update step {self.update_count} completed")
        else:
            logger.warning("Measurement rejected by innovation gate")
    
    def _update_square_root_prediction(self, predicted_sigma_points: np.ndarray) -> None:
        """Update square-root matrix during prediction."""
        # Centered sigma points
        centered_points = predicted_sigma_points - self.state_estimate
        
        # Weight matrix
        W = np.diag(np.sqrt(np.abs(self.weights_cov[1:])))
        
        # QR decomposition
        A = np.hstack([
            W @ centered_points[1:].T,
            np.linalg.cholesky(self.process_noise)
        ])
        
        Q, R = np.linalg.qr(A.T)
        self.S_x = R[:self.n_states, :self.n_states].T
        
        # Handle first sigma point separately
        if self.weights_cov[0] < 0:
            self.S_x = self._cholupdate(self.S_x, centered_points[0], -1)
        else:
            self.S_x = self._cholupdate(self.S_x, centered_points[0], 1)
    
    def _update_square_root_correction(self, kalman_gain: np.ndarray, innovation_cov: np.ndarray) -> None:
        """Update square-root matrix during correction."""
        # Simplified square-root update
        try:
            self.S_x = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
            eigenvals = np.maximum(eigenvals, 1e-12)
            self.S_x = eigenvecs @ np.diag(np.sqrt(eigenvals))
    
    def _cholupdate(self, R: np.ndarray, x: np.ndarray, sign: int) -> np.ndarray:
        """Cholesky rank-1 update/downdate."""
        # Simplified implementation
        if sign > 0:
            # Rank-1 update
            return np.linalg.cholesky(R @ R.T + np.outer(x, x))
        else:
            # Rank-1 downdate (simplified)
            return np.linalg.cholesky(np.maximum(R @ R.T - np.outer(x, x), 
                                               1e-12 * np.eye(len(x))))
    
    def _dynamics_model(self, state: np.ndarray, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Nonlinear dynamics model."""
        # Default constant velocity model
        n = len(state) // 2
        F = np.eye(len(state))
        F[:n, n:] = dt * np.eye(n)
        
        next_state = F @ state
        
        if control_input is not None:
            # Add control input (simplified)
            next_state[:n] += 0.5 * dt**2 * control_input[:n] if len(control_input) >= n else 0
            next_state[n:] += dt * control_input[:n] if len(control_input) >= n else 0
        
        return next_state
    
    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        """Nonlinear measurement model."""
        # Default identity measurement model
        return state[:self.n_measurements]
    
    def get_state_estimate(self) -> State:
        """Get current state estimate as State object."""
        state = State()
        
        if len(self.state_estimate) >= 3:
            state.position = self.state_estimate[:3]
        if len(self.state_estimate) >= 6:
            state.velocity = self.state_estimate[3:6]
        if len(self.state_estimate) >= 9:
            # Convert Euler angles to quaternion (simplified)
            from pyquaternion import Quaternion
            euler = self.state_estimate[6:9]
            state.attitude = Quaternion(axis=[0, 0, 1], angle=euler[2]) * \
                           Quaternion(axis=[0, 1, 0], angle=euler[1]) * \
                           Quaternion(axis=[1, 0, 0], angle=euler[0])
        if len(self.state_estimate) >= 12:
            state.angular_velocity = self.state_estimate[9:12]
        
        return state
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get UKF-specific diagnostic information."""
        diagnostics = super().get_diagnostics()
        
        diagnostics.update({
            "ukf_specific": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "kappa": self.config.kappa,
                "lambda": self.lambda_param,
                "n_sigma_points": self.n_sigma,
                "sigma_point_method": self.config.sigma_point_method,
                "use_square_root": self.config.use_square_root,
                "weights_mean_sum": np.sum(self.weights_mean),
                "weights_cov_sum": np.sum(self.weights_cov)
            }
        })
        
        return diagnostics 