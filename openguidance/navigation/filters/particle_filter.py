"""Particle Filter implementation for OpenGuidance navigation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
import logging

from .base import Filter, FilterConfig
from openguidance.core.types import State

logger = logging.getLogger(__name__)


@dataclass
class ParticleFilterConfig(FilterConfig):
    """Configuration for Particle Filter."""
    num_particles: int = 1000
    resampling_threshold: float = 0.5  # Effective sample size threshold
    resampling_method: str = "systematic"  # "systematic", "stratified", "multinomial"
    
    # Roughening parameters
    roughening_factor: float = 0.01
    enable_roughening: bool = True


class ParticleFilter(Filter):
    """Particle Filter for nonlinear state estimation.
    
    The particle filter represents the posterior distribution using
    a set of weighted particles.
    """
    
    def __init__(self, config: ParticleFilterConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize noise matrices
        self.process_noise = config.process_noise_std**2 * np.eye(config.state_dim)
        self.measurement_noise = config.process_noise_std**2 * np.eye(config.state_dim // 2)  # Assume position measurements
        self.n_measurements = config.state_dim // 2  # Assume position measurements
        
        # Initialize particles
        self.particles = np.zeros((config.num_particles, config.state_dim))
        self.weights = np.ones(config.num_particles) / config.num_particles
        
        # Initialize state tracking
        self.state_estimate = np.zeros(config.state_dim)
        self.covariance = np.eye(config.state_dim)
        self.prediction_count = 0
        self.update_count = 0
        
        # Effective sample size
        self.effective_sample_size = config.num_particles
        
        logger.info(f"Particle Filter initialized with {config.num_particles} particles")
    
    def predict(self, state: State, dt: float, control_input: Optional[np.ndarray] = None) -> None:
        """Prediction step of the particle filter."""
        # Propagate each particle through the dynamics model
        for i in range(self.config.num_particles):
            # Add process noise
            process_noise = np.random.multivariate_normal(
                np.zeros(self.config.state_dim),
                self.process_noise
            )
            
            # Propagate particle
            self.particles[i] = self._dynamics_model(
                self.particles[i], dt, control_input
            ) + process_noise
        
        # Update state estimate (weighted mean)
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Update covariance
        self._update_covariance()
        
        self.prediction_count += 1
        logger.debug(f"Particle filter prediction step {self.prediction_count} completed")
    
    def update(self, measurement: np.ndarray, measurement_noise: Optional[np.ndarray] = None) -> None:
        """Update step of the particle filter."""
        if measurement_noise is None:
            measurement_noise = self.measurement_noise
        
        # Update particle weights based on likelihood
        for i in range(self.config.num_particles):
            predicted_measurement = self._measurement_model(self.particles[i])
            
            # Compute likelihood
            innovation = measurement - predicted_measurement
            
            # Multivariate Gaussian likelihood
            try:
                if measurement_noise is not None:
                    inv_cov = np.linalg.inv(measurement_noise)
                    det_cov = np.linalg.det(measurement_noise)
                    likelihood = np.exp(-0.5 * innovation.T @ inv_cov @ innovation)
                    likelihood /= np.sqrt((2 * np.pi) ** len(measurement) * det_cov)
                else:
                    likelihood = 1e-10
            except np.linalg.LinAlgError:
                # Fallback for singular covariance
                likelihood = 1e-10
            
            self.weights[i] *= likelihood
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-15:
            self.weights /= weight_sum
        else:
            # Reset to uniform if all weights are zero
            self.weights = np.ones(self.config.num_particles) / self.config.num_particles
            logger.warning("All particle weights became zero, resetting to uniform")
        
        # Compute effective sample size
        self.effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        
        # Resample if necessary
        if self.effective_sample_size < self.config.resampling_threshold * self.config.num_particles:
            self._resample()
        
        # Update state estimate
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        self._update_covariance()
        
        self.update_count += 1
        logger.debug(f"Particle filter update step {self.update_count} completed")
    
    def _resample(self) -> None:
        """Resample particles based on their weights."""
        if self.config.resampling_method == "systematic":
            indices = self._systematic_resample()
        elif self.config.resampling_method == "stratified":
            indices = self._stratified_resample()
        elif self.config.resampling_method == "multinomial":
            indices = self._multinomial_resample()
        else:
            raise ValueError(f"Unknown resampling method: {self.config.resampling_method}")
        
        # Resample particles
        self.particles = self.particles[indices]
        
        # Reset weights to uniform
        self.weights = np.ones(self.config.num_particles) / self.config.num_particles
        
        # Apply roughening if enabled
        if self.config.enable_roughening:
            self._apply_roughening()
        
        # Reset effective sample size
        self.effective_sample_size = self.config.num_particles
        
        logger.debug("Particle resampling completed")
    
    def _systematic_resample(self) -> np.ndarray:
        """Systematic resampling."""
        indices = np.zeros(self.config.num_particles, dtype=int)
        
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)
        
        # Generate systematic samples
        u = np.random.random() / self.config.num_particles
        j = 0
        
        for i in range(self.config.num_particles):
            while cumsum[j] < u:
                j += 1
            indices[i] = j
            u += 1.0 / self.config.num_particles
        
        return indices
    
    def _stratified_resample(self) -> np.ndarray:
        """Stratified resampling."""
        indices = np.zeros(self.config.num_particles, dtype=int)
        
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)
        
        # Generate stratified samples
        for i in range(self.config.num_particles):
            u = (i + np.random.random()) / self.config.num_particles
            j = 0
            while cumsum[j] < u:
                j += 1
            indices[i] = j
        
        return indices
    
    def _multinomial_resample(self) -> np.ndarray:
        """Multinomial resampling."""
        return np.random.choice(
            self.config.num_particles,
            size=self.config.num_particles,
            p=self.weights
        )
    
    def _apply_roughening(self) -> None:
        """Apply roughening to prevent particle depletion."""
        # Compute sample covariance
        sample_cov = np.cov(self.particles.T)
        
        # Roughening noise
        roughening_cov = self.config.roughening_factor * sample_cov
        
        # Add noise to particles
        noise = np.random.multivariate_normal(
            np.zeros(self.config.state_dim),
            roughening_cov,
            size=self.config.num_particles
        )
        
        self.particles += noise
    
    def _update_covariance(self) -> None:
        """Update covariance matrix from particles."""
        # Weighted covariance
        mean = self.state_estimate
        weighted_diff = np.sqrt(self.weights)[:, np.newaxis] * (self.particles - mean)
        self.covariance = weighted_diff.T @ weighted_diff
    
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
        from pyquaternion import Quaternion
        
        # Initialize with default values
        position = np.zeros(3)
        velocity = np.zeros(3)
        attitude = Quaternion()
        angular_velocity = np.zeros(3)
        
        # Extract from state estimate
        if len(self.state_estimate) >= 3:
            position = self.state_estimate[:3]
        if len(self.state_estimate) >= 6:
            velocity = self.state_estimate[3:6]
        if len(self.state_estimate) >= 9:
            # Convert Euler angles to quaternion
            euler = self.state_estimate[6:9]
            attitude = Quaternion(axis=[0, 0, 1], angle=euler[2]) * \
                      Quaternion(axis=[0, 1, 0], angle=euler[1]) * \
                      Quaternion(axis=[1, 0, 0], angle=euler[0])
        if len(self.state_estimate) >= 12:
            angular_velocity = self.state_estimate[9:12]
        
        state = State(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            time=0.0
        )
        
        return state
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get particle filter specific diagnostic information."""
        diagnostics = self.get_statistics()
        
        diagnostics.update({
            "particle_filter_specific": {
                "num_particles": self.config.num_particles,
                "effective_sample_size": self.effective_sample_size,
                "resampling_threshold": self.config.resampling_threshold,
                "resampling_method": self.config.resampling_method,
                "weight_statistics": {
                    "min": float(np.min(self.weights)),
                    "max": float(np.max(self.weights)),
                    "mean": float(np.mean(self.weights)),
                    "std": float(np.std(self.weights))
                }
            }
        })
        
        return diagnostics
    
    def get_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current particles and weights."""
        return self.particles.copy(), self.weights.copy()
    
    def set_particles(self, particles: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """Set particles and weights."""
        if particles.shape[0] != self.config.num_particles:
            raise ValueError(f"Expected {self.config.num_particles} particles, got {particles.shape[0]}")
        
        if particles.shape[1] != self.config.state_dim:
            raise ValueError(f"Expected state dimension {self.config.state_dim}, got {particles.shape[1]}")
        
        self.particles = particles.copy()
        
        if weights is not None:
            if len(weights) != self.config.num_particles:
                raise ValueError(f"Expected {self.config.num_particles} weights, got {len(weights)}")
            self.weights = weights.copy()
            # Normalize weights
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.config.num_particles) / self.config.num_particles
        
        # Update state estimate
        self.state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        self._update_covariance()
        
        # Update effective sample size
        self.effective_sample_size = 1.0 / np.sum(self.weights ** 2) 