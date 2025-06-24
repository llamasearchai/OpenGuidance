"""
Linear Quadratic Regulator (LQR) controller implementation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import scipy.linalg
import logging

logger = logging.getLogger(__name__)


@dataclass
class LQRGains:
    """LQR controller gains and parameters."""
    Q: np.ndarray  # State cost matrix
    R: np.ndarray  # Control cost matrix
    N: Optional[np.ndarray] = None  # Cross-coupling matrix
    
    def __post_init__(self):
        """Validate gain matrices."""
        if self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q matrix must be square")
        if self.R.shape[0] != self.R.shape[1]:
            raise ValueError("R matrix must be square")
        
        # Check positive definiteness
        if not np.all(np.linalg.eigvals(self.Q) >= 0):
            logger.warning("Q matrix is not positive semi-definite")
        if not np.all(np.linalg.eigvals(self.R) > 0):
            raise ValueError("R matrix must be positive definite")


class LQRController:
    """Linear Quadratic Regulator controller."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, gains: LQRGains):
        """Initialize LQR controller.
        
        Args:
            A: State transition matrix
            B: Control input matrix
            gains: LQR cost matrices
        """
        self.A = A
        self.B = B
        self.gains = gains
        
        # Validate dimensions
        n_states = A.shape[0]
        n_controls = B.shape[1]
        
        if A.shape != (n_states, n_states):
            raise ValueError("A matrix must be square")
        if B.shape[0] != n_states:
            raise ValueError("B matrix rows must match A matrix size")
        if gains.Q.shape != (n_states, n_states):
            raise ValueError("Q matrix size must match state dimension")
        if gains.R.shape != (n_controls, n_controls):
            raise ValueError("R matrix size must match control dimension")
        
        # Solve Riccati equation
        self.K, self.P = self._solve_lqr()
        
        # Controller state
        self.reference = np.zeros(n_states)
        self.integral_error = np.zeros(n_states)
        self.enable_integral = False
        self.integral_gain = 0.0
        
        logger.info(f"LQR controller initialized with {n_states} states and {n_controls} controls")
    
    def _solve_lqr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the discrete-time LQR problem."""
        try:
            # Solve discrete algebraic Riccati equation
            P = scipy.linalg.solve_discrete_are(
                self.A, self.B, self.gains.Q, self.gains.R, 
                s=self.gains.N
            )
            
            # Compute optimal gain matrix
            if self.gains.N is not None:
                K = np.linalg.inv(self.gains.R + self.B.T @ P @ self.B) @ (
                    self.B.T @ P @ self.A + self.gains.N.T
                )
            else:
                K = np.linalg.inv(self.gains.R + self.B.T @ P @ self.B) @ (
                    self.B.T @ P @ self.A
                )
            
            return K, P
            
        except Exception as e:
            logger.error(f"Failed to solve LQR: {e}")
            # Fallback to simple gains
            n_states = self.A.shape[0]
            n_controls = self.B.shape[1]
            K = np.eye(n_controls, n_states) * 0.1
            P = np.eye(n_states)
            return K, P
    
    def update(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Compute LQR control command.
        
        Args:
            state: Current state vector
            dt: Time step for integral control
            
        Returns:
            Control command vector
        """
        # State error
        error = state - self.reference
        
        # Basic LQR control
        control = -self.K @ error
        
        # Optional integral control
        if self.enable_integral:
            self.integral_error += error * dt
            control += -self.integral_gain * self.integral_error
        
        return control
    
    def set_reference(self, reference: np.ndarray) -> None:
        """Set reference state."""
        if len(reference) != len(self.reference):
            raise ValueError("Reference dimension must match state dimension")
        self.reference = reference.copy()
    
    def enable_integral_control(self, gain: float) -> None:
        """Enable integral control with specified gain."""
        self.enable_integral = True
        self.integral_gain = gain
        self.integral_error = np.zeros_like(self.reference)
    
    def disable_integral_control(self) -> None:
        """Disable integral control."""
        self.enable_integral = False
        self.integral_error = np.zeros_like(self.reference)
    
    def reset(self) -> None:
        """Reset controller state."""
        self.integral_error = np.zeros_like(self.reference)
    
    def get_stability_margins(self) -> Dict[str, float]:
        """Compute stability margins."""
        try:
            # Closed-loop system matrix
            A_cl = self.A - self.B @ self.K
            
            # Eigenvalues
            eigenvals = np.linalg.eigvals(A_cl)
            
            # Stability margin (distance from unit circle)
            stability_margin = 1.0 - np.max(np.abs(eigenvals))
            
            # Phase margin (simplified)
            phase_margin = np.min(np.angle(eigenvals)) * 180 / np.pi
            
            return {
                "stability_margin": float(stability_margin),
                "phase_margin": float(phase_margin),
                "max_eigenvalue_magnitude": float(np.max(np.abs(eigenvals))),
                "is_stable": stability_margin > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to compute stability margins: {e}")
            return {
                "stability_margin": 0.0,
                "phase_margin": 0.0,
                "max_eigenvalue_magnitude": 1.0,
                "is_stable": False
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get controller performance metrics."""
        return {
            "gain_matrix_norm": np.linalg.norm(self.K),
            "riccati_solution_norm": np.linalg.norm(self.P),
            "condition_number": np.linalg.cond(self.P),
            "stability_margins": self.get_stability_margins()
        }


class AdaptiveLQRController(LQRController):
    """Adaptive LQR controller with online parameter estimation."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, gains: LQRGains, 
                 adaptation_rate: float = 0.01):
        """Initialize adaptive LQR controller."""
        super().__init__(A, B, gains)
        
        self.adaptation_rate = adaptation_rate
        self.A_estimated = A.copy()
        self.B_estimated = B.copy()
        
        # Parameter estimation
        self.estimation_buffer_size = 100
        self.state_history = []
        self.control_history = []
        
    def update(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Update with parameter adaptation."""
        # Store history for parameter estimation
        if len(self.state_history) >= self.estimation_buffer_size:
            self.state_history.pop(0)
            self.control_history.pop(0)
        
        # Get control command
        control = super().update(state, dt)
        
        # Store current state and control
        self.state_history.append(state.copy())
        self.control_history.append(control.copy())
        
        # Update parameter estimates
        if len(self.state_history) > 10:  # Need sufficient data
            self._update_parameter_estimates()
        
        return control
    
    def _update_parameter_estimates(self) -> None:
        """Update system parameter estimates using least squares."""
        try:
            if len(self.state_history) < 2:
                return
            
            # Prepare data for least squares estimation
            X = np.array(self.state_history[:-1])  # x(k)
            U = np.array(self.control_history[:-1])  # u(k)
            Y = np.array(self.state_history[1:])     # x(k+1)
            
            # Combine state and control
            Z = np.hstack([X, U])  # [x(k), u(k)]
            
            # Least squares estimation: Y = Z * [A B]^T
            if Z.shape[0] > Z.shape[1]:  # Overdetermined system
                theta = np.linalg.lstsq(Z, Y, rcond=None)[0]
                
                n_states = self.A.shape[0]
                
                # Extract estimated A and B matrices
                A_new = theta[:n_states, :].T
                B_new = theta[n_states:, :].T
                
                # Adaptive update
                alpha = self.adaptation_rate
                self.A_estimated = (1 - alpha) * self.A_estimated + alpha * A_new
                self.B_estimated = (1 - alpha) * self.B_estimated + alpha * B_new
                
                # Re-solve LQR with updated parameters
                old_A, old_B = self.A, self.B
                self.A, self.B = self.A_estimated, self.B_estimated
                self.K, self.P = self._solve_lqr()
                self.A, self.B = old_A, old_B  # Restore original for consistency
                
        except Exception as e:
            logger.warning(f"Parameter estimation failed: {e}")


def design_lqr_gains(A: np.ndarray, B: np.ndarray, 
                    Q_weights: np.ndarray, R_weights: np.ndarray) -> LQRGains:
    """Design LQR gains using Bryson's rule."""
    n_states = A.shape[0]
    n_controls = B.shape[1]
    
    # Bryson's rule: Q_ii = 1/max_acceptable_xi^2, R_jj = 1/max_acceptable_uj^2
    Q = np.diag(1.0 / (Q_weights ** 2))
    R = np.diag(1.0 / (R_weights ** 2))
    
    return LQRGains(Q=Q, R=R)


def tune_lqr_weights(A: np.ndarray, B: np.ndarray, 
                    performance_specs: Dict[str, float]) -> LQRGains:
    """Automatically tune LQR weights based on performance specifications."""
    n_states = A.shape[0]
    n_controls = B.shape[1]
    
    # Default weights
    Q_weights = np.ones(n_states)
    R_weights = np.ones(n_controls)
    
    # Adjust based on specifications
    if "settling_time" in performance_specs:
        # Faster settling requires higher Q weights
        settling_time = performance_specs["settling_time"]
        Q_weights *= 1.0 / settling_time
    
    if "control_effort" in performance_specs:
        # Lower control effort requires higher R weights
        control_effort = performance_specs["control_effort"]
        R_weights *= control_effort
    
    if "overshoot" in performance_specs:
        # Lower overshoot requires more balanced Q and R
        overshoot = performance_specs["overshoot"]
        if overshoot < 0.1:  # Less than 10% overshoot
            R_weights *= 2.0
    
    return design_lqr_gains(A, B, Q_weights, R_weights) 