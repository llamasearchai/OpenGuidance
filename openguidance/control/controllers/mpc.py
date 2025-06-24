"""
Model Predictive Control (MPC) controller implementation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class MPCSolver(Enum):
    """Available MPC solvers."""
    QUADPROG = auto()
    OSQP = auto()
    CVXPY = auto()


@dataclass
class MPCConfig:
    """MPC controller configuration."""
    prediction_horizon: int = 10
    control_horizon: int = 5
    state_weights: Optional[np.ndarray] = None
    control_weights: Optional[np.ndarray] = None
    terminal_weights: Optional[np.ndarray] = None
    solver: MPCSolver = MPCSolver.QUADPROG
    max_iterations: int = 100
    tolerance: float = 1e-6


class MPCController:
    """Model Predictive Control controller."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, config: MPCConfig):
        """Initialize MPC controller.
        
        Args:
            A: State transition matrix
            B: Control input matrix
            config: MPC configuration
        """
        self.A = A
        self.B = B
        self.config = config
        
        # System dimensions
        self.n_states = A.shape[0]
        self.n_controls = B.shape[1]
        
        # Validate dimensions
        if A.shape != (self.n_states, self.n_states):
            raise ValueError("A matrix must be square")
        if B.shape[0] != self.n_states:
            raise ValueError("B matrix rows must match A matrix size")
        
        # Initialize weights
        self._initialize_weights()
        
        # Constraint matrices
        self.state_constraints = None
        self.control_constraints = None
        self.terminal_constraints = None
        
        # Reference trajectory
        self.reference_trajectory = np.zeros((config.prediction_horizon, self.n_states))
        
        logger.info(f"MPC controller initialized: {self.n_states} states, {self.n_controls} controls")
    
    def _initialize_weights(self) -> None:
        """Initialize cost function weights."""
        if self.config.state_weights is None:
            self.config.state_weights = np.eye(self.n_states)
        
        if self.config.control_weights is None:
            self.config.control_weights = np.eye(self.n_controls)
        
        if self.config.terminal_weights is None:
            self.config.terminal_weights = self.config.state_weights * 10
    
    def update(self, current_state: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute MPC control action.
        
        Args:
            current_state: Current system state
            reference: Reference trajectory (optional)
            
        Returns:
            Optimal control action
        """
        if reference is not None:
            self.set_reference_trajectory(reference)
        
        # Solve MPC optimization problem
        try:
            control_sequence = self._solve_mpc(current_state)
            return control_sequence[0]  # Return first control action
        except Exception as e:
            logger.error(f"MPC optimization failed: {e}")
            # Return zero control as fallback
            return np.zeros(self.n_controls)
    
    def _solve_mpc(self, current_state: np.ndarray) -> np.ndarray:
        """Solve MPC optimization problem."""
        N = self.config.prediction_horizon
        M = self.config.control_horizon
        
        # Build prediction matrices
        Phi, Gamma = self._build_prediction_matrices()
        
        # Build cost matrices
        Q_bar, R_bar = self._build_cost_matrices()
        
        # Predicted states: X = Phi * x0 + Gamma * U
        # Cost: J = (X - X_ref)^T * Q_bar * (X - X_ref) + U^T * R_bar * U
        
        # Reference trajectory over prediction horizon
        X_ref = self.reference_trajectory[:N].flatten()
        
        # Quadratic cost: J = 0.5 * U^T * H * U + f^T * U
        H = 2 * (Gamma.T @ Q_bar @ Gamma + R_bar)
        f = 2 * Gamma.T @ Q_bar @ (Phi @ current_state - X_ref)
        
        # Solve QP problem
        if self.config.solver == MPCSolver.QUADPROG:
            U_opt = self._solve_quadprog(H, f, M)
        else:
            # Fallback to simple solution
            U_opt = self._solve_unconstrained(H, f, M)
        
        return U_opt.reshape(M, self.n_controls)
    
    def _build_prediction_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build prediction matrices Phi and Gamma."""
        N = self.config.prediction_horizon
        
        # Phi matrix (state prediction)
        Phi = np.zeros((N * self.n_states, self.n_states))
        A_power = np.eye(self.n_states)
        
        for i in range(N):
            start_idx = i * self.n_states
            end_idx = (i + 1) * self.n_states
            Phi[start_idx:end_idx, :] = A_power
            A_power = A_power @ self.A
        
        # Gamma matrix (control prediction)
        M = self.config.control_horizon
        Gamma = np.zeros((N * self.n_states, M * self.n_controls))
        
        for i in range(N):
            for j in range(min(i + 1, M)):
                row_start = i * self.n_states
                row_end = (i + 1) * self.n_states
                col_start = j * self.n_controls
                col_end = (j + 1) * self.n_controls
                
                # Gamma[i,j] = A^(i-j) * B
                A_power = np.linalg.matrix_power(self.A, i - j)
                Gamma[row_start:row_end, col_start:col_end] = A_power @ self.B
        
        return Phi, Gamma
    
    def _build_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build cost matrices Q_bar and R_bar."""
        N = self.config.prediction_horizon
        M = self.config.control_horizon
        
        # Q_bar (state cost matrix)
        Q_bar = np.zeros((N * self.n_states, N * self.n_states))
        for i in range(N - 1):
            start_idx = i * self.n_states
            end_idx = (i + 1) * self.n_states
            Q_bar[start_idx:end_idx, start_idx:end_idx] = self.config.state_weights
        
        # Terminal cost
        start_idx = (N - 1) * self.n_states
        end_idx = N * self.n_states
        Q_bar[start_idx:end_idx, start_idx:end_idx] = self.config.terminal_weights
        
        # R_bar (control cost matrix)
        R_bar = np.zeros((M * self.n_controls, M * self.n_controls))
        for i in range(M):
            start_idx = i * self.n_controls
            end_idx = (i + 1) * self.n_controls
            R_bar[start_idx:end_idx, start_idx:end_idx] = self.config.control_weights
        
        return Q_bar, R_bar
    
    def _solve_quadprog(self, H: np.ndarray, f: np.ndarray, M: int) -> np.ndarray:
        """Solve QP using simple method (placeholder for actual QP solver)."""
        try:
            # Simple unconstrained solution: H * U = -f
            U_opt = np.linalg.solve(H, -f)
            return U_opt
        except np.linalg.LinAlgError:
            logger.warning("QP solver failed, using pseudo-inverse")
            U_opt = np.linalg.pinv(H) @ (-f)
            return U_opt
    
    def _solve_unconstrained(self, H: np.ndarray, f: np.ndarray, M: int) -> np.ndarray:
        """Solve unconstrained QP problem."""
        try:
            U_opt = np.linalg.solve(H, -f)
            return U_opt
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback
            U_opt = np.linalg.pinv(H) @ (-f)
            return U_opt
    
    def set_reference_trajectory(self, reference: np.ndarray) -> None:
        """Set reference trajectory."""
        if reference.ndim == 1:
            # Single reference point - replicate over horizon
            self.reference_trajectory = np.tile(reference, (self.config.prediction_horizon, 1))
        else:
            # Full trajectory provided
            horizon_length = min(reference.shape[0], self.config.prediction_horizon)
            self.reference_trajectory[:horizon_length] = reference[:horizon_length]
            
            # Extend last point if needed
            if reference.shape[0] < self.config.prediction_horizon:
                last_point = reference[-1]
                for i in range(reference.shape[0], self.config.prediction_horizon):
                    self.reference_trajectory[i] = last_point
    
    def set_state_constraints(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        """Set state constraints."""
        self.state_constraints = {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
    
    def set_control_constraints(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        """Set control constraints."""
        self.control_constraints = {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
    
    def get_predicted_trajectory(self, current_state: np.ndarray) -> np.ndarray:
        """Get predicted state trajectory."""
        try:
            control_sequence = self._solve_mpc(current_state)
            
            # Simulate forward
            trajectory = np.zeros((self.config.prediction_horizon, self.n_states))
            state = current_state.copy()
            
            for i in range(self.config.prediction_horizon):
                trajectory[i] = state
                if i < self.config.control_horizon:
                    control = control_sequence[i]
                else:
                    control = control_sequence[-1]  # Hold last control
                
                state = self.A @ state + self.B @ control
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Trajectory prediction failed: {e}")
            return np.zeros((self.config.prediction_horizon, self.n_states))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get controller performance metrics."""
        return {
            "prediction_horizon": self.config.prediction_horizon,
            "control_horizon": self.config.control_horizon,
            "state_dimension": self.n_states,
            "control_dimension": self.n_controls,
            "solver": self.config.solver.name,
            "has_constraints": (
                self.state_constraints is not None or 
                self.control_constraints is not None
            )
        }


def create_mpc_controller(
    A: np.ndarray, 
    B: np.ndarray,
    prediction_horizon: int = 10,
    control_horizon: int = 5,
    state_weights: Optional[np.ndarray] = None,
    control_weights: Optional[np.ndarray] = None
) -> MPCController:
    """Create MPC controller with default configuration."""
    config = MPCConfig(
        prediction_horizon=prediction_horizon,
        control_horizon=control_horizon,
        state_weights=state_weights,
        control_weights=control_weights
    )
    return MPCController(A, B, config) 