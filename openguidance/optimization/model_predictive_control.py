"""Model Predictive Control implementation for OpenGuidance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from openguidance.core.types import State, Control, Vehicle

logger = logging.getLogger(__name__)


class MPCObjective(Enum):
    """MPC objective function types."""
    TRACKING = auto()
    REGULATION = auto()
    ECONOMIC = auto()


@dataclass
class MPCConfig:
    """Configuration for Model Predictive Controller."""
    # Prediction horizon
    prediction_horizon: int = 10
    control_horizon: int = 10
    
    # Sampling time
    sampling_time: float = 0.1
    
    # Objective function
    objective_type: MPCObjective = MPCObjective.TRACKING
    
    # Weights
    state_weights: np.ndarray = field(default_factory=lambda: np.ones(12))
    control_weights: np.ndarray = field(default_factory=lambda: np.ones(4))
    terminal_weights: np.ndarray = field(default_factory=lambda: np.ones(12))
    
    # Constraints
    state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    control_rate_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Solver settings
    solver_type: str = "quadprog"  # "quadprog", "cvxpy", "casadi"
    max_iterations: int = 100
    tolerance: float = 1e-6
    
    # Warm start
    enable_warm_start: bool = True


class ModelPredictiveController:
    """Model Predictive Controller for trajectory tracking and regulation."""
    
    def __init__(self, config: MPCConfig, vehicle: Vehicle):
        self.config = config
        self.vehicle = vehicle
        
        # MPC matrices
        self.A = None  # State transition matrix
        self.B = None  # Control input matrix
        self.C = None  # Output matrix
        
        # Prediction matrices
        self.Phi = None  # State prediction matrix
        self.Gamma = None  # Control prediction matrix
        
        # QP matrices
        self.H = None  # Hessian matrix
        self.f = None  # Linear term
        self.A_ineq = None  # Inequality constraint matrix
        self.b_ineq = None  # Inequality constraint vector
        
        # Previous solution for warm start
        self.previous_solution = None
        
        # Statistics
        self.solve_count = 0
        self.solve_times = []
        
        logger.info(f"MPC controller initialized with horizon {config.prediction_horizon}")
    
    def set_system_matrices(self, A: np.ndarray, B: np.ndarray, C: Optional[np.ndarray] = None) -> None:
        """Set system matrices for MPC formulation."""
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy() if C is not None else np.eye(A.shape[0])
        
        # Build prediction matrices
        self._build_prediction_matrices()
        
        logger.info(f"System matrices set: A {A.shape}, B {B.shape}, C {self.C.shape}")
    
    def _build_prediction_matrices(self) -> None:
        """Build prediction matrices for MPC formulation."""
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("System matrices must be set before building prediction matrices")
        
        n_states = self.A.shape[0]
        n_controls = self.B.shape[1]
        n_outputs = self.C.shape[0]
        N = self.config.prediction_horizon
        
        # State prediction matrix Phi
        self.Phi = np.zeros((N * n_outputs, n_states))
        A_power = np.eye(n_states)
        
        for i in range(N):
            A_power = A_power @ self.A
            self.Phi[i * n_outputs:(i + 1) * n_outputs, :] = self.C @ A_power
        
        # Control prediction matrix Gamma
        self.Gamma = np.zeros((N * n_outputs, N * n_controls))
        
        for i in range(N):
            A_power = np.eye(n_states)
            for j in range(i + 1):
                if j == 0:
                    CB = self.C @ self.B
                else:
                    A_power = A_power @ self.A
                    CB = self.C @ A_power @ self.B
                
                row_start = i * n_outputs
                row_end = (i + 1) * n_outputs
                col_start = (i - j) * n_controls
                col_end = (i - j + 1) * n_controls
                
                if col_start >= 0:
                    self.Gamma[row_start:row_end, col_start:col_end] = CB
    
    def _build_qp_matrices(self, current_state: np.ndarray, reference: np.ndarray) -> None:
        """Build QP matrices for current optimization problem."""
        if self.B is None:
            raise ValueError("System matrices must be set before building QP matrices")
        
        N = self.config.prediction_horizon
        n_controls = self.B.shape[1]
        
        # Weight matrices
        Q = np.kron(np.eye(N), np.diag(self.config.state_weights))
        R = np.kron(np.eye(N), np.diag(self.config.control_weights))
        
        # Terminal weight
        if self.config.terminal_weights is not None:
            Q[-len(self.config.terminal_weights):, -len(self.config.terminal_weights):] = np.diag(self.config.terminal_weights)
        
        # Hessian matrix
        self.H = self.Gamma.T @ Q @ self.Gamma + R
        
        # Linear term
        prediction_error = self.Phi @ current_state - reference
        self.f = self.Gamma.T @ Q @ prediction_error
        
        # Ensure H is positive definite
        eigenvals = np.linalg.eigvals(self.H)
        if np.min(eigenvals) <= 1e-12:
            self.H += np.eye(self.H.shape[0]) * 1e-6
    
    def _build_constraints(self) -> None:
        """Build constraint matrices for QP problem."""
        if self.B is None:
            raise ValueError("System matrices must be set before building constraints")
        
        N = self.config.prediction_horizon
        n_controls = self.B.shape[1]
        
        constraints = []
        bounds = []
        
        # Control bounds
        if self.config.control_bounds is not None:
            u_min, u_max = self.config.control_bounds
            
            # Lower bounds: -u >= -u_max
            A_u_max = -np.kron(np.eye(N), np.eye(n_controls))
            b_u_max = -np.tile(u_max, N)
            constraints.append(A_u_max)
            bounds.append(b_u_max)
            
            # Upper bounds: u >= u_min
            A_u_min = np.kron(np.eye(N), np.eye(n_controls))
            b_u_min = np.tile(u_min, N)
            constraints.append(A_u_min)
            bounds.append(b_u_min)
        
        # Control rate bounds
        if self.config.control_rate_bounds is not None:
            du_min, du_max = self.config.control_rate_bounds
            
            # Build difference matrix
            D = np.kron(np.eye(N), np.eye(n_controls))
            D[n_controls:, :-n_controls] -= np.kron(np.eye(N-1), np.eye(n_controls))
            
            # Rate constraints
            A_du_max = -D
            b_du_max = -np.tile(du_max, N)
            constraints.append(A_du_max)
            bounds.append(b_du_max)
            
            A_du_min = D
            b_du_min = np.tile(du_min, N)
            constraints.append(A_du_min)
            bounds.append(b_du_min)
        
        # Combine constraints
        if constraints:
            self.A_ineq = np.vstack(constraints)
            self.b_ineq = np.concatenate(bounds)
        else:
            self.A_ineq = None
            self.b_ineq = None
    
    def solve(self, current_state: State, reference_trajectory: List[State]) -> Control:
        """Solve MPC optimization problem."""
        import time
        start_time = time.time()
        
        # Convert state to vector
        current_state_vec = self._state_to_vector(current_state)
        
        # Convert reference trajectory to vector
        reference_vec = self._reference_to_vector(reference_trajectory)
        
        # Build QP matrices
        self._build_qp_matrices(current_state_vec, reference_vec)
        self._build_constraints()
        
        # Solve QP problem
        try:
            if self.config.solver_type == "quadprog":
                solution = self._solve_quadprog()
            elif self.config.solver_type == "cvxpy":
                solution = self._solve_cvxpy()
            else:
                solution = self._solve_simple()
                
        except Exception as e:
            logger.error(f"MPC solver failed: {e}")
            # Return zero control as fallback
            solution = np.zeros(self.B.shape[1] * self.config.prediction_horizon)
        
        # Extract first control action
        if self.B is None:
            raise ValueError("System matrices must be set")
        control_vec = solution[:self.B.shape[1]]
        control = self._vector_to_control(control_vec)
        
        # Store solution for warm start
        if self.config.enable_warm_start:
            self.previous_solution = solution
        
        # Update statistics
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)
        self.solve_count += 1
        
        logger.debug(f"MPC solved in {solve_time:.4f}s")
        
        return control
    
    def _solve_simple(self) -> np.ndarray:
        """Simple unconstrained QP solver."""
        if self.H is None or self.f is None:
            raise ValueError("QP matrices must be built before solving")
        
        try:
            solution = -np.linalg.solve(self.H, self.f)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            solution = -np.linalg.pinv(self.H) @ self.f
        
        return solution
    
    def _solve_quadprog(self) -> np.ndarray:
        """Solve using quadprog (if available)."""
        try:
            import quadprog
            
            if self.A_ineq is not None and self.b_ineq is not None:
                solution = quadprog.solve_qp(self.H, self.f, -self.A_ineq.T, -self.b_ineq)[0]
            else:
                if self.H is None or self.f is None:
                    raise ValueError("QP matrices must be built")
                solution = -np.linalg.solve(self.H, self.f)
            
            return solution
        except ImportError:
            logger.warning("quadprog not available, falling back to simple solver")
            return self._solve_simple()
    
    def _solve_cvxpy(self) -> np.ndarray:
        """Solve using CVXPY (if available)."""
        try:
            import cvxpy as cp
            
            if self.H is None:
                raise ValueError("QP matrices must be built")
            n_vars = self.H.shape[0]
            u = cp.Variable(n_vars)
            
            objective = cp.Minimize(0.5 * cp.quad_form(u, self.H) + self.f.T @ u)
            
            constraints = []
            if self.A_ineq is not None and self.b_ineq is not None:
                constraints.append(self.A_ineq @ u <= self.b_ineq)
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value
            else:
                logger.warning(f"CVXPY solver status: {problem.status}")
                return self._solve_simple()
                
        except ImportError:
            logger.warning("CVXPY not available, falling back to simple solver")
            return self._solve_simple()
    
    def _state_to_vector(self, state: State) -> np.ndarray:
        """Convert State object to vector."""
        vector = np.zeros(12)
        
        if hasattr(state, 'position') and state.position is not None:
            vector[0:3] = state.position
        if hasattr(state, 'velocity') and state.velocity is not None:
            vector[3:6] = state.velocity
        if hasattr(state, 'attitude') and state.attitude is not None:
            # Convert to Euler angles
            euler = state.attitude.yaw_pitch_roll
            vector[6:9] = euler
        if hasattr(state, 'angular_velocity') and state.angular_velocity is not None:
            vector[9:12] = state.angular_velocity
        
        return vector
    
    def _reference_to_vector(self, reference_trajectory: List[State]) -> np.ndarray:
        """Convert reference trajectory to vector."""
        N = self.config.prediction_horizon
        n_states = 12  # Assuming 12-state system
        
        reference_vec = np.zeros(N * n_states)
        
        for i in range(min(N, len(reference_trajectory))):
            state_vec = self._state_to_vector(reference_trajectory[i])
            reference_vec[i * n_states:(i + 1) * n_states] = state_vec
        
        # Extend last reference if trajectory is shorter than horizon
        if len(reference_trajectory) < N:
            last_state_vec = self._state_to_vector(reference_trajectory[-1])
            for i in range(len(reference_trajectory), N):
                reference_vec[i * n_states:(i + 1) * n_states] = last_state_vec
        
        return reference_vec
    
    def _vector_to_control(self, control_vec: np.ndarray) -> Control:
        """Convert control vector to Control object."""
        control = Control()
        
        if len(control_vec) >= 1:
            control.throttle = float(control_vec[0])
        if len(control_vec) >= 2:
            control.aileron = float(control_vec[1])
        if len(control_vec) >= 3:
            control.elevator = float(control_vec[2])
        if len(control_vec) >= 4:
            control.rudder = float(control_vec[3])
        
        return control
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get MPC diagnostic information."""
        diagnostics = {
            "solve_count": self.solve_count,
            "average_solve_time": float(np.mean(self.solve_times)) if self.solve_times else 0.0,
            "max_solve_time": float(np.max(self.solve_times)) if self.solve_times else 0.0,
            "prediction_horizon": self.config.prediction_horizon,
            "control_horizon": self.config.control_horizon,
            "solver_type": self.config.solver_type,
        }
        
        if self.H is not None:
            diagnostics["hessian_condition_number"] = float(np.linalg.cond(self.H))
        
        return diagnostics
    
    def reset(self) -> None:
        """Reset MPC controller state."""
        self.previous_solution = None
        self.solve_count = 0
        self.solve_times = []
        logger.info("MPC controller reset") 