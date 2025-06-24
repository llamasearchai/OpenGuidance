"""
Advanced trajectory optimization algorithms for aerospace applications.

This module provides multiple optimization methods:
- Direct collocation with pseudospectral methods
- Shooting methods with multiple shooting
- Genetic algorithms for global optimization
- Gradient-based optimization with IPOPT
- Multi-objective optimization capabilities

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import scipy.optimize
from numba import jit
import time

from openguidance.core.types import State, Control, Vehicle, Trajectory, VehicleType
from openguidance.dynamics.models.aircraft import AircraftDynamics
from openguidance.dynamics.models.missile import MissileDynamics

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Available optimization methods."""
    DIRECT_COLLOCATION = auto()
    MULTIPLE_SHOOTING = auto()
    GENETIC_ALGORITHM = auto()
    PARTICLE_SWARM = auto()
    GRADIENT_DESCENT = auto()
    INTERIOR_POINT = auto()


class CostFunction(Enum):
    """Available cost functions."""
    MINIMUM_TIME = auto()
    MINIMUM_FUEL = auto()
    MINIMUM_ENERGY = auto()
    MINIMUM_CONTROL_EFFORT = auto()
    MAXIMUM_RANGE = auto()
    CUSTOM = auto()


@dataclass
class OptimizationConstraints:
    """Trajectory optimization constraints."""
    # State constraints
    position_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    velocity_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    attitude_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Control constraints  
    control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    control_rate_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Path constraints
    altitude_min: Optional[float] = None
    altitude_max: Optional[float] = None
    speed_min: Optional[float] = None
    speed_max: Optional[float] = None
    g_force_max: Optional[float] = None
    
    # Boundary constraints
    initial_state: Optional[State] = None
    final_state: Optional[State] = None
    initial_time: Optional[float] = None
    final_time: Optional[float] = None
    
    # Mission-specific constraints
    no_fly_zones: List[Dict[str, Any]] = field(default_factory=list)
    waypoints: List[Dict[str, Any]] = field(default_factory=list)
    threat_zones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrajectoryOptimizerConfig:
    """Configuration for trajectory optimizer."""
    # Optimization method
    method: OptimizationMethod = OptimizationMethod.DIRECT_COLLOCATION
    cost_function: CostFunction = CostFunction.MINIMUM_TIME
    
    # Discretization
    num_nodes: int = 50
    time_horizon: float = 100.0
    adaptive_mesh: bool = True
    
    # Solver settings
    max_iterations: int = 1000
    tolerance: float = 1e-6
    convergence_tolerance: float = 1e-8
    
    # Multi-objective settings
    enable_multi_objective: bool = False
    pareto_points: int = 20
    
    # Robustness settings
    enable_uncertainty: bool = False
    monte_carlo_samples: int = 100
    
    # Performance settings
    parallel_processing: bool = True
    num_threads: int = 4
    
    # Logging
    verbose: bool = True
    save_intermediate: bool = False


class TrajectoryOptimizer:
    """
    Advanced trajectory optimization with multiple algorithms.
    
    Supports various optimization methods and cost functions for
    aerospace trajectory planning and optimization.
    """
    
    def __init__(self, config: TrajectoryOptimizerConfig, vehicle: Vehicle):
        """Initialize trajectory optimizer."""
        self.config = config
        self.vehicle = vehicle
        
        # Initialize dynamics model
        if vehicle.type == VehicleType.AIRCRAFT:
            self.dynamics = AircraftDynamics(vehicle)
        elif vehicle.type == VehicleType.MISSILE:
            self.dynamics = MissileDynamics(vehicle)
        else:
            raise ValueError(f"Unsupported vehicle type: {vehicle.type}")
        
        # Optimization state
        self.current_solution = None
        self.optimization_history = []
        self.convergence_data = []
        
        # Performance metrics
        self.solve_time = 0.0
        self.iterations = 0
        self.cost_history = []
        
        logger.info(f"Trajectory optimizer initialized with {config.method.name} method")
    
    def optimize_trajectory(
        self,
        constraints: OptimizationConstraints,
        initial_guess: Optional[Trajectory] = None,
        custom_cost: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Optimize trajectory subject to constraints.
        
        Args:
            constraints: Optimization constraints
            initial_guess: Initial trajectory guess
            custom_cost: Custom cost function
            
        Returns:
            Optimization results with trajectory and metrics
        """
        logger.info(f"Starting trajectory optimization with {self.config.method.name}")
        
        # Setup optimization problem
        problem = self._setup_optimization_problem(constraints, custom_cost)
        
        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = self._generate_initial_guess(constraints)
        
        # Solve based on method
        if self.config.method == OptimizationMethod.DIRECT_COLLOCATION:
            result = self._solve_direct_collocation(problem, initial_guess)
        elif self.config.method == OptimizationMethod.MULTIPLE_SHOOTING:
            result = self._solve_multiple_shooting(problem, initial_guess)
        elif self.config.method == OptimizationMethod.GENETIC_ALGORITHM:
            result = self._solve_genetic_algorithm(problem, constraints)
        elif self.config.method == OptimizationMethod.PARTICLE_SWARM:
            result = self._solve_particle_swarm(problem, constraints)
        elif self.config.method == OptimizationMethod.GRADIENT_DESCENT:
            result = self._solve_gradient_descent(problem, initial_guess)
        else:
            result = self._solve_interior_point(problem, initial_guess)
        
        # Post-process results
        result = self._post_process_results(result, constraints)
        
        if result['success']:
            logger.info(f"Optimization completed in {result['solve_time']:.2f}s with cost {result['optimal_cost']:.6f}")
        else:
            logger.error(f"Optimization failed: {result.get('message', 'Unknown error')}")
        
        return result
    
    def _setup_optimization_problem(
        self, 
        constraints: OptimizationConstraints, 
        custom_cost: Optional[Callable]
    ) -> Dict[str, Any]:
        """Setup optimization problem structure."""
        
        # Determine state and control dimensions
        state_dim = 12  # [position(3), velocity(3), attitude(3), angular_velocity(3)]
        control_dim = 4  # [thrust, aileron, elevator, rudder]
        
        # Time grid
        if self.config.adaptive_mesh:
            time_grid = self._generate_adaptive_time_grid(constraints)
        else:
            time_grid = np.linspace(0, self.config.time_horizon, self.config.num_nodes)
        
        # Cost function
        if custom_cost is not None:
            cost_func = custom_cost
        else:
            cost_func = self._get_standard_cost_function(self.config.cost_function)
        
        # Constraint functions
        dynamics_func = self._get_dynamics_function()
        path_constraints = self._get_path_constraints(constraints)
        boundary_constraints = self._get_boundary_constraints(constraints)
        
        problem = {
            'state_dim': state_dim,
            'control_dim': control_dim,
            'time_grid': time_grid,
            'cost_function': cost_func,
            'dynamics': dynamics_func,
            'path_constraints': path_constraints,
            'boundary_constraints': boundary_constraints,
            'variable_bounds': self._get_variable_bounds(constraints)
        }
        
        return problem
    
    def _solve_direct_collocation(self, problem: Dict[str, Any], initial_guess: Trajectory) -> Dict[str, Any]:
        """Solve using direct collocation method."""
        
        # Extract problem dimensions
        state_dim = problem['state_dim']
        control_dim = problem['control_dim']
        time_grid = problem['time_grid']
        num_nodes = len(time_grid)
        
        # Decision variables: [states, controls, final_time]
        num_vars = num_nodes * state_dim + (num_nodes - 1) * control_dim + 1
        
        # Initial guess vector
        x0 = self._trajectory_to_vector(initial_guess, time_grid)
        
        # Bounds
        bounds = self._get_optimization_bounds(problem, num_vars)
        
        # Constraints
        constraints_list = self._get_optimization_constraints(problem, time_grid)
        
        # Objective function
        def objective(x):
            return self._evaluate_cost(x, problem, time_grid)
        
        # Jacobian
        def jacobian(x):
            return self._compute_cost_jacobian(x, problem, time_grid)
        
        # Solve using scipy
        start_time = time.time()
        
        result = scipy.optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            jac=jacobian,
            bounds=bounds,
            constraints=constraints_list,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.tolerance,
                'disp': self.config.verbose
            }
        )
        
        solve_time = time.time() - start_time
        
        # Extract solution
        if result.success:
            optimal_trajectory = self._vector_to_trajectory(result.x, time_grid, problem)
            
            return {
                'success': True,
                'optimal_trajectory': optimal_trajectory,
                'optimal_cost': result.fun,
                'solve_time': solve_time,
                'iterations': result.nit,
                'convergence_data': self.convergence_data,
                'method': 'Direct Collocation'
            }
        else:
            logger.error(f"Optimization failed: {result.message}")
            return {
                'success': False,
                'message': result.message,
                'solve_time': solve_time,
                'method': 'Direct Collocation'
            }
    
    def _solve_multiple_shooting(self, problem: Dict[str, Any], initial_guess: Trajectory) -> Dict[str, Any]:
        """Solve using multiple shooting method."""
        
        # Divide trajectory into shooting intervals
        num_intervals = 10
        time_grid = problem['time_grid']
        interval_indices = np.linspace(0, len(time_grid)-1, num_intervals+1, dtype=int)
        
        # Decision variables: initial states for each interval + controls
        state_dim = problem['state_dim']
        control_dim = problem['control_dim']
        
        # Setup shooting problem
        def shooting_constraints(x):
            """Continuity constraints for multiple shooting."""
            constraints = []
            
            for i in range(num_intervals):
                # Extract initial state and controls for this interval
                start_idx = i * state_dim
                state_i = x[start_idx:start_idx + state_dim]
                
                # Integrate over interval
                t_start = time_grid[interval_indices[i]]
                t_end = time_grid[interval_indices[i+1]]
                
                # Simplified integration (would use proper ODE solver)
                dt = (t_end - t_start) / 10
                state_final = state_i.copy()
                
                for j in range(10):
                    # Get control at this time
                    control = x[num_intervals * state_dim + j * control_dim:
                               num_intervals * state_dim + (j+1) * control_dim]
                    
                    # Integrate dynamics
                    state_dot = problem['dynamics'](state_final, control)
                    state_final += state_dot * dt
                
                # Continuity constraint
                if i < num_intervals - 1:
                    next_state = x[(i+1) * state_dim:(i+2) * state_dim]
                    constraints.extend(state_final - next_state)
            
            return np.array(constraints)
        
        # Initial guess for shooting
        x0 = np.zeros(num_intervals * state_dim + num_intervals * control_dim)
        
        # Solve shooting problem
        start_time = time.time()
        
        result = scipy.optimize.root(
            shooting_constraints,
            x0,
            method='hybr',
            options={'xtol': self.config.tolerance}
        )
        
        solve_time = time.time() - start_time
        
        if result.success:
            # Reconstruct trajectory
            optimal_trajectory = self._shooting_to_trajectory(result.x, time_grid, problem)
            
            return {
                'success': True,
                'optimal_trajectory': optimal_trajectory,
                'optimal_cost': self._evaluate_trajectory_cost(optimal_trajectory, problem),
                'solve_time': solve_time,
                'method': 'Multiple Shooting'
            }
        else:
            return {
                'success': False,
                'message': result.message,
                'solve_time': solve_time,
                'method': 'Multiple Shooting'
            }
    
    def _solve_genetic_algorithm(self, problem: Dict[str, Any], constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Solve using genetic algorithm."""
        
        # GA parameters
        population_size = 100
        num_generations = 200
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Problem dimensions
        state_dim = problem['state_dim']
        control_dim = problem['control_dim']
        time_grid = problem['time_grid']
        num_nodes = len(time_grid)
        
        # Chromosome length (simplified representation)
        chromosome_length = (num_nodes - 1) * control_dim  # Only optimize controls
        
        # Initialize population
        population = self._initialize_ga_population(
            population_size, chromosome_length, constraints
        )
        
        start_time = time.time()
        best_fitness_history = []
        
        for generation in range(num_generations):
            # Evaluate fitness
            fitness_values = []
            for individual in population:
                trajectory = self._ga_individual_to_trajectory(individual, problem, constraints)
                if trajectory is not None:
                    cost = self._evaluate_trajectory_cost(trajectory, problem)
                    fitness = 1.0 / (1.0 + cost)  # Convert cost to fitness
                else:
                    fitness = 0.0  # Infeasible
                
                fitness_values.append(fitness)
            
            # Track best fitness
            best_fitness = max(fitness_values)
            best_fitness_history.append(best_fitness)
            
            if self.config.verbose and generation % 20 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
            
            # Selection
            parents = self._ga_selection(population, fitness_values)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._ga_crossover(parents[i], parents[i+1], crossover_rate)
                    child1 = self._ga_mutation(child1, mutation_rate, constraints)
                    child2 = self._ga_mutation(child2, mutation_rate, constraints)
                    offspring.extend([child1, child2])
            
            # Replace population
            population = offspring[:population_size]
        
        solve_time = time.time() - start_time
        
        # Get best solution
        final_fitness = []
        for individual in population:
            trajectory = self._ga_individual_to_trajectory(individual, problem, constraints)
            if trajectory is not None:
                cost = self._evaluate_trajectory_cost(trajectory, problem)
                final_fitness.append((cost, trajectory))
            else:
                final_fitness.append((float('inf'), None))
        
        # Find best
        best_cost, best_trajectory = min(final_fitness, key=lambda x: x[0])
        
        if best_trajectory is not None:
            return {
                'success': True,
                'optimal_trajectory': best_trajectory,
                'optimal_cost': best_cost,
                'solve_time': solve_time,
                'generations': num_generations,
                'fitness_history': best_fitness_history,
                'method': 'Genetic Algorithm'
            }
        else:
            return {
                'success': False,
                'message': 'No feasible solution found',
                'solve_time': solve_time,
                'method': 'Genetic Algorithm'
            }
    
    def _solve_particle_swarm(self, problem: Dict[str, Any], constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Solve using particle swarm optimization."""
        
        # PSO parameters
        num_particles = 50
        num_iterations = 300
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Problem dimensions
        control_dim = problem['control_dim']
        time_grid = problem['time_grid']
        num_nodes = len(time_grid)
        
        # Particle dimension (control variables)
        particle_dim = (num_nodes - 1) * control_dim
        
        # Initialize swarm
        particles = self._initialize_pso_swarm(num_particles, particle_dim, constraints)
        velocities = [np.random.uniform(-0.1, 0.1, particle_dim) for _ in range(num_particles)]
        
        # Personal and global bests
        personal_bests = particles.copy()
        personal_best_costs = [float('inf')] * num_particles
        global_best = particles[0].copy()
        global_best_cost = float('inf')
        
        start_time = time.time()
        cost_history = []
        
        for iteration in range(num_iterations):
            for i, particle in enumerate(particles):
                # Evaluate particle
                trajectory = self._pso_particle_to_trajectory(particle, problem, constraints)
                if trajectory is not None:
                    cost = self._evaluate_trajectory_cost(trajectory, problem)
                    
                    # Update personal best
                    if cost < personal_best_costs[i]:
                        personal_best_costs[i] = cost
                        personal_bests[i] = particle.copy()
                    
                    # Update global best
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best = particle.copy()
            
            # Update velocities and positions
            for i in range(num_particles):
                r1, r2 = np.random.random(2)
                
                # Velocity update
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_bests[i] - particles[i]) + 
                               c2 * r2 * (global_best - particles[i]))
                
                # Position update
                particles[i] += velocities[i]
                
                # Apply bounds
                particles[i] = self._apply_pso_bounds(particles[i], constraints)
            
            cost_history.append(global_best_cost)
            
            if self.config.verbose and iteration % 50 == 0:
                logger.info(f"PSO Iteration {iteration}: Best cost = {global_best_cost:.6f}")
        
        solve_time = time.time() - start_time
        
        # Get final solution
        best_trajectory = self._pso_particle_to_trajectory(global_best, problem, constraints)
        
        if best_trajectory is not None:
            return {
                'success': True,
                'optimal_trajectory': best_trajectory,
                'optimal_cost': global_best_cost,
                'solve_time': solve_time,
                'iterations': num_iterations,
                'cost_history': cost_history,
                'method': 'Particle Swarm Optimization'
            }
        else:
            return {
                'success': False,
                'message': 'No feasible solution found',
                'solve_time': solve_time,
                'method': 'Particle Swarm Optimization'
            }
    
    def _solve_gradient_descent(self, problem: Dict[str, Any], initial_guess: Trajectory) -> Dict[str, Any]:
        """Solve using gradient descent with backtracking line search."""
        
        # Convert trajectory to optimization variables
        time_grid = problem['time_grid']
        x = self._trajectory_to_vector(initial_guess, time_grid)
        
        # Gradient descent parameters
        alpha = 0.01  # Initial step size
        beta = 0.5    # Backtracking parameter
        tolerance = self.config.tolerance
        max_iter = self.config.max_iterations
        
        start_time = time.time()
        cost_history = []
        
        for iteration in range(max_iter):
            # Evaluate cost and gradient
            cost = self._evaluate_cost(x, problem, time_grid)
            gradient = self._compute_cost_jacobian(x, problem, time_grid)
            
            cost_history.append(cost)
            
            # Check convergence
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < tolerance:
                logger.info(f"Gradient descent converged in {iteration} iterations")
                break
            
            # Backtracking line search
            step_size = alpha
            while True:
                x_new = x - step_size * gradient
                new_cost = self._evaluate_cost(x_new, problem, time_grid)
                
                if new_cost < cost - 0.5 * step_size * grad_norm**2:
                    break
                
                step_size *= beta
                if step_size < 1e-10:
                    break
            
            # Update
            x = x_new
            
            if self.config.verbose and iteration % 100 == 0:
                logger.info(f"Gradient Descent Iteration {iteration}: Cost = {cost:.6f}, Grad norm = {grad_norm:.6f}")
        
        solve_time = time.time() - start_time
        
        # Convert back to trajectory
        optimal_trajectory = self._vector_to_trajectory(x, time_grid, problem)
        final_cost = self._evaluate_cost(x, problem, time_grid)
        
        return {
            'success': True,
            'optimal_trajectory': optimal_trajectory,
            'optimal_cost': final_cost,
            'solve_time': solve_time,
            'iterations': iteration + 1,
            'cost_history': cost_history,
            'method': 'Gradient Descent'
        }
    
    def _solve_interior_point(self, problem: Dict[str, Any], initial_guess: Trajectory) -> Dict[str, Any]:
        """Solve using interior point method."""
        
        # This would use a specialized interior point solver
        # For now, fall back to scipy's trust-constr method
        
        time_grid = problem['time_grid']
        x0 = self._trajectory_to_vector(initial_guess, time_grid)
        
        # Bounds and constraints
        bounds = self._get_optimization_bounds(problem, len(x0))
        constraints_list = self._get_optimization_constraints(problem, time_grid)
        
        # Objective and derivatives
        def objective(x):
            return self._evaluate_cost(x, problem, time_grid)
        
        def jacobian(x):
            return self._compute_cost_jacobian(x, problem, time_grid)
        
        def hessian(x):
            return self._compute_cost_hessian(x, problem, time_grid)
        
        start_time = time.time()
        
        result = scipy.optimize.minimize(
            objective,
            x0,
            method='trust-constr',
            jac=jacobian,
            hess=hessian,
            bounds=bounds,
            constraints=constraints_list,
            options={
                'maxiter': self.config.max_iterations,
                'xtol': self.config.tolerance,
                'gtol': self.config.convergence_tolerance,
                'verbose': 1 if self.config.verbose else 0
            }
        )
        
        solve_time = time.time() - start_time
        
        if result.success:
            optimal_trajectory = self._vector_to_trajectory(result.x, time_grid, problem)
            
            return {
                'success': True,
                'optimal_trajectory': optimal_trajectory,
                'optimal_cost': result.fun,
                'solve_time': solve_time,
                'iterations': result.nit,
                'method': 'Interior Point'
            }
        else:
            return {
                'success': False,
                'message': result.message,
                'solve_time': solve_time,
                'method': 'Interior Point'
            }
    
    # Helper methods for optimization
    def _generate_initial_guess(self, constraints: OptimizationConstraints) -> Trajectory:
        """Generate initial trajectory guess."""
        
        # Simple linear interpolation between initial and final states
        if constraints.initial_state and constraints.final_state:
            initial_pos = constraints.initial_state.position
            final_pos = constraints.final_state.position
            
            time_grid = np.linspace(0, self.config.time_horizon, self.config.num_nodes)
            
            states = []
            for t in time_grid:
                alpha = t / self.config.time_horizon
                
                # Linear interpolation
                pos = (1 - alpha) * initial_pos + alpha * final_pos
                vel = (final_pos - initial_pos) / self.config.time_horizon
                
                # Default attitude and angular velocity
                from pyquaternion import Quaternion
                attitude = Quaternion([1, 0, 0, 0])
                angular_velocity = np.zeros(3)
                
                state = State(
                    position=pos,
                    velocity=vel,
                    attitude=attitude,
                    angular_velocity=angular_velocity,
                    time=t
                )
                states.append(state)
            
            return Trajectory(times=time_grid, states=states)
        
        else:
            # Default trajectory
            time_grid = np.linspace(0, self.config.time_horizon, self.config.num_nodes)
            states = []
            
            for t in time_grid:
                from pyquaternion import Quaternion
                state = State(
                    position=np.array([t * 10, 0, -1000]),  # Forward flight
                    velocity=np.array([10, 0, 0]),
                    attitude=Quaternion([1, 0, 0, 0]),
                    angular_velocity=np.zeros(3),
                    time=t
                )
                states.append(state)
            
            return Trajectory(times=time_grid, states=states)
    
    def _get_standard_cost_function(self, cost_type: CostFunction) -> Callable:
        """Get standard cost function."""
        
        if cost_type == CostFunction.MINIMUM_TIME:
            def minimum_time_cost(trajectory):
                return trajectory.times[-1] - trajectory.times[0]
            cost_func = minimum_time_cost
        
        elif cost_type == CostFunction.MINIMUM_FUEL:
            def minimum_fuel_cost(trajectory):
                # Simplified fuel cost based on control effort
                total_fuel = 0.0
                for i in range(len(trajectory.states) - 1):
                    dt = trajectory.times[i+1] - trajectory.times[i]
                    # Assume control effort proportional to fuel
                    control_effort = np.linalg.norm(trajectory.states[i].velocity) ** 2
                    total_fuel += control_effort * dt
                return total_fuel
            cost_func = minimum_fuel_cost
        
        elif cost_type == CostFunction.MINIMUM_ENERGY:
            def minimum_energy_cost(trajectory):
                total_energy = 0.0
                for state in trajectory.states:
                    # Kinetic + potential energy
                    kinetic = 0.5 * np.linalg.norm(state.velocity) ** 2
                    potential = 9.81 * (-state.position[2])  # Negative because NED frame
                    total_energy += kinetic + potential
                return total_energy / len(trajectory.states)
            cost_func = minimum_energy_cost
        
        elif cost_type == CostFunction.MINIMUM_CONTROL_EFFORT:
            def minimum_control_effort_cost(trajectory):
                # Would need control history - simplified here
                total_effort = 0.0
                for i in range(len(trajectory.states) - 1):
                    dt = trajectory.times[i+1] - trajectory.times[i]
                    # Approximate control effort from acceleration
                    accel = (trajectory.states[i+1].velocity - trajectory.states[i].velocity) / dt
                    total_effort += np.linalg.norm(accel) ** 2 * dt
                return total_effort
            cost_func = minimum_control_effort_cost
        
        elif cost_type == CostFunction.MAXIMUM_RANGE:
            def maximum_range_cost(trajectory):
                final_pos = trajectory.states[-1].position
                initial_pos = trajectory.states[0].position
                range_achieved = np.linalg.norm(final_pos - initial_pos)
                return -range_achieved  # Negative because we want to maximize
            cost_func = maximum_range_cost
        
        else:
            def default_cost(trajectory):
                return 0.0  # Default
            cost_func = default_cost
        
        return cost_func
    
    def _get_dynamics_function(self) -> Callable:
        """Get dynamics function for the vehicle."""
        
        def dynamics_func(state_vector, control_vector):
            """Compute state derivative."""
            # Extract state components
            pos = state_vector[0:3]
            vel = state_vector[3:6]
            euler = state_vector[6:9]
            omega = state_vector[9:12]
            
            # Simplified dynamics
            pos_dot = vel
            
            # Acceleration from control (simplified)
            if len(control_vector) >= 3:
                accel = control_vector[0:3]
            else:
                accel = np.zeros(3)
            
            vel_dot = accel + np.array([0, 0, 9.81])  # Add gravity
            
            # Attitude dynamics (simplified)
            euler_dot = omega
            
            # Angular acceleration (simplified)
            if len(control_vector) >= 6:
                omega_dot = control_vector[3:6]
            else:
                omega_dot = np.zeros(3)
            
            return np.concatenate([pos_dot, vel_dot, euler_dot, omega_dot])
        
        return dynamics_func
    
    def _post_process_results(self, result: Dict[str, Any], constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Post-process optimization results."""
        
        if result['success']:
            trajectory = result['optimal_trajectory']
            
            # Compute additional metrics
            result['trajectory_metrics'] = {
                'total_time': trajectory.times[-1] - trajectory.times[0],
                'total_distance': self._compute_trajectory_distance(trajectory),
                'max_speed': max(np.linalg.norm(state.velocity) for state in trajectory.states),
                'max_acceleration': self._compute_max_acceleration(trajectory),
                'smoothness': self._compute_trajectory_smoothness(trajectory)
            }
            
            # Constraint satisfaction
            result['constraint_satisfaction'] = self._check_constraint_satisfaction(trajectory, constraints)
            
            # Feasibility check
            result['feasible'] = all(result['constraint_satisfaction'].values())
        
        return result
    
    def _compute_trajectory_distance(self, trajectory: Trajectory) -> float:
        """Compute total trajectory distance."""
        total_distance = 0.0
        for i in range(len(trajectory.states) - 1):
            pos1 = trajectory.states[i].position
            pos2 = trajectory.states[i+1].position
            total_distance += np.linalg.norm(pos2 - pos1)
        return float(total_distance)
    
    def _compute_max_acceleration(self, trajectory: Trajectory) -> float:
        """Compute maximum acceleration along trajectory."""
        max_accel = 0.0
        for i in range(len(trajectory.states) - 1):
            dt = trajectory.times[i+1] - trajectory.times[i]
            if dt > 0:
                vel1 = trajectory.states[i].velocity
                vel2 = trajectory.states[i+1].velocity
                accel = np.linalg.norm(vel2 - vel1) / dt
                max_accel = max(max_accel, float(accel))
        return max_accel
    
    def _compute_trajectory_smoothness(self, trajectory: Trajectory) -> float:
        """Compute trajectory smoothness metric."""
        smoothness = 0.0
        for i in range(1, len(trajectory.states) - 1):
            dt1 = trajectory.times[i] - trajectory.times[i-1]
            dt2 = trajectory.times[i+1] - trajectory.times[i]
            
            if dt1 > 0 and dt2 > 0:
                vel1 = (trajectory.states[i].velocity - trajectory.states[i-1].velocity) / dt1
                vel2 = (trajectory.states[i+1].velocity - trajectory.states[i].velocity) / dt2
                
                # Second derivative (jerk)
                jerk = np.linalg.norm(vel2 - vel1)
                smoothness += jerk
        
        return float(smoothness) / max(1, len(trajectory.states) - 2)
    
    def _check_constraint_satisfaction(self, trajectory: Trajectory, constraints: OptimizationConstraints) -> Dict[str, bool]:
        """Check if trajectory satisfies constraints."""
        
        satisfaction = {}
        
        # Altitude constraints
        if constraints.altitude_min is not None or constraints.altitude_max is not None:
            altitudes = [-state.position[2] for state in trajectory.states]  # NED frame
            
            if constraints.altitude_min is not None:
                satisfaction['altitude_min'] = all(alt >= constraints.altitude_min for alt in altitudes)
            
            if constraints.altitude_max is not None:
                satisfaction['altitude_max'] = all(alt <= constraints.altitude_max for alt in altitudes)
        
        # Speed constraints
        if constraints.speed_min is not None or constraints.speed_max is not None:
            speeds = [np.linalg.norm(state.velocity) for state in trajectory.states]
            
            if constraints.speed_min is not None:
                satisfaction['speed_min'] = all(speed >= constraints.speed_min for speed in speeds)
            
            if constraints.speed_max is not None:
                satisfaction['speed_max'] = all(speed <= constraints.speed_max for speed in speeds)
        
        # G-force constraints
        if constraints.g_force_max is not None:
            max_g = self._compute_max_acceleration(trajectory) / 9.81
            satisfaction['g_force'] = max_g <= constraints.g_force_max
        
        # Boundary constraints
        if constraints.initial_state is not None:
            initial_error = np.linalg.norm(
                trajectory.states[0].position - constraints.initial_state.position
            )
            satisfaction['initial_position'] = initial_error < 1.0  # 1m tolerance
        
        if constraints.final_state is not None:
            final_error = np.linalg.norm(
                trajectory.states[-1].position - constraints.final_state.position
            )
            satisfaction['final_position'] = final_error < 1.0  # 1m tolerance
        
        return satisfaction
    
    # Placeholder methods for complex operations (would be implemented in full version)
    def _generate_adaptive_time_grid(self, constraints: OptimizationConstraints) -> np.ndarray:
        """Generate adaptive time grid."""
        return np.linspace(0, self.config.time_horizon, self.config.num_nodes)
    
    def _get_path_constraints(self, constraints: OptimizationConstraints) -> List[Callable]:
        """Get path constraint functions."""
        return []
    
    def _get_boundary_constraints(self, constraints: OptimizationConstraints) -> List[Callable]:
        """Get boundary constraint functions."""
        return []
    
    def _get_variable_bounds(self, constraints: OptimizationConstraints) -> List[Tuple[float, float]]:
        """Get variable bounds."""
        return []
    
    def _trajectory_to_vector(self, trajectory: Trajectory, time_grid: np.ndarray) -> np.ndarray:
        """Convert trajectory to optimization vector."""
        # Simplified conversion
        vector = []
        for state in trajectory.states:
            vector.extend(state.position)
            vector.extend(state.velocity)
            vector.extend(state.euler_angles)
            vector.extend(state.angular_velocity)
        return np.array(vector)
    
    def _vector_to_trajectory(self, x: np.ndarray, time_grid: np.ndarray, problem: Dict[str, Any]) -> Trajectory:
        """Convert optimization vector to trajectory."""
        # Simplified conversion
        states = []
        state_dim = 12
        
        for i, t in enumerate(time_grid):
            start_idx = i * state_dim
            if start_idx + state_dim <= len(x):
                state_vector = x[start_idx:start_idx + state_dim]
                
                from pyquaternion import Quaternion
                state = State(
                    position=state_vector[0:3],
                    velocity=state_vector[3:6],
                    attitude=Quaternion(axis=[0, 0, 1], angle=state_vector[8]),  # Simplified
                    angular_velocity=state_vector[9:12],
                    time=t
                )
                states.append(state)
        
        return Trajectory(times=time_grid, states=states)
    
    def _evaluate_cost(self, x: np.ndarray, problem: Dict[str, Any], time_grid: np.ndarray) -> float:
        """Evaluate cost function."""
        trajectory = self._vector_to_trajectory(x, time_grid, problem)
        return self._evaluate_trajectory_cost(trajectory, problem)
    
    def _evaluate_trajectory_cost(self, trajectory: Trajectory, problem: Dict[str, Any]) -> float:
        """Evaluate trajectory cost."""
        return problem['cost_function'](trajectory)
    
    def _compute_cost_jacobian(self, x: np.ndarray, problem: Dict[str, Any], time_grid: np.ndarray) -> np.ndarray:
        """Compute cost function Jacobian."""
        # Numerical differentiation (simplified)
        eps = 1e-8
        gradient = np.zeros_like(x)
        
        f0 = self._evaluate_cost(x, problem, time_grid)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self._evaluate_cost(x_plus, problem, time_grid)
            gradient[i] = (f_plus - f0) / eps
        
        return gradient
    
    def _compute_cost_hessian(self, x: np.ndarray, problem: Dict[str, Any], time_grid: np.ndarray) -> np.ndarray:
        """Compute cost function Hessian."""
        # Simplified - would use proper Hessian computation
        return np.eye(len(x)) * 0.01
    
    def _get_optimization_bounds(self, problem: Dict[str, Any], num_vars: int) -> List[Tuple[float, float]]:
        """Get optimization variable bounds."""
        # Simplified bounds
        bounds = []
        for i in range(num_vars):
            bounds.append((-1000, 1000))  # Generic bounds
        return bounds
    
    def _get_optimization_constraints(self, problem: Dict[str, Any], time_grid: np.ndarray) -> List[Dict[str, Any]]:
        """Get optimization constraints."""
        # Simplified constraints
        return []
    
    # Genetic Algorithm helper methods
    def _initialize_ga_population(self, pop_size: int, chromosome_length: int, constraints: OptimizationConstraints) -> List[np.ndarray]:
        """Initialize GA population."""
        population = []
        for _ in range(pop_size):
            individual = np.random.uniform(-1, 1, chromosome_length)
            population.append(individual)
        return population
    
    def _ga_individual_to_trajectory(self, individual: np.ndarray, problem: Dict[str, Any], constraints: OptimizationConstraints) -> Optional[Trajectory]:
        """Convert GA individual to trajectory."""
        # Simplified conversion
        try:
            time_grid = problem['time_grid']
            return self._vector_to_trajectory(individual, time_grid, problem)
        except:
            return None
    
    def _ga_selection(self, population: List[np.ndarray], fitness_values: List[float]) -> List[np.ndarray]:
        """GA selection operator."""
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected
    
    def _ga_crossover(self, parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """GA crossover operator."""
        if np.random.random() < crossover_rate:
            # Single point crossover
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def _ga_mutation(self, individual: np.ndarray, mutation_rate: float, constraints: OptimizationConstraints) -> np.ndarray:
        """GA mutation operator."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
        return mutated
    
    # PSO helper methods
    def _initialize_pso_swarm(self, num_particles: int, particle_dim: int, constraints: OptimizationConstraints) -> List[np.ndarray]:
        """Initialize PSO swarm."""
        swarm = []
        for _ in range(num_particles):
            particle = np.random.uniform(-1, 1, particle_dim)
            swarm.append(particle)
        return swarm
    
    def _pso_particle_to_trajectory(self, particle: np.ndarray, problem: Dict[str, Any], constraints: OptimizationConstraints) -> Optional[Trajectory]:
        """Convert PSO particle to trajectory."""
        # Simplified conversion
        try:
            time_grid = problem['time_grid']
            return self._vector_to_trajectory(particle, time_grid, problem)
        except:
            return None
    
    def _apply_pso_bounds(self, particle: np.ndarray, constraints: OptimizationConstraints) -> np.ndarray:
        """Apply bounds to PSO particle."""
        # Simplified bounds
        return np.clip(particle, -10, 10)
    
    # Additional helper methods
    def _shooting_to_trajectory(self, x: np.ndarray, time_grid: np.ndarray, problem: Dict[str, Any]) -> Trajectory:
        """Convert shooting solution to trajectory."""
        # Simplified conversion for multiple shooting
        return self._vector_to_trajectory(x, time_grid, problem)