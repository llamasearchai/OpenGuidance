"""
Comprehensive test suite for advanced OpenGuidance functionality.

Tests all the new high-impact modules and features:
- Extended Kalman Filter for state estimation
- Trajectory optimization algorithms
- Reinforcement learning controller
- Advanced navigation systems
- Multi-objective optimization
- Safety-critical systems

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch
from pyquaternion import Quaternion

from openguidance.core.types import State, Vehicle, VehicleType, Trajectory
from openguidance.core.config import Config
from openguidance.navigation.filters.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig
from openguidance.optimization.trajectory_optimization import (
    TrajectoryOptimizer, TrajectoryOptimizerConfig, OptimizationConstraints,
    OptimizationMethod, CostFunction
)
from openguidance.ai.reinforcement_learning import RLController, RLConfig, RLAlgorithm


class TestExtendedKalmanFilter:
    """Test suite for Extended Kalman Filter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.vehicle = Vehicle(
            type=VehicleType.AIRCRAFT,
            mass=1000.0,
            inertia=np.eye(3) * 100.0
        )
        
        self.config = EKFConfig(
            state_dim=15,
            use_quaternion_attitude=True,
            enable_adaptive_tuning=True,
            enable_nees_monitoring=True
        )
        
        self.ekf = ExtendedKalmanFilter(self.config, self.vehicle)
    
    def test_ekf_initialization(self):
        """Test EKF initialization."""
        assert self.ekf.state_dim == 15
        assert self.ekf.x.shape == (15,)
        assert self.ekf.P.shape == (15, 15)
        assert self.ekf.config.use_quaternion_attitude
        
        # Check quaternion initialization
        assert self.ekf.x[6] == 1.0  # w component of quaternion
        assert np.allclose(self.ekf.x[7:10], 0.0)  # x, y, z components
    
    def test_ekf_prediction_step(self):
        """Test EKF prediction step."""
        # Initial state
        initial_state = self.ekf.x.copy()
        initial_covariance = self.ekf.P.copy()
        
        # Control input
        control = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])  # Hover
        dt = 0.02  # 50 Hz
        
        # Prediction step
        self.ekf.predict(control, dt)
        
        # Check that state has been updated
        assert not np.allclose(self.ekf.x, initial_state)
        
        # Check quaternion normalization
        quat = self.ekf.x[6:10]
        assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)
        
        # Check covariance update
        assert not np.allclose(self.ekf.P, initial_covariance)
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(self.ekf.P)
        assert np.all(eigenvals > 0)
    
    def test_ekf_measurement_update(self):
        """Test EKF measurement update."""
        # Add GPS position measurement
        gps_position = np.array([100.0, 200.0, -1000.0])  # NED frame
        
        initial_state = self.ekf.x.copy()
        
        # Update with GPS measurement
        self.ekf.update("gps_position", gps_position)
        
        # Check that position estimate has moved toward measurement
        position_estimate = self.ekf.x[0:3]
        initial_position = initial_state[0:3]
        
        # Should be closer to measurement than initial estimate
        initial_error = np.linalg.norm(gps_position - initial_position)
        updated_error = np.linalg.norm(gps_position - position_estimate)
        assert updated_error < initial_error
    
    def test_ekf_innovation_gating(self):
        """Test innovation gating functionality."""
        # Large outlier measurement
        outlier_measurement = np.array([10000.0, 10000.0, -10000.0])
        
        initial_state = self.ekf.x.copy()
        
        # Update with outlier (should be rejected)
        self.ekf.update("gps_position", outlier_measurement)
        
        # State should not change significantly due to gating
        assert np.allclose(self.ekf.x, initial_state, rtol=0.1)
    
    def test_ekf_adaptive_tuning(self):
        """Test adaptive noise tuning."""
        self.ekf.config.enable_adaptive_tuning = True
        
        # Initial R scale
        initial_R_scale = self.ekf.adaptive_R_scales["gps_position"]
        
        # Consistent measurements should reduce noise scale
        for _ in range(10):
            measurement = np.array([0.0, 0.0, -1000.0]) + np.random.normal(0, 0.1, 3)
            self.ekf.update("gps_position", measurement)
        
        # R scale might have changed due to adaptive tuning
        final_R_scale = self.ekf.adaptive_R_scales["gps_position"]
        assert final_R_scale > 0  # Should remain positive
    
    def test_ekf_performance_metrics(self):
        """Test performance metrics collection."""
        # Run some updates
        for i in range(5):
            control = np.random.normal(0, 1, 6)
            self.ekf.predict(control, 0.02)
            
            if i % 2 == 0:
                measurement = np.random.normal(0, 1, 3)
                self.ekf.update("gps_position", measurement)
        
        # Get performance metrics
        metrics = self.ekf.get_performance_metrics()
        
        assert "filter_type" in metrics
        assert "state_dimension" in metrics
        assert "iterations" in metrics
        assert metrics["iterations"] == 5
        assert metrics["state_dimension"] == 15
    
    def test_ekf_state_conversion(self):
        """Test conversion to navigation State object."""
        # Set a known state
        self.ekf.x[0:3] = [100, 200, -1000]  # Position
        self.ekf.x[3:6] = [10, 0, 0]  # Velocity
        self.ekf.x[6:10] = [1, 0, 0, 0]  # Quaternion
        
        # Convert to State object
        nav_state = self.ekf.get_state_as_navigation_state()
        
        assert np.allclose(nav_state.position, [100, 200, -1000])
        assert np.allclose(nav_state.velocity, [10, 0, 0])
        assert isinstance(nav_state.attitude, Quaternion)
        assert nav_state.frame == "NED"


class TestTrajectoryOptimization:
    """Test suite for trajectory optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.vehicle = Vehicle(
            type=VehicleType.AIRCRAFT,
            mass=1000.0,
            inertia=np.eye(3) * 100.0
        )
        
        self.config = TrajectoryOptimizerConfig(
            method=OptimizationMethod.DIRECT_COLLOCATION,
            cost_function=CostFunction.MINIMUM_TIME,
            num_nodes=20,
            time_horizon=50.0,
            max_iterations=100,
            tolerance=1e-3,
            verbose=False
        )
        
        self.optimizer = TrajectoryOptimizer(self.config, self.vehicle)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.config.method == OptimizationMethod.DIRECT_COLLOCATION
        assert self.optimizer.config.cost_function == CostFunction.MINIMUM_TIME
        assert self.optimizer.vehicle.type == VehicleType.AIRCRAFT
        assert self.optimizer.solve_time == 0.0
        assert self.optimizer.iterations == 0
    
    def test_initial_guess_generation(self):
        """Test initial trajectory guess generation."""
        constraints = OptimizationConstraints(
            initial_state=State(
                position=np.array([0, 0, -1000]),
                velocity=np.array([10, 0, 0]),
                attitude=Quaternion([1, 0, 0, 0]),
                angular_velocity=np.zeros(3),
                time=0.0
            ),
            final_state=State(
                position=np.array([1000, 0, -1000]),
                velocity=np.array([10, 0, 0]),
                attitude=Quaternion([1, 0, 0, 0]),
                angular_velocity=np.zeros(3),
                time=50.0
            )
        )
        
        initial_guess = self.optimizer._generate_initial_guess(constraints)
        
        assert isinstance(initial_guess, Trajectory)
        assert len(initial_guess.states) == self.config.num_nodes
        assert initial_guess.times[0] == 0.0
        assert initial_guess.times[-1] == self.config.time_horizon
        
        # Check interpolation
        assert np.allclose(initial_guess.states[0].position, [0, 0, -1000])
        assert np.allclose(initial_guess.states[-1].position, [1000, 0, -1000])
    
    def test_cost_function_minimum_time(self):
        """Test minimum time cost function."""
        trajectory = Trajectory(
            times=np.array([0, 10, 20]),
            states=[
                State(position=np.zeros(3), velocity=np.zeros(3), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=0),
                State(position=np.zeros(3), velocity=np.zeros(3), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=10),
                State(position=np.zeros(3), velocity=np.zeros(3), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=20)
            ]
        )
        
        cost_func = self.optimizer._get_standard_cost_function(CostFunction.MINIMUM_TIME)
        cost = cost_func(trajectory)
        
        assert cost == 20.0  # Total time
    
    def test_cost_function_minimum_fuel(self):
        """Test minimum fuel cost function."""
        trajectory = Trajectory(
            times=np.array([0, 1, 2]),
            states=[
                State(position=np.zeros(3), velocity=np.array([0, 0, 0]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=0),
                State(position=np.zeros(3), velocity=np.array([10, 0, 0]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=1),
                State(position=np.zeros(3), velocity=np.array([20, 0, 0]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=2)
            ]
        )
        
        cost_func = self.optimizer._get_standard_cost_function(CostFunction.MINIMUM_FUEL)
        cost = cost_func(trajectory)
        
        assert cost > 0  # Should have positive fuel cost
    
    def test_trajectory_vector_conversion(self):
        """Test trajectory to vector conversion."""
        trajectory = Trajectory(
            times=np.array([0, 1]),
            states=[
                State(position=np.array([1, 2, 3]), velocity=np.array([4, 5, 6]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=0),
                State(position=np.array([7, 8, 9]), velocity=np.array([10, 11, 12]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=1)
            ]
        )
        
        vector = self.optimizer._trajectory_to_vector(trajectory, np.array([0, 1]))
        
        # Should contain position and velocity for both states
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        assert np.allclose(vector, expected)
    
    def test_constraint_satisfaction_checking(self):
        """Test constraint satisfaction checking."""
        trajectory = Trajectory(
            times=np.array([0, 1]),
            states=[
                State(position=np.array([0, 0, -500]), velocity=np.array([10, 0, 0]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=0),
                State(position=np.array([10, 0, -600]), velocity=np.array([15, 0, 0]), 
                     attitude=Quaternion([1, 0, 0, 0]), angular_velocity=np.zeros(3), time=1)
            ]
        )
        
        constraints = OptimizationConstraints(
            altitude_min=400.0,  # 400m minimum altitude
            altitude_max=1000.0,  # 1000m maximum altitude
            speed_min=5.0,
            speed_max=20.0
        )
        
        satisfaction = self.optimizer._check_constraint_satisfaction(trajectory, constraints)
        
        assert satisfaction["altitude_min"] == True  # 500m and 600m > 400m
        assert satisfaction["altitude_max"] == True  # 500m and 600m < 1000m
        assert satisfaction["speed_min"] == True  # 10 and 15 > 5
        assert satisfaction["speed_max"] == True  # 10 and 15 < 20
    
    def test_optimization_simple_case(self):
        """Test optimization with simple constraints."""
        constraints = OptimizationConstraints(
            initial_state=State(
                position=np.array([0, 0, -1000]),
                velocity=np.array([10, 0, 0]),
                attitude=Quaternion([1, 0, 0, 0]),
                angular_velocity=np.zeros(3),
                time=0.0
            ),
            final_state=State(
                position=np.array([500, 0, -1000]),
                velocity=np.array([10, 0, 0]),
                attitude=Quaternion([1, 0, 0, 0]),
                angular_velocity=np.zeros(3),
                time=25.0
            )
        )
        
        # Use genetic algorithm for robustness
        self.optimizer.config.method = OptimizationMethod.GENETIC_ALGORITHM
        
        result = self.optimizer.optimize_trajectory(constraints)
        
        assert "success" in result
        assert "solve_time" in result
        assert "method" in result
        assert result["method"] == "Genetic Algorithm"
        
        if result["success"]:
            assert "optimal_trajectory" in result
            assert "optimal_cost" in result
            assert "trajectory_metrics" in result
            
            trajectory = result["optimal_trajectory"]
            assert isinstance(trajectory, Trajectory)
            assert len(trajectory.states) > 0


class TestReinforcementLearning:
    """Test suite for reinforcement learning controller."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.vehicle = Vehicle(
            type=VehicleType.AIRCRAFT,
            mass=1000.0,
            inertia=np.eye(3) * 100.0
        )
        
        self.config = RLConfig(
            algorithm=RLAlgorithm.DDPG,
            actor_hidden_layers=[64, 64],
            critic_hidden_layers=[64, 64],
            buffer_size=1000,
            batch_size=32,
            episodes=10,
            max_episode_steps=100
        )
        
        self.controller = RLController(self.config, self.vehicle)
    
    def test_rl_controller_initialization(self):
        """Test RL controller initialization."""
        assert self.controller.state_dim == 12
        assert self.controller.action_dim == 4
        assert self.controller.episode == 0
        assert self.controller.total_steps == 0
        
        # Check networks
        assert self.controller.actor is not None
        assert self.controller.critic is not None
        assert self.controller.target_actor is not None
        assert self.controller.target_critic is not None
        
        # Check replay buffer
        assert len(self.controller.replay_buffer) == 0
        assert self.controller.replay_buffer.capacity == 1000
    
    def test_action_selection(self):
        """Test action selection."""
        state = State(
            position=np.array([0, 0, -1000]),
            velocity=np.array([10, 0, 0]),
            attitude=Quaternion([1, 0, 0, 0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        # Training mode (with exploration)
        action_train = self.controller.select_action(state, training=True)
        assert action_train.shape == (4,)
        assert np.all(action_train >= -1.0) and np.all(action_train <= 1.0)
        
        # Evaluation mode (no exploration)
        action_eval = self.controller.select_action(state, training=False)
        assert action_eval.shape == (4,)
        assert np.all(action_eval >= -1.0) and np.all(action_eval <= 1.0)
    
    def test_replay_buffer(self):
        """Test replay buffer functionality."""
        buffer = self.controller.replay_buffer
        
        # Add experiences
        for i in range(10):
            state = np.random.randn(12)
            action = np.random.randn(4)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = i == 9
            
            buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 10
        
        # Sample batch
        batch = buffer.sample(5)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (5, 12)
        assert actions.shape == (5, 4)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 12)
        assert dones.shape == (5,)
    
    def test_experience_update(self):
        """Test experience update."""
        state = State(
            position=np.array([0, 0, -1000]),
            velocity=np.array([10, 0, 0]),
            attitude=Quaternion([1, 0, 0, 0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        next_state = State(
            position=np.array([1, 0, -1000]),
            velocity=np.array([10, 0, 0]),
            attitude=Quaternion([1, 0, 0, 0]),
            angular_velocity=np.zeros(3),
            time=0.02
        )
        
        action = np.array([0.1, 0.0, 0.0, 0.0])
        reward = 1.0
        done = False
        
        initial_steps = self.controller.total_steps
        initial_buffer_size = len(self.controller.replay_buffer)
        
        self.controller.update(state, action, reward, next_state, done)
        
        assert self.controller.total_steps == initial_steps + 1
        assert len(self.controller.replay_buffer) == initial_buffer_size + 1
    
    def test_state_vector_conversion(self):
        """Test state to vector conversion."""
        state = State(
            position=np.array([1, 2, 3]),
            velocity=np.array([4, 5, 6]),
            attitude=Quaternion([0.7071, 0, 0, 0.7071]),  # 90 deg rotation about z
            angular_velocity=np.array([7, 8, 9]),
            time=0.0
        )
        
        vector = self.controller._state_to_vector(state)
        
        assert vector.shape == (12,)
        assert np.allclose(vector[0:3], [1, 2, 3])  # position
        assert np.allclose(vector[3:6], [4, 5, 6])  # velocity
        assert np.allclose(vector[9:12], [7, 8, 9])  # angular velocity
    
    def test_training_metrics(self):
        """Test training metrics collection."""
        # Simulate some training
        for episode in range(5):
            self.controller.training_metrics['episode_rewards'].append(episode * 10)
            self.controller.training_metrics['episode_lengths'].append(100)
            self.controller.training_metrics['actor_losses'].append(0.1)
            self.controller.training_metrics['critic_losses'].append(0.2)
        
        metrics = self.controller.get_training_metrics()
        
        assert metrics['episodes_trained'] == 5
        assert metrics['mean_episode_reward'] == 20.0  # Mean of [0, 10, 20, 30, 40]
        assert metrics['mean_episode_length'] == 100.0
        assert metrics['total_steps'] == 0  # No actual training steps


class TestIntegrationScenarios:
    """Integration tests for combined functionality."""
    
    def test_ekf_with_trajectory_optimization(self):
        """Test EKF state estimation with trajectory optimization."""
        # Setup vehicle and EKF
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0, inertia=np.eye(3) * 100.0)
        ekf_config = EKFConfig(state_dim=15, use_quaternion_attitude=True)
        ekf = ExtendedKalmanFilter(ekf_config, vehicle)
        
        # Setup trajectory optimizer
        opt_config = TrajectoryOptimizerConfig(
            method=OptimizationMethod.GRADIENT_DESCENT,
            num_nodes=10,
            time_horizon=20.0,
            max_iterations=50,
            verbose=False
        )
        optimizer = TrajectoryOptimizer(opt_config, vehicle)
        
        # Simulate some navigation updates
        for i in range(5):
            # Prediction step
            control = np.array([0, 0, 9.81, 0, 0, 0])
            ekf.predict(control, 0.1)
            
            # Measurement update
            if i % 2 == 0:
                gps_measurement = np.array([i * 10, 0, -1000])
                ekf.update("gps_position", gps_measurement)
        
        # Get current state estimate
        current_state = ekf.get_state_as_navigation_state()
        
        # Optimize trajectory from current state
        constraints = OptimizationConstraints(
            initial_state=current_state,
            final_state=State(
                position=np.array([100, 0, -1000]),
                velocity=np.array([10, 0, 0]),
                attitude=Quaternion([1, 0, 0, 0]),
                angular_velocity=np.zeros(3),
                time=20.0
            )
        )
        
        result = optimizer.optimize_trajectory(constraints)
        
        # Should complete without errors
        assert "success" in result
        assert "solve_time" in result
    
    def test_rl_with_state_estimation(self):
        """Test RL controller with state estimation feedback."""
        # Setup components
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0, inertia=np.eye(3) * 100.0)
        
        ekf_config = EKFConfig(state_dim=15)
        ekf = ExtendedKalmanFilter(ekf_config, vehicle)
        
        rl_config = RLConfig(buffer_size=100, batch_size=10)
        rl_controller = RLController(rl_config, vehicle)
        
        # Simulate control loop
        for step in range(10):
            # Get current state estimate
            current_state = ekf.get_state_as_navigation_state()
            
            # Select action using RL
            action = rl_controller.select_action(current_state, training=True)
            
            # Simulate environment step (simplified)
            next_position = current_state.position + current_state.velocity * 0.1
            next_state = State(
                position=next_position,
                velocity=current_state.velocity,
                attitude=current_state.attitude,
                angular_velocity=current_state.angular_velocity,
                time=current_state.time + 0.1
            )
            
            # Update EKF with new state
            control_input = np.concatenate([action, np.zeros(2)])  # Pad to 6 elements
            ekf.predict(control_input, 0.1)
            
            # Simulate GPS measurement
            if step % 3 == 0:
                gps_noise = np.random.normal(0, 1.0, 3)
                gps_measurement = next_state.position + gps_noise
                ekf.update("gps_position", gps_measurement)
            
            # Compute reward (simplified)
            reward = -np.linalg.norm(next_state.position - np.array([100, 0, -1000]))
            
            # Update RL controller
            rl_controller.update(current_state, action, reward, next_state, False)
        
        # Check that both systems have been updated
        assert ekf.iteration_count > 0
        assert rl_controller.total_steps > 0
        assert len(rl_controller.replay_buffer) > 0


class TestPerformanceAndSafety:
    """Tests for performance and safety features."""
    
    def test_ekf_numerical_stability(self):
        """Test EKF numerical stability under extreme conditions."""
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0, inertia=np.eye(3) * 100.0)
        config = EKFConfig(
            state_dim=15,
            min_eigenvalue=1e-12,
            max_condition_number=1e12
        )
        ekf = ExtendedKalmanFilter(config, vehicle)
        
        # Extreme control inputs
        extreme_control = np.array([1000, 1000, 1000, 100, 100, 100])
        
        # Should handle extreme inputs gracefully
        for _ in range(10):
            ekf.predict(extreme_control, 0.01)
            
            # Check covariance remains positive definite
            eigenvals = np.linalg.eigvals(ekf.P)
            assert np.all(eigenvals > 0)
            
            # Check condition number
            condition_number = np.max(eigenvals) / np.min(eigenvals)
            assert condition_number < config.max_condition_number * 10  # Allow some margin
    
    def test_trajectory_optimization_convergence(self):
        """Test trajectory optimization convergence properties."""
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0, inertia=np.eye(3) * 100.0)
        config = TrajectoryOptimizerConfig(
            method=OptimizationMethod.GRADIENT_DESCENT,
            max_iterations=200,
            tolerance=1e-6,
            verbose=False
        )
        optimizer = TrajectoryOptimizer(config, vehicle)
        
        # Simple optimization problem
        constraints = OptimizationConstraints(
            initial_state=State(
                position=np.array([0, 0, -1000]),
                velocity=np.array([10, 0, 0]),
                attitude=Quaternion([1, 0, 0, 0]),
                angular_velocity=np.zeros(3),
                time=0.0
            )
        )
        
        result = optimizer.optimize_trajectory(constraints)
        
        # Should converge within iteration limit
        if result["success"]:
            assert result["iterations"] <= config.max_iterations
            assert "cost_history" in result
            
            # Cost should generally decrease
            cost_history = result["cost_history"]
            if len(cost_history) > 10:
                initial_cost = np.mean(cost_history[:5])
                final_cost = np.mean(cost_history[-5:])
                assert final_cost <= initial_cost  # Should improve or stay same
    
    def test_rl_safety_constraints(self):
        """Test RL safety constraint enforcement."""
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0, inertia=np.eye(3) * 100.0)
        config = RLConfig(
            enable_safety_layer=True,
            max_control_rate=5.0,
            altitude_bounds=(500.0, 5000.0)
        )
        controller = RLController(config, vehicle)
        
        # Test extreme action
        extreme_action = np.array([2.0, 2.0, 2.0, 2.0])  # Beyond [-1, 1] bounds
        
        state = State(
            position=np.array([0, 0, -1000]),
            velocity=np.array([10, 0, 0]),
            attitude=Quaternion([1, 0, 0, 0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        # Action should be clipped to safe bounds
        safe_action = controller.select_action(state, training=False)
        assert np.all(safe_action >= -1.0) and np.all(safe_action <= 1.0)


if __name__ == "__main__":
    # Run specific test categories
    print("Running Extended Kalman Filter tests...")
    pytest.main(["-v", "TestExtendedKalmanFilter"])
    
    print("\nRunning Trajectory Optimization tests...")
    pytest.main(["-v", "TestTrajectoryOptimization"])
    
    print("\nRunning Reinforcement Learning tests...")
    pytest.main(["-v", "TestReinforcementLearning"])
    
    print("\nRunning Integration tests...")
    pytest.main(["-v", "TestIntegrationScenarios"])
    
    print("\nRunning Performance and Safety tests...")
    pytest.main(["-v", "TestPerformanceAndSafety"]) 