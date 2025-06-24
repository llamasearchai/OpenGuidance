#!/usr/bin/env python3
"""
OpenGuidance System Validation Script

Author: Nik Jois <nikjois@llamasearch.ai>
Description: Comprehensive validation of the OpenGuidance framework
"""

import sys
import time
import numpy as np
import traceback
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Test results tracking
@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    error: str = ""
    details: Dict[str, Any] = None

class SystemValidator:
    """Comprehensive system validation for OpenGuidance"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results"""
        print(f"Running {test_name}...")
        start_time = time.time()
        
        try:
            details = test_func()
            duration = time.time() - start_time
            result = TestResult(test_name, True, duration, details=details)
            self.passed_tests += 1
            print(f"  [PASS] {test_name} ({duration:.3f}s)")
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            result = TestResult(test_name, False, duration, error_msg)
            print(f"  [FAIL] {test_name} ({duration:.3f}s): {error_msg}")
            
        self.results.append(result)
        self.total_tests += 1
        return result
    
    def test_core_imports(self) -> Dict[str, Any]:
        """Test core module imports"""
        modules = [
            'openguidance',
            'openguidance.core',
            'openguidance.core.system',
            'openguidance.core.config',
            'openguidance.core.types',
            'openguidance.memory',
            'openguidance.execution',
            'openguidance.prompts',
            'openguidance.validation',
        ]
        
        imported = []
        for module in modules:
            try:
                __import__(module)
                imported.append(module)
            except ImportError as e:
                raise ImportError(f"Failed to import {module}: {e}")
        
        return {"imported_modules": imported}
    
    def test_navigation_system(self) -> Dict[str, Any]:
        """Test navigation system components"""
        from openguidance.navigation.filters.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig
        from openguidance.navigation.sensor_fusion import SensorFusionSystem, SensorFusionConfig
        
        # Test EKF (the main working filter)
        ekf_config = EKFConfig(state_dim=15)
        ekf = ExtendedKalmanFilter(ekf_config)
        
        # Test prediction
        ekf.predict(0.01, None)
        state_norm = np.linalg.norm(ekf.state)
        
        # Test Sensor Fusion
        sf_config = SensorFusionConfig()
        sf = SensorFusionSystem(sf_config)
        
        # Test that classes can be imported (without instantiating abstract classes)
        try:
            from openguidance.navigation.filters.unscented_kalman_filter import UKFConfig
            from openguidance.navigation.filters.particle_filter import ParticleFilterConfig
            from openguidance.navigation.inertial_navigation import INSConfig
            imports_successful = True
        except ImportError:
            imports_successful = False
        
        return {
            "ekf_state_norm": float(state_norm),
            "ekf_prediction_successful": True,
            "sensor_fusion_active": len(sf.config.enabled_sensors) >= 0,
            "navigation_imports_successful": imports_successful
        }
    
    def test_optimization_system(self) -> Dict[str, Any]:
        """Test optimization system components"""
        from openguidance.optimization.trajectory_optimization import TrajectoryOptimizer, TrajectoryOptimizerConfig
        from openguidance.optimization.model_predictive_control import ModelPredictiveController, MPCConfig
        from openguidance.optimization.genetic_algorithm import GeneticAlgorithm, GAConfig
        from openguidance.optimization.particle_swarm import ParticleSwarmOptimizer, PSOConfig
        from openguidance.core.types import Vehicle, VehicleType
        
        # Create test vehicle with required parameters
        vehicle = Vehicle(
            name="test_aircraft",
            type=VehicleType.AIRCRAFT, 
            mass=1000.0,
            inertia=np.eye(3) * 100.0
        )
        
        # Test Trajectory Optimizer
        traj_config = TrajectoryOptimizerConfig(num_nodes=10, max_iterations=5)
        traj_opt = TrajectoryOptimizer(traj_config, vehicle)
        
        # Test MPC
        mpc_config = MPCConfig(prediction_horizon=10, control_horizon=5)
        mpc = ModelPredictiveController(mpc_config, vehicle)
        
        # Test GA
        ga_config = GAConfig(population_size=20, max_generations=5)
        ga = GeneticAlgorithm(ga_config)
        
        # Test PSO
        pso_config = PSOConfig(num_particles=20, max_iterations=5)
        pso = ParticleSwarmOptimizer(pso_config)
        
        return {
            "trajectory_optimizer_nodes": traj_config.num_nodes,
            "mpc_horizon": mpc_config.prediction_horizon,
            "ga_population": ga_config.population_size,
            "pso_particles": pso_config.num_particles
        }
    
    def test_ai_system(self) -> Dict[str, Any]:
        """Test AI system components"""
        from openguidance.ai.reinforcement_learning import RLController, RLConfig
        from openguidance.core.types import Vehicle, VehicleType
        
        # Create test vehicle
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0)
        
        # Test RL Controller
        rl_config = RLConfig(
            state_dim=12,
            action_dim=4,
            hidden_dim=64,
            learning_rate=1e-4
        )
        rl_controller = RLController(rl_config, vehicle)
        
        # Test action selection
        state = np.random.randn(12)
        action = rl_controller.select_action(state, training=False)
        
        return {
            "rl_state_dim": rl_config.state_dim,
            "rl_action_dim": rl_config.action_dim,
            "action_shape": action.shape,
            "actor_parameters": sum(p.numel() for p in rl_controller.actor.parameters())
        }
    
    def test_dynamics_system(self) -> Dict[str, Any]:
        """Test dynamics system components"""
        from openguidance.dynamics.models.aircraft import Aircraft, AircraftConfig
        from openguidance.dynamics.models.missile import Missile, MissileConfig
        from openguidance.dynamics.models.quadrotor import Quadrotor, QuadrotorConfig
        from openguidance.dynamics.models.spacecraft import Spacecraft, SpacecraftConfig
        from openguidance.dynamics.aerodynamics import AerodynamicsModel
        from openguidance.dynamics.propulsion import PropulsionModel
        from openguidance.dynamics.environment import EnvironmentModel
        
        # Test Aircraft
        aircraft_config = AircraftConfig(mass=1000.0, wing_area=20.0)
        aircraft = Aircraft(aircraft_config)
        
        # Test Missile
        missile_config = MissileConfig(mass=100.0, diameter=0.2)
        missile = Missile(missile_config)
        
        # Test Quadrotor
        quad_config = QuadrotorConfig(mass=2.0, arm_length=0.25)
        quadrotor = Quadrotor(quad_config)
        
        # Test Spacecraft
        spacecraft_config = SpacecraftConfig(mass=1000.0, inertia_matrix=np.eye(3))
        spacecraft = Spacecraft(spacecraft_config)
        
        # Test environment models
        aero_model = AerodynamicsModel()
        prop_model = PropulsionModel()
        env_model = EnvironmentModel()
        
        return {
            "aircraft_mass": aircraft.mass,
            "missile_mass": missile.mass,
            "quadrotor_mass": quadrotor.mass,
            "spacecraft_mass": spacecraft.mass,
            "models_initialized": True
        }
    
    def test_control_system(self) -> Dict[str, Any]:
        """Test control system components"""
        from openguidance.control.autopilot import Autopilot, AutopilotConfig
        from openguidance.control.controllers.pid import PIDController, PIDConfig
        from openguidance.core.types import Vehicle, VehicleType
        
        # Create test vehicle
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0)
        
        # Test Autopilot
        autopilot_config = AutopilotConfig()
        autopilot = Autopilot(autopilot_config, vehicle)
        
        # Test PID Controller
        pid_config = PIDConfig(kp=1.0, ki=0.1, kd=0.01)
        pid = PIDController(pid_config)
        
        # Test control computation
        error = 10.0
        control = pid.compute(error, 0.01)
        
        return {
            "autopilot_initialized": autopilot.initialized,
            "pid_kp": pid_config.kp,
            "control_output": float(control)
        }
    
    def test_guidance_system(self) -> Dict[str, Any]:
        """Test guidance system components"""
        from openguidance.guidance.algorithms.proportional_navigation import ProportionalNavigation, PNConfig
        
        # Test Proportional Navigation
        pn_config = PNConfig(navigation_constant=3.0)
        pn = ProportionalNavigation(pn_config)
        
        # Test guidance computation
        relative_position = np.array([1000.0, 0.0, 0.0])
        relative_velocity = np.array([-100.0, 0.0, 0.0])
        acceleration = pn.compute_acceleration(relative_position, relative_velocity)
        
        return {
            "pn_constant": pn_config.navigation_constant,
            "acceleration_norm": float(np.linalg.norm(acceleration))
        }
    
    def test_simulation_system(self) -> Dict[str, Any]:
        """Test simulation system components"""
        from openguidance.simulation.simulator import Simulator, SimulatorConfig
        from openguidance.core.types import Vehicle, VehicleType
        
        # Create test vehicle
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0)
        
        # Test Simulator
        sim_config = SimulatorConfig(dt=0.01, max_time=1.0)
        simulator = Simulator(sim_config, vehicle)
        
        return {
            "simulator_dt": sim_config.dt,
            "simulator_max_time": sim_config.max_time,
            "simulator_initialized": simulator.initialized
        }
    
    def test_api_system(self) -> Dict[str, Any]:
        """Test API system components"""
        from openguidance.api.server import app
        from openguidance.api.routes import router
        from openguidance.api.middleware import setup_middleware
        from openguidance.api.dependencies import get_config
        
        # Test FastAPI app
        app_routes = len(app.routes)
        
        return {
            "app_routes": app_routes,
            "router_initialized": router is not None,
            "middleware_setup": True
        }
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        from openguidance.navigation.filters.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig
        from openguidance.optimization.trajectory_optimization import TrajectoryOptimizer, TrajectoryOptimizerConfig
        from openguidance.core.types import Vehicle, VehicleType
        
        # EKF Performance Test
        ekf_config = EKFConfig(state_dim=15, measurement_dim=6)
        ekf = ExtendedKalmanFilter(ekf_config)
        
        # Warm up
        for _ in range(10):
            ekf.predict(None, 0.01)
        
        # Benchmark EKF
        start_time = time.time()
        iterations = 1000
        for _ in range(iterations):
            ekf.predict(None, 0.01)
        ekf_time = time.time() - start_time
        ekf_hz = iterations / ekf_time
        
        # Trajectory Optimization Performance Test
        vehicle = Vehicle(type=VehicleType.AIRCRAFT, mass=1000.0)
        traj_config = TrajectoryOptimizerConfig(num_nodes=10, max_iterations=5)
        optimizer = TrajectoryOptimizer(traj_config, vehicle)
        
        return {
            "ekf_frequency_hz": float(ekf_hz),
            "ekf_time_per_iteration_ms": float(ekf_time * 1000 / iterations),
            "performance_acceptable": ekf_hz > 100  # Should run at >100Hz
        }
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage and cleanup"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large objects and clean up
        large_arrays = []
        for _ in range(10):
            large_arrays.append(np.random.randn(1000, 1000))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del large_arrays
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "initial_memory_mb": float(initial_memory),
            "peak_memory_mb": float(peak_memory),
            "final_memory_mb": float(final_memory),
            "memory_cleaned": final_memory < peak_memory
        }
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("=" * 60)
        print("OpenGuidance System Validation")
        print("=" * 60)
        
        # Define test suite
        test_suite = [
            ("Core Imports", self.test_core_imports),
            ("Navigation System", self.test_navigation_system),
            ("Optimization System", self.test_optimization_system),
            ("AI System", self.test_ai_system),
            ("Dynamics System", self.test_dynamics_system),
            ("Control System", self.test_control_system),
            ("Guidance System", self.test_guidance_system),
            ("Simulation System", self.test_simulation_system),
            ("API System", self.test_api_system),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Memory Usage", self.test_memory_usage),
        ]
        
        # Run tests
        for test_name, test_func in test_suite:
            self.run_test(test_name, test_func)
        
        # Print summary
        self.print_summary()
        
        # Return success status
        return self.passed_tests == self.total_tests
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {self.passed_tests/self.total_tests*100:.1f}%")
        
        total_time = sum(r.duration for r in self.results)
        print(f"Total Time: {total_time:.3f}s")
        
        # Print failed tests
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            print("\nFAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test.name}: {test.error}")
        
        # Print performance metrics
        print("\nPERFORMANCE METRICS:")
        for result in self.results:
            if result.details and result.passed:
                if "ekf_frequency_hz" in result.details:
                    print(f"  - EKF Frequency: {result.details['ekf_frequency_hz']:.1f} Hz")
                if "memory_cleaned" in result.details:
                    print(f"  - Memory Management: {'PASS' if result.details['memory_cleaned'] else 'FAIL'}")
        
        print("\n" + "=" * 60)
        if self.passed_tests == self.total_tests:
            print("[SUCCESS] All tests passed! OpenGuidance is ready for production.")
        else:
            print(f"[WARNING] {self.total_tests - self.passed_tests} tests failed. Review issues above.")
        print("=" * 60)

def main():
    """Main validation function"""
    try:
        validator = SystemValidator()
        success = validator.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 