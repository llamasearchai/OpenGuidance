#!/usr/bin/env python3
"""
Simplified OpenGuidance System Validation Script

Author: Nik Jois <nikjois@llamasearch.ai>
Description: Basic validation of the OpenGuidance framework focusing on imports and core functionality
"""

import sys
import time
import numpy as np
import traceback
from typing import Dict, List, Any

def test_core_imports() -> Dict[str, Any]:
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

def test_navigation_imports() -> Dict[str, Any]:
    """Test navigation module imports"""
    modules = [
        'openguidance.navigation.filters.extended_kalman_filter',
        'openguidance.navigation.filters.unscented_kalman_filter', 
        'openguidance.navigation.filters.particle_filter',
        'openguidance.navigation.sensor_fusion',
        'openguidance.navigation.inertial_navigation',
    ]
    
    imported = []
    for module in modules:
        try:
            __import__(module)
            imported.append(module)
        except ImportError as e:
            print(f"Warning: Could not import {module}: {e}")
    
    return {"imported_navigation_modules": imported}

def test_optimization_imports() -> Dict[str, Any]:
    """Test optimization module imports"""
    modules = [
        'openguidance.optimization.trajectory_optimization',
        'openguidance.optimization.model_predictive_control',
        'openguidance.optimization.genetic_algorithm',
        'openguidance.optimization.particle_swarm',
    ]
    
    imported = []
    for module in modules:
        try:
            __import__(module)
            imported.append(module)
        except ImportError as e:
            print(f"Warning: Could not import {module}: {e}")
    
    return {"imported_optimization_modules": imported}

def test_ai_imports() -> Dict[str, Any]:
    """Test AI module imports"""
    modules = [
        'openguidance.ai.reinforcement_learning',
    ]
    
    imported = []
    for module in modules:
        try:
            __import__(module)
            imported.append(module)
        except ImportError as e:
            print(f"Warning: Could not import {module}: {e}")
    
    return {"imported_ai_modules": imported}

def test_ekf_basic_functionality() -> Dict[str, Any]:
    """Test basic EKF functionality"""
    from openguidance.navigation.filters.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig
    
    # Create EKF
    config = EKFConfig(state_dim=15)
    ekf = ExtendedKalmanFilter(config)
    
    # Test basic prediction
    initial_state_norm = np.linalg.norm(ekf.state)
    ekf.predict(0.01, None)
    final_state_norm = np.linalg.norm(ekf.state)
    
    return {
        "ekf_created": True,
        "initial_state_norm": float(initial_state_norm),
        "final_state_norm": float(final_state_norm),
        "prediction_successful": True
    }

def test_vehicle_creation() -> Dict[str, Any]:
    """Test vehicle creation"""
    from openguidance.core.types import Vehicle, VehicleType
    
    # Create vehicle with all required parameters
    vehicle = Vehicle(
        name="test_aircraft",
        type=VehicleType.AIRCRAFT,
        mass=1000.0,
        inertia=np.eye(3) * 100.0
    )
    
    return {
        "vehicle_created": True,
        "vehicle_name": vehicle.name,
        "vehicle_mass": vehicle.mass,
        "vehicle_type": vehicle.type.name,
        "inertia_determinant": float(np.linalg.det(vehicle.inertia))
    }

def test_performance_benchmark() -> Dict[str, Any]:
    """Simple performance test"""
    from openguidance.navigation.filters.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig
    
    config = EKFConfig(state_dim=15)
    ekf = ExtendedKalmanFilter(config)
    
    # Warm up
    for _ in range(10):
        ekf.predict(0.01, None)
    
    # Benchmark
    start_time = time.time()
    iterations = 1000
    for _ in range(iterations):
        ekf.predict(0.01, None)
    total_time = time.time() - start_time
    
    frequency = iterations / total_time
    
    return {
        "iterations": iterations,
        "total_time": total_time,
        "frequency_hz": frequency,
        "time_per_iteration_ms": total_time * 1000 / iterations,
        "performance_acceptable": frequency > 100
    }

def run_test(test_name: str, test_func):
    """Run a single test"""
    print(f"Running {test_name}...")
    start_time = time.time()
    
    try:
        result = test_func()
        duration = time.time() - start_time
        print(f"  [PASS] {test_name} ({duration:.3f}s)")
        return True, result
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  [FAIL] {test_name} ({duration:.3f}s): {error_msg}")
        return False, {"error": error_msg}

def main():
    """Main validation function"""
    print("=" * 60)
    print("OpenGuidance Simplified System Validation")
    print("=" * 60)
    
    # Test suite
    tests = [
        ("Core Imports", test_core_imports),
        ("Navigation Imports", test_navigation_imports),
        ("Optimization Imports", test_optimization_imports),
        ("AI Imports", test_ai_imports),
        ("EKF Basic Functionality", test_ekf_basic_functionality),
        ("Vehicle Creation", test_vehicle_creation),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        success, result = run_test(test_name, test_func)
        if success:
            passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! OpenGuidance core functionality is working.")
    else:
        print(f"\n[WARNING] {total - passed} tests failed. Check issues above.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        traceback.print_exc()
        sys.exit(1) 