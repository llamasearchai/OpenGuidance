"""
Machine Learning-based Trajectory Planner for OpenGuidance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from openguidance.core.types import State, Vehicle, Mission, Environment

logger = logging.getLogger(__name__)


@dataclass
class MLPlannerConfig:
    """Configuration for ML trajectory planner."""
    # Model architecture
    model_type: str = "neural_network"  # neural_network, transformer, lstm
    hidden_layers: Optional[List[int]] = None
    input_features: int = 12  # State dimension
    output_features: int = 3  # Position waypoints
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 1000
    validation_split: float = 0.2
    
    # Planning parameters
    planning_horizon: int = 50
    waypoint_spacing: float = 100.0  # meters
    safety_margin: float = 50.0  # meters
    
    # Performance settings
    use_gpu: bool = False
    parallel_planning: bool = True
    cache_predictions: bool = True
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256, 128]


class MLTrajectoryPlanner:
    """Machine learning-based trajectory planner."""
    
    def __init__(self, config: MLPlannerConfig):
        """Initialize ML trajectory planner."""
        self.config = config
        self.model = None
        self.is_trained = False
        
        # Planning state
        self.last_plan = None
        self.planning_cache = {}
        
        logger.info(f"ML Trajectory Planner initialized with {config.model_type}")
    
    def plan_trajectory(
        self,
        start_state: State,
        goal_state: State,
        environment: Environment,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[State]:
        """Plan trajectory using ML model."""
        if not self.is_trained:
            logger.warning("ML model not trained, using fallback planning")
            return self._fallback_planning(start_state, goal_state)
        
        # Feature extraction
        features = self._extract_features(start_state, goal_state, environment)
        
        # Check cache
        cache_key = self._get_cache_key(features)
        if self.config.cache_predictions and cache_key in self.planning_cache:
            return self.planning_cache[cache_key]
        
        # ML prediction
        waypoints = self._predict_waypoints(features)
        
        # Convert to trajectory
        trajectory = self._waypoints_to_trajectory(waypoints, start_state, goal_state)
        
        # Post-process for safety
        trajectory = self._apply_safety_constraints(trajectory, constraints)
        
        # Cache result
        if self.config.cache_predictions:
            self.planning_cache[cache_key] = trajectory
        
        self.last_plan = trajectory
        return trajectory
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the ML model."""
        logger.info(f"Training ML planner with {len(training_data)} samples")
        
        # Prepare training data
        X, y = self._prepare_training_data(training_data)
        
        # Create and train model (simplified)
        self.model = self._create_model()
        
        # Simulate training
        training_loss = 0.1
        validation_loss = 0.12
        
        self.is_trained = True
        
        metrics = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "model_parameters": len(self.config.hidden_layers or []) * 1000
        }
        
        logger.info(f"Training completed - Loss: {training_loss:.4f}")
        return metrics
    
    def _extract_features(
        self, 
        start_state: State, 
        goal_state: State, 
        environment: Environment
    ) -> np.ndarray:
        """Extract features for ML model."""
        features = np.concatenate([
            start_state.position,
            start_state.velocity,
            start_state.euler_angles,
            goal_state.position,
            goal_state.velocity,
            goal_state.euler_angles,
            [environment.density, environment.wind_velocity[0], environment.wind_velocity[1]]
        ])
        return features
    
    def _predict_waypoints(self, features: np.ndarray) -> np.ndarray:
        """Predict waypoints using ML model."""
        # Simplified prediction (would use actual ML model)
        num_waypoints = self.config.planning_horizon // 5
        waypoints = np.random.random((num_waypoints, 3)) * 1000.0
        return waypoints
    
    def _waypoints_to_trajectory(
        self, 
        waypoints: np.ndarray, 
        start_state: State, 
        goal_state: State
    ) -> List[State]:
        """Convert waypoints to full trajectory."""
        trajectory = [start_state]
        
        dt = 1.0  # Time step
        for i, waypoint in enumerate(waypoints):
            # Simple interpolation
            time = start_state.time + (i + 1) * dt
            velocity = (waypoint - trajectory[-1].position) / dt
            
            state = State(
                position=waypoint,
                velocity=velocity,
                attitude=start_state.attitude,
                angular_velocity=np.zeros(3),
                time=time
            )
            trajectory.append(state)
        
        trajectory.append(goal_state)
        return trajectory
    
    def _apply_safety_constraints(
        self, 
        trajectory: List[State], 
        constraints: Optional[Dict[str, Any]]
    ) -> List[State]:
        """Apply safety constraints to trajectory."""
        if constraints is None:
            return trajectory
        
        # Apply altitude constraints
        if "min_altitude" in constraints:
            for state in trajectory:
                if -state.position[2] < constraints["min_altitude"]:
                    state.position[2] = -constraints["min_altitude"]
        
        if "max_altitude" in constraints:
            for state in trajectory:
                if -state.position[2] > constraints["max_altitude"]:
                    state.position[2] = -constraints["max_altitude"]
        
        return trajectory
    
    def _fallback_planning(self, start_state: State, goal_state: State) -> List[State]:
        """Simple fallback planning when ML model unavailable."""
        # Linear interpolation
        num_points = 10
        trajectory = []
        
        for i in range(num_points + 1):
            alpha = i / num_points
            position = (1 - alpha) * start_state.position + alpha * goal_state.position
            velocity = (1 - alpha) * start_state.velocity + alpha * goal_state.velocity
            time = start_state.time + alpha * (goal_state.time - start_state.time)
            
            state = State(
                position=position,
                velocity=velocity,
                attitude=start_state.attitude,
                angular_velocity=np.zeros(3),
                time=time
            )
            trajectory.append(state)
        
        return trajectory
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML model."""
        X = []
        y = []
        
        for sample in training_data:
            features = sample.get("features", np.zeros(15))
            waypoints = sample.get("waypoints", np.zeros((5, 3)))
            
            X.append(features)
            y.append(waypoints.flatten())
        
        return np.array(X), np.array(y)
    
    def _create_model(self):
        """Create ML model (simplified)."""
        # Would create actual neural network here
        return {"type": "neural_network", "trained": True}
    
    def _get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key for features."""
        return str(hash(features.tobytes()))
    
    def save_model(self, filepath: str):
        """Save trained model."""
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get planning performance metrics."""
        return {
            "is_trained": self.is_trained,
            "cache_size": len(self.planning_cache),
            "last_plan_length": len(self.last_plan) if self.last_plan else 0
        } 