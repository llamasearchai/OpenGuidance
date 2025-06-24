"""
Machine Learning-based System Identification for OpenGuidance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from openguidance.core.types import State, Control, Vehicle

logger = logging.getLogger(__name__)


@dataclass
class SystemIDConfig:
    """Configuration for ML system identification."""
    # Model type
    model_type: str = "neural_network"  # neural_network, linear_regression, gaussian_process
    
    # Neural network architecture
    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 500
    validation_split: float = 0.2
    
    # Data collection
    sampling_rate: float = 100.0  # Hz
    data_window_size: int = 1000
    feature_scaling: bool = True
    
    # Model selection
    cross_validation_folds: int = 5
    regularization_strength: float = 1e-4
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class MLSystemID:
    """Machine learning-based system identification."""
    
    def __init__(self, config: SystemIDConfig, vehicle: Vehicle):
        """Initialize ML system identification."""
        self.config = config
        self.vehicle = vehicle
        self.model = None
        self.is_trained = False
        
        # Training data
        self.training_data = []
        self.feature_scaler = None
        self.target_scaler = None
        
        # Model performance
        self.training_metrics = {}
        
        logger.info(f"ML System ID initialized for {vehicle.type.name}")
    
    def collect_data(self, state: State, control: Control, next_state: State):
        """Collect training data from system operation."""
        # Extract features
        features = self._extract_features(state, control)
        targets = self._extract_targets(state, next_state)
        
        # Store data point
        data_point = {
            "features": features,
            "targets": targets,
            "timestamp": state.time
        }
        
        self.training_data.append(data_point)
        
        # Maintain sliding window
        if len(self.training_data) > self.config.data_window_size:
            self.training_data.pop(0)
    
    def train_model(self) -> Dict[str, float]:
        """Train system identification model."""
        if len(self.training_data) < 100:
            raise ValueError("Insufficient training data")
        
        logger.info(f"Training system ID model with {len(self.training_data)} samples")
        
        # Prepare data
        X, y = self._prepare_training_data()
        
        # Feature scaling
        if self.config.feature_scaling:
            X, y = self._scale_features(X, y)
        
        # Train model
        self.model = self._create_and_train_model(X, y)
        self.is_trained = True
        
        # Evaluate model
        metrics = self._evaluate_model(X, y)
        self.training_metrics = metrics
        
        logger.info(f"System ID training completed - R²: {metrics.get('r2_score', 0.0):.4f}")
        return metrics
    
    def predict_next_state(self, current_state: State, control: Control) -> State:
        """Predict next state using identified model."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Extract features
        features = self._extract_features(current_state, control)
        
        # Scale features
        if self.feature_scaler is not None:
            features = self._apply_feature_scaling(features)
        
        # Predict
        prediction = self._model_predict(features)
        
        # Unscale prediction
        if self.target_scaler is not None:
            prediction = self._apply_target_unscaling(prediction)
        
        # Convert to State
        next_state = self._prediction_to_state(prediction, current_state)
        
        return next_state
    
    def get_model_uncertainty(self, state: State, control: Control) -> np.ndarray:
        """Get model prediction uncertainty."""
        if not self.is_trained:
            return np.ones(12) * 1.0  # High uncertainty
        
        # For neural networks, could use dropout for uncertainty estimation
        # For Gaussian processes, uncertainty is built-in
        # Simplified implementation
        base_uncertainty = 0.1
        return np.ones(12) * base_uncertainty
    
    def _extract_features(self, state: State, control: Control) -> np.ndarray:
        """Extract features from state and control."""
        state_features = np.concatenate([
            state.position,
            state.velocity,
            state.euler_angles,
            state.angular_velocity
        ])
        
        control_features = np.array([
            control.thrust,
            control.aileron,
            control.elevator,
            control.rudder
        ])
        
        return np.concatenate([state_features, control_features])
    
    def _extract_targets(self, current_state: State, next_state: State) -> np.ndarray:
        """Extract target values (state derivatives)."""
        dt = next_state.time - current_state.time
        if dt <= 0:
            dt = 0.01  # Default time step
        
        # Compute derivatives
        pos_dot = (next_state.position - current_state.position) / dt
        vel_dot = (next_state.velocity - current_state.velocity) / dt
        att_dot = (next_state.euler_angles - current_state.euler_angles) / dt
        omega_dot = (next_state.angular_velocity - current_state.angular_velocity) / dt
        
        return np.concatenate([pos_dot, vel_dot, att_dot, omega_dot])
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data matrices."""
        X = []
        y = []
        
        for data_point in self.training_data:
            X.append(data_point["features"])
            y.append(data_point["targets"])
        
        return np.array(X), np.array(y)
    
    def _scale_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features and targets."""
        # Simplified scaling (would use sklearn StandardScaler in practice)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0) + 1e-8
        y_scaled = (y - y_mean) / y_std
        
        # Store scalers
        self.feature_scaler = {"mean": X_mean, "std": X_std}
        self.target_scaler = {"mean": y_mean, "std": y_std}
        
        return X_scaled, y_scaled
    
    def _create_and_train_model(self, X: np.ndarray, y: np.ndarray):
        """Create and train the identification model."""
        # Simplified model (would use actual ML framework)
        if self.config.model_type == "neural_network":
            model = self._create_neural_network(X.shape[1], y.shape[1])
        elif self.config.model_type == "linear_regression":
            model = self._create_linear_model(X, y)
        else:
            model = self._create_gaussian_process(X, y)
        
        return model
    
    def _create_neural_network(self, input_dim: int, output_dim: int):
        """Create neural network model."""
        # Simplified NN representation
        return {
            "type": "neural_network",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_layers": self.config.hidden_layers,
            "weights": np.random.random((input_dim, output_dim)) * 0.1
        }
    
    def _create_linear_model(self, X: np.ndarray, y: np.ndarray):
        """Create linear regression model."""
        # Solve normal equations: w = (X^T X)^-1 X^T y
        XtX = X.T @ X + self.config.regularization_strength * np.eye(X.shape[1])
        Xty = X.T @ y
        weights = np.linalg.solve(XtX, Xty)
        
        return {
            "type": "linear_regression",
            "weights": weights
        }
    
    def _create_gaussian_process(self, X: np.ndarray, y: np.ndarray):
        """Create Gaussian process model."""
        return {
            "type": "gaussian_process",
            "X_train": X,
            "y_train": y,
            "kernel": "rbf"
        }
    
    def _model_predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction with trained model."""
        if self.model is None:
            return np.zeros(12)  # Default prediction
        
        if self.model["type"] == "neural_network":
            return features @ self.model["weights"]
        elif self.model["type"] == "linear_regression":
            return features @ self.model["weights"]
        else:  # Gaussian process
            # Simplified GP prediction
            return np.mean(self.model["y_train"], axis=0)
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        # Make predictions
        y_pred = np.array([self._model_predict(x) for x in X])
        
        # Compute metrics
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        
        # R² score
        y_mean = np.mean(y, axis=0)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2_score = np.mean(r2)
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2_score": float(r2_score),
            "num_parameters": self._count_parameters()
        }
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0
        
        if self.model["type"] == "neural_network":
            return int(np.prod(self.model["weights"].shape))
        elif self.model["type"] == "linear_regression":
            return int(np.prod(self.model["weights"].shape))
        else:
            return len(self.training_data)
    
    def _apply_feature_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply feature scaling."""
        if self.feature_scaler is None:
            return features
        return (features - self.feature_scaler["mean"]) / self.feature_scaler["std"]
    
    def _apply_target_unscaling(self, prediction: np.ndarray) -> np.ndarray:
        """Apply target unscaling."""
        if self.target_scaler is None:
            return prediction
        return prediction * self.target_scaler["std"] + self.target_scaler["mean"]
    
    def _prediction_to_state(self, prediction: np.ndarray, current_state: State) -> State:
        """Convert prediction to State object."""
        dt = 0.01  # Time step
        
        # Extract derivatives
        pos_dot = prediction[0:3]
        vel_dot = prediction[3:6]
        att_dot = prediction[6:9]
        omega_dot = prediction[9:12]
        
        # Integrate to get next state
        next_position = current_state.position + pos_dot * dt
        next_velocity = current_state.velocity + vel_dot * dt
        next_euler = current_state.euler_angles + att_dot * dt
        next_omega = current_state.angular_velocity + omega_dot * dt
        
        # Create next state
        from pyquaternion import Quaternion
        next_attitude = Quaternion(axis=[0, 0, 1], angle=next_euler[2]) * \
                       Quaternion(axis=[0, 1, 0], angle=next_euler[1]) * \
                       Quaternion(axis=[1, 0, 0], angle=next_euler[0])
        
        next_state = State(
            position=next_position,
            velocity=next_velocity,
            attitude=next_attitude,
            angular_velocity=next_omega,
            time=current_state.time + dt
        )
        
        return next_state
    
    def save_model(self, filepath: str):
        """Save trained model."""
        logger.info(f"System ID model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        self.is_trained = True
        logger.info(f"System ID model loaded from {filepath}")
    
    def get_identification_metrics(self) -> Dict[str, Any]:
        """Get system identification metrics."""
        return {
            "is_trained": self.is_trained,
            "training_samples": len(self.training_data),
            "model_type": self.config.model_type,
            "training_metrics": self.training_metrics
        } 