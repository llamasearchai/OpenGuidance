"""
Neural Networks module for OpenGuidance AI systems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeuralNetworkConfig:
    """Configuration for neural network architectures."""
    input_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 1
    activation: str = "relu"
    dropout_rate: float = 0.1
    batch_norm: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class NeuralStateEstimator(nn.Module):
    """Neural network for state estimation tasks."""
    
    def __init__(self, config: NeuralNetworkConfig):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"NeuralStateEstimator initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ActorNetwork(nn.Module):
    """Actor network for policy-based reinforcement learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with appropriate scaling."""
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -3e-3, 3e-3)
            nn.init.uniform_(module.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class CriticNetwork(nn.Module):
    """Critic network for value-based reinforcement learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with appropriate scaling."""
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -3e-3, 3e-3)
            nn.init.uniform_(module.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value."""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ConvolutionalStateEstimator(nn.Module):
    """Convolutional neural network for image-based state estimation."""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            # Assume input size of 84x84 (common for RL)
            dummy_input = torch.zeros(1, 3, 84, 84)
            output = self.conv_layers(dummy_input)
            return int(np.prod(output.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional and fully connected layers."""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class TransformerStateEstimator(nn.Module):
    """Transformer-based state estimator for sequence modeling."""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, output_dim: int = 64):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        
        # Use the last token's representation
        x = x[:, -1, :]
        x = self.output_projection(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class EnsembleNetwork(nn.Module):
    """Ensemble of neural networks for uncertainty quantification."""
    
    def __init__(self, network_configs: List[NeuralNetworkConfig], num_networks: int = 5):
        super().__init__()
        
        self.networks = nn.ModuleList([
            NeuralStateEstimator(config) for config in network_configs[:num_networks]
        ])
        self.num_networks = len(self.networks)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and uncertainty."""
        outputs = torch.stack([net(x) for net in self.networks], dim=0)
        
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty quantification."""
        mean, std = self.forward(x)
        
        return {
            'mean': mean,
            'std': std,
            'epistemic_uncertainty': std,
            'confidence': 1.0 / (1.0 + std)
        }


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for representation learning."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def create_mlp(input_dim: int, output_dim: int, hidden_dims: List[int], 
               activation: str = "relu", dropout: float = 0.0) -> nn.Module:
    """Create a multi-layer perceptron with specified architecture."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "gelu":
            layers.append(nn.GELU())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


def initialize_weights(module: nn.Module, method: str = "xavier"):
    """Initialize network weights using specified method."""
    if isinstance(module, nn.Linear):
        if method == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif method == "he":
            nn.init.kaiming_uniform_(module.weight)
        elif method == "normal":
            nn.init.normal_(module.weight, 0, 0.02)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        if method == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif method == "he":
            nn.init.kaiming_uniform_(module.weight)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class NetworkTrainer:
    """Utility class for training neural networks."""
    
    def __init__(self, model: nn.Module, config: NeuralNetworkConfig):
        self.model = model
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_step(self, batch_data: torch.Tensor, batch_targets: torch.Tensor) -> float:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_data = batch_data.to(self.device)
        batch_targets = batch_targets.to(self.device)
        
        outputs = self.model(batch_data)
        loss = F.mse_loss(outputs, batch_targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in data_loader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_data)
                loss = F.mse_loss(outputs, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}


# Export main classes and functions
__all__ = [
    'NeuralNetworkConfig',
    'NeuralStateEstimator',
    'ActorNetwork',
    'CriticNetwork',
    'ConvolutionalStateEstimator',
    'TransformerStateEstimator',
    'EnsembleNetwork',
    'VariationalAutoencoder',
    'NetworkTrainer',
    'create_mlp',
    'initialize_weights'
] 