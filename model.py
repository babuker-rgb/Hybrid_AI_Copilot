# ================================================================
# Hybrid AI Tablet Model - Neural Network Architecture
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

class PhysicsInformedLayer(nn.Module):
    """Physics-informed constraints layer"""
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        # Ensure physical constraints
        density = torch.sigmoid(x[:, 0]) * 0.4 + 0.55
        tensile = torch.sigmoid(x[:, 1]) * 8.0 + 0.5
        efrf = torch.sigmoid(x[:, 2]) * 1.0
        disintegration = torch.sigmoid(x[:, 3]) * 45.0 + 2.0
        dissolution = torch.sigmoid(x[:, 4]) * 80.0 + 10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], dim=1)

class HybridTabletModel(nn.Module):
    """Hybrid physics-informed neural network for tablet properties"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, dropout_rate: float = 0.2):
        super().__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Output layer
        self.fc5 = nn.Linear(hidden_dim, 5)
        
        # Physics constraints
        self.physics_layer = PhysicsInformedLayer()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs
        x = torch.sigmoid(x)
        
        # Neural network layers with residual connections
        h1 = F.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout1(h1)
        
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h2 = self.dropout2(h2 + h1)  # Residual connection
        
        h3 = F.relu(self.bn3(self.fc3(h2)))
        h3 = self.dropout3(h3 + h2)  # Residual connection
        
        h4 = F.relu(self.bn4(self.fc4(h3)))
        h4 = h4 + h3  # Residual connection
        
        # Output
        out = self.fc5(h4)
        
        # Apply physics constraints
        return self.physics_layer(out)
    
    def predict_single(self, x: np.ndarray) -> Dict[str, float]:
        """Predict tablet properties for a single input"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            prediction = self.forward(x_tensor).squeeze().numpy()
            
        return {
            'density': float(prediction[0]),
            'tensile': float(prediction[1]),
            'efrf': float(prediction[2]),
            'disintegration': float(prediction[3]),
            'dissolution': float(prediction[4])
        }
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict tablet properties for batch inputs"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X)
            predictions = self.forward(x_tensor).numpy()
        return predictions

class HybridLoss(nn.Module):
    """Hybrid loss function combining MSE and physics constraints"""
    
    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Physics constraints loss
        physics_loss = self._physics_constraints(pred)
        
        # Combined loss
        return mse_loss + self.physics_weight * physics_loss
    
    def _physics_constraints(self, pred: torch.Tensor) -> torch.Tensor:
        """Enforce physical constraints"""
        density = pred[:, 0]
        tensile = pred[:, 1]
        efrf = pred[:, 2]
        disintegration = pred[:, 3]
        dissolution = pred[:, 4]
        
        # Constraint violations
        density_violation = torch.relu(0.55 - density) + torch.relu(density - 0.95)
        tensile_violation = torch.relu(0.5 - tensile) + torch.relu(tensile - 8.5)
        efrf_violation = torch.relu(efrf - 1.0)
        disintegration_violation = torch.relu(2.0 - disintegration) + torch.relu(disintegration - 47.0)
        dissolution_violation = torch.relu(10.0 - dissolution) + torch.relu(dissolution - 90.0)
        
        return (density_violation + tensile_violation + efrf_violation + 
                disintegration_violation + dissolution_violation).mean()
