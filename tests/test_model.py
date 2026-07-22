# ================================================================
# Unit Tests for Hybrid AI Model
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

import unittest
import numpy as np
import torch
from model import HybridTabletModel, HybridLoss

class TestHybridModel(unittest.TestCase):
    
    def setUp(self):
        self.model = HybridTabletModel(input_dim=8, hidden_dim=64)
        self.loss_fn = HybridLoss(physics_weight=0.1)
        
    def test_model_forward(self):
        """Test forward pass produces correct output shape"""
        batch_size = 10
        x = torch.randn(batch_size, 8)
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size, 5))
        
    def test_physics_constraints(self):
        """Test physics constraints are satisfied"""
        x = torch.randn(5, 8)
        output = self.model(x)
        densities = output[:, 0]
        self.assertTrue(torch.all(densities >= 0.55) and torch.all(densities <= 0.95))
        
    def test_loss_function(self):
        """Test hybrid loss function"""
        pred = torch.randn(5, 5)
        target = torch.randn(5, 5)
        loss = self.loss_fn(pred, target)
        self.assertTrue(loss >= 0)

class TestOptimizer(unittest.TestCase):
    
    def setUp(self):
        self.model = HybridTabletModel(input_dim=8, hidden_dim=64)
        
    def test_optimization(self):
        """Test optimization runs without errors"""
        # This is a placeholder for actual optimization tests
        pass

if __name__ == '__main__':
    unittest.main()
