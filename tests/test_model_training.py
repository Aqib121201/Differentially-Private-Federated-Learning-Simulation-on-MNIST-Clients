"""
Unit tests for model training module
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model_utils import (
    MNISTNet, create_model, get_optimizer, get_criterion,
    train_epoch, evaluate_model, get_model_parameters,
    set_model_parameters, aggregate_parameters, count_parameters
)
from src.config import model_config, federated_config, training_config

class TestModelUtils:
    """Test class for model utility functions"""
    
    def test_mnist_net_creation(self):
        """Test MNISTNet model creation"""
        model = MNISTNet()
        
        # Test model structure
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'network')
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
        output = model(x)
        
        assert output.shape == (2, 10)  # 2 samples, 10 classes
        assert not torch.isnan(output).any()
    
    def test_create_model(self):
        """Test model creation utility"""
        model = create_model()
        
        assert isinstance(model, MNISTNet)
        assert model.network is not None
        
        # Test device placement
        expected_device = torch.device(training_config.device)
        assert next(model.parameters()).device == expected_device
    
    def test_get_optimizer(self):
        """Test optimizer creation"""
        model = create_model()
        optimizer = get_optimizer(model)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['lr'] == federated_config.learning_rate
        assert optimizer.param_groups[0]['momentum'] == federated_config.momentum
    
    def test_get_criterion(self):
        """Test loss function creation"""
        criterion = get_criterion()
        
        assert isinstance(criterion, nn.CrossEntropyLoss)
    
    def test_model_parameters(self):
        """Test parameter extraction and setting"""
        model = create_model()
        
        # Get parameters
        params = get_model_parameters(model)
        
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)
        
        # Create new model and set parameters
        new_model = create_model()
        set_model_parameters(new_model, params)
        
        # Check that parameters are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_aggregate_parameters(self):
        """Test parameter aggregation"""
        model = create_model()
        params = get_model_parameters(model)
        
        # Create multiple parameter sets (simulating multiple clients)
        param_sets = [params, params, params]  # 3 clients with same parameters
        
        # Aggregate
        aggregated = aggregate_parameters(param_sets, "fedavg")
        
        assert isinstance(aggregated, list)
        assert len(aggregated) == len(params)
        
        # Check that aggregated parameters are the same as original (since all clients had same params)
        for p1, p2 in zip(params, aggregated):
            assert torch.allclose(p1, p2)
    
    def test_count_parameters(self):
        """Test parameter counting"""
        model = create_model()
        num_params = count_parameters(model)
        
        assert isinstance(num_params, int)
        assert num_params > 0
        
        # Verify count manually
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == manual_count

class TestTrainingFunctions:
    """Test class for training functions"""
    
    def test_train_epoch(self):
        """Test training epoch function"""
        # Create mock dataloader
        class MockDataLoader:
            def __init__(self, num_batches=5):
                self.num_batches = num_batches
            
            def __len__(self):
                return self.num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    # Mock batch: (data, targets)
                    data = torch.randn(4, 1, 28, 28)  # 4 samples
                    targets = torch.randint(0, 10, (4,))  # 4 labels
                    yield data, targets
        
        model = create_model()
        dataloader = MockDataLoader()
        optimizer = get_optimizer(model)
        criterion = get_criterion()
        device = torch.device(training_config.device)
        
        # Test training epoch
        loss, accuracy = train_epoch(model, dataloader, optimizer, criterion, device)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss >= 0
        assert 0 <= accuracy <= 100
    
    def test_evaluate_model(self):
        """Test model evaluation function"""
        # Create mock dataloader
        class MockDataLoader:
            def __init__(self, num_batches=3):
                self.num_batches = num_batches
            
            def __len__(self):
                return self.num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    data = torch.randn(4, 1, 28, 28)
                    targets = torch.randint(0, 10, (4,))
                    yield data, targets
        
        model = create_model()
        dataloader = MockDataLoader()
        criterion = get_criterion()
        device = torch.device(training_config.device)
        
        # Test evaluation
        loss, accuracy, metrics = evaluate_model(model, dataloader, criterion, device)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert isinstance(metrics, dict)
        assert loss >= 0
        assert 0 <= accuracy <= 100
        
        # Check metrics structure
        expected_keys = ['precision', 'recall', 'f1_score', 'confusion_matrix']
        for key in expected_keys:
            assert key in metrics

class TestModelPersistence:
    """Test class for model saving and loading"""
    
    def test_save_and_load_model(self, tmp_path):
        """Test model saving and loading"""
        from src.model_utils import save_model, load_model
        
        # Create model
        model = create_model()
        
        # Save model
        save_path = tmp_path / "test_model.pth"
        save_model(model, str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = load_model(str(save_path))
        
        assert isinstance(loaded_model, MNISTNet)
        
        # Check that parameters are the same
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

class TestModelConfigurations:
    """Test class for different model configurations"""
    
    def test_model_with_different_hidden_sizes(self):
        """Test model with different hidden layer sizes"""
        hidden_sizes = [256, 128, 64]
        model = MNISTNet(hidden_sizes=hidden_sizes)
        
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (1, 10)
    
    def test_model_with_different_dropout(self):
        """Test model with different dropout rates"""
        model = MNISTNet(dropout_rate=0.5)
        
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (1, 10)
    
    def test_model_with_different_input_size(self):
        """Test model with different input sizes"""
        model = MNISTNet(input_size=1024, hidden_sizes=[512, 256])
        
        x = torch.randn(1, 1024)  # Flattened input
        output = model(x)
        
        assert output.shape == (1, 10)

if __name__ == "__main__":
    pytest.main([__file__]) 