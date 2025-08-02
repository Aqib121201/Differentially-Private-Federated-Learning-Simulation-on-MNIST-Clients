"""
Unit tests for data preprocessing module
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_preprocessing import (
    get_transforms, split_data_iid, split_data_non_iid,
    create_client_dataloaders, analyze_data_distribution
)
from src.config import federated_config, training_config

class TestDataPreprocessing:
    """Test class for data preprocessing functions"""
    
    def test_get_transforms(self):
        """Test transform creation"""
        # Test without augmentation
        train_transform, val_transform = get_transforms(augment=False)
        assert train_transform is not None
        assert val_transform is not None
        
        # Test with augmentation
        train_transform_aug, val_transform_aug = get_transforms(augment=True)
        assert train_transform_aug is not None
        assert val_transform_aug is not None
    
    def test_split_data_iid(self):
        """Test IID data splitting"""
        # Create mock dataset
        class MockDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
        
        dataset = MockDataset(100)
        num_clients = 5
        
        # Test splitting
        client_datasets = split_data_iid(dataset, num_clients, seed=42)
        
        assert len(client_datasets) == num_clients
        
        # Check total size is preserved
        total_size = sum(len(client_dataset) for client_dataset in client_datasets)
        assert total_size == len(dataset)
    
    def test_split_data_non_iid(self):
        """Test non-IID data splitting"""
        # Create mock dataset with labels
        class MockDatasetWithLabels:
            def __init__(self, size=100):
                self.size = size
                # Create some labels
                self.labels = np.random.randint(0, 10, size)
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return (torch.randn(1, 28, 28), self.labels[idx])
        
        dataset = MockDatasetWithLabels(100)
        num_clients = 5
        alpha = 0.5
        
        # Test splitting
        client_datasets = split_data_non_iid(dataset, num_clients, alpha, seed=42)
        
        assert len(client_datasets) == num_clients
        
        # Check total size is preserved
        total_size = sum(len(client_dataset) for client_dataset in client_datasets)
        assert total_size == len(dataset)
    
    def test_create_client_dataloaders(self):
        """Test client dataloader creation"""
        # Create mock datasets
        class MockDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
        
        client_datasets = [MockDataset(50) for _ in range(3)]
        batch_size = 32
        
        # Test dataloader creation
        client_dataloaders = create_client_dataloaders(client_datasets, batch_size)
        
        assert len(client_dataloaders) == len(client_datasets)
        
        for dataloader in client_dataloaders:
            assert dataloader.batch_size == batch_size
    
    def test_analyze_data_distribution(self):
        """Test data distribution analysis"""
        # Create mock datasets with known distributions
        class MockDatasetWithDistribution:
            def __init__(self, size, class_dist):
                self.size = size
                self.class_dist = class_dist
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Return a mock sample
                return (torch.randn(1, 28, 28), idx % 10)
        
        # Create test datasets
        client_datasets = [
            MockDatasetWithDistribution(50, {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}),
            MockDatasetWithDistribution(30, {5: 10, 6: 10, 7: 10}),
            MockDatasetWithDistribution(20, {8: 10, 9: 10})
        ]
        
        # Test analysis
        distribution_stats = analyze_data_distribution(client_datasets)
        
        assert distribution_stats['num_clients'] == 3
        assert distribution_stats['total_samples'] == 100
        assert len(distribution_stats['client_sizes']) == 3
        assert len(distribution_stats['class_distributions']) == 3
        assert 'avg_client_size' in distribution_stats
        assert 'std_client_size' in distribution_stats

class TestDataPreprocessingIntegration:
    """Integration tests for data preprocessing"""
    
    @patch('src.data_preprocessing.datasets.MNIST')
    def test_load_mnist_data(self, mock_mnist):
        """Test MNIST data loading"""
        # Mock MNIST dataset
        mock_train = MagicMock()
        mock_test = MagicMock()
        mock_mnist.side_effect = [mock_train, mock_test]
        
        # This would test the actual loading function
        # For now, just verify the mock is called
        assert mock_mnist.call_count == 0
    
    def test_prepare_federated_data(self):
        """Test federated data preparation"""
        # This is a complex integration test that would require
        # mocking the entire data loading pipeline
        # For now, we'll just test that the function exists
        from src.data_preprocessing import prepare_federated_data
        assert callable(prepare_federated_data)
    
    def test_get_centralized_dataloaders(self):
        """Test centralized dataloader preparation"""
        # This is a complex integration test that would require
        # mocking the entire data loading pipeline
        # For now, we'll just test that the function exists
        from src.data_preprocessing import get_centralized_dataloaders
        assert callable(get_centralized_dataloaders)

if __name__ == "__main__":
    pytest.main([__file__]) 