"""
Data preprocessing module for MNIST federated learning simulation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

from .config import data_config, federated_config, training_config

logger = logging.getLogger(__name__)

class MNISTDataset(Dataset):
    """Custom MNIST dataset wrapper"""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

def get_transforms(augment: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Base transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = base_transform
    
    val_transform = base_transform
    
    return train_transform, val_transform

def load_mnist_data() -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Load MNIST dataset
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_transform, test_transform = get_transforms(data_config.augment)
    
    train_dataset = datasets.MNIST(
        root=data_config.raw_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_config.raw_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    logger.info(f"Loaded MNIST dataset: {len(train_dataset)} training, {len(test_dataset)} test samples")
    return train_dataset, test_dataset

def split_data_iid(dataset: Dataset, num_clients: int, seed: int = None) -> List[Dataset]:
    """
    Split dataset in IID fashion across clients
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        seed: Random seed for reproducibility
        
    Returns:
        List of datasets for each client
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Calculate samples per client
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    
    # Create splits
    splits = [samples_per_client] * num_clients
    # Distribute remaining samples
    splits[0] += num_samples - sum(splits)
    
    client_datasets = random_split(dataset, splits)
    
    logger.info(f"Split {num_samples} samples IID across {num_clients} clients")
    for i, client_dataset in enumerate(client_datasets):
        logger.info(f"Client {i}: {len(client_dataset)} samples")
    
    return client_datasets

def split_data_non_iid(dataset: Dataset, num_clients: int, alpha: float = 0.5, seed: int = None) -> List[Dataset]:
    """
    Split dataset in non-IID fashion using Dirichlet distribution
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        alpha: Dirichlet distribution parameter (smaller = more non-IID)
        seed: Random seed for reproducibility
        
    Returns:
        List of datasets for each client
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Get labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    
    # Create label indices
    label_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    # Sample from Dirichlet distribution
    client_data_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        np.random.shuffle(label_indices[k])
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = np.array([p * len(label_indices[k]) for p in proportions])
        
        for client_id, prop in enumerate(proportions):
            size = int(prop)
            client_data_indices[client_id].extend(label_indices[k][:size])
            label_indices[k] = label_indices[k][size:]
    
    # Create client datasets
    client_datasets = []
    for client_id in range(num_clients):
        indices = client_data_indices[client_id]
        client_dataset = torch.utils.data.Subset(dataset, indices)
        client_datasets.append(client_dataset)
        
        # Log class distribution
        client_labels = [dataset[i][1] for i in indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        logger.info(f"Client {client_id}: {len(indices)} samples, class distribution: {dict(zip(unique, counts))}")
    
    return client_datasets

def create_client_dataloaders(client_datasets: List[Dataset], batch_size: int) -> List[DataLoader]:
    """
    Create DataLoaders for each client
    
    Args:
        client_datasets: List of client datasets
        batch_size: Batch size for training
        
    Returns:
        List of DataLoaders
    """
    client_dataloaders = []
    
    for client_id, dataset in enumerate(client_datasets):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory
        )
        client_dataloaders.append(dataloader)
        
        logger.info(f"Client {client_id} dataloader: {len(dataloader)} batches")
    
    return client_dataloaders

def prepare_federated_data() -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Prepare data for federated learning simulation
    
    Returns:
        Tuple of (client_dataloaders, val_dataloader, test_dataloader)
    """
    # Load datasets
    train_dataset, test_dataset = load_mnist_data()
    
    # Split training data into train and validation
    train_size = int(data_config.train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(training_config.seed)
    )
    
    # Split training data across clients
    if federated_config.data_distribution == "iid":
        client_datasets = split_data_iid(
            train_subset, 
            federated_config.num_clients, 
            training_config.seed
        )
    else:
        client_datasets = split_data_non_iid(
            train_subset,
            federated_config.num_clients,
            federated_config.non_iid_alpha,
            training_config.seed
        )
    
    # Create dataloaders
    client_dataloaders = create_client_dataloaders(
        client_datasets, 
        federated_config.batch_size
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=federated_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=federated_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    logger.info(f"Prepared federated data: {len(client_dataloaders)} clients, "
                f"{len(val_dataloader)} val batches, {len(test_dataloader)} test batches")
    
    return client_dataloaders, val_dataloader, test_dataloader

def get_centralized_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for centralized training (baseline)
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Load datasets
    train_dataset, test_dataset = load_mnist_data()
    
    # Split training data
    train_size = int(data_config.train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(training_config.seed)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=federated_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=federated_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=federated_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    logger.info(f"Prepared centralized data: {len(train_dataloader)} train batches, "
                f"{len(val_dataloader)} val batches, {len(test_dataloader)} test batches")
    
    return train_dataloader, val_dataloader, test_dataloader

def analyze_data_distribution(client_datasets: List[Dataset]) -> Dict:
    """
    Analyze the distribution of data across clients
    
    Args:
        client_datasets: List of client datasets
        
    Returns:
        Dictionary with distribution statistics
    """
    distribution_stats = {
        "num_clients": len(client_datasets),
        "client_sizes": [],
        "class_distributions": [],
        "total_samples": 0
    }
    
    for client_id, dataset in enumerate(client_datasets):
        # Get client data
        client_data = [dataset[i] for i in range(len(dataset))]
        client_labels = [data[1] for data in client_data]
        
        # Calculate statistics
        client_size = len(client_data)
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        class_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        distribution_stats["client_sizes"].append(client_size)
        distribution_stats["class_distributions"].append(class_dist)
        distribution_stats["total_samples"] += client_size
        
        logger.info(f"Client {client_id}: {client_size} samples, classes: {class_dist}")
    
    # Calculate additional statistics
    distribution_stats["avg_client_size"] = np.mean(distribution_stats["client_sizes"])
    distribution_stats["std_client_size"] = np.std(distribution_stats["client_sizes"])
    distribution_stats["min_client_size"] = min(distribution_stats["client_sizes"])
    distribution_stats["max_client_size"] = max(distribution_stats["client_sizes"])
    
    return distribution_stats 