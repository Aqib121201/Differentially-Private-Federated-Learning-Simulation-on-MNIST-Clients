"""
Model utilities for federated learning simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import OrderedDict
import copy

from .config import model_config, federated_config, training_config

logger = logging.getLogger(__name__)

class MNISTNet(nn.Module):
    """Neural network for MNIST classification"""
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = None, 
                 num_classes: int = 10, dropout_rate: float = 0.2):
        super(MNISTNet, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)

def create_model() -> MNISTNet:
    """Create a new model instance"""
    model = MNISTNet(
        input_size=model_config.input_size,
        hidden_sizes=model_config.hidden_sizes,
        num_classes=model_config.num_classes,
        dropout_rate=model_config.dropout_rate
    )
    
    # Move to device
    device = torch.device(training_config.device)
    model = model.to(device)
    
    return model

def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """Create optimizer for the model"""
    return optim.SGD(
        model.parameters(),
        lr=federated_config.learning_rate,
        momentum=federated_config.momentum,
        weight_decay=federated_config.weight_decay
    )

def get_criterion() -> nn.Module:
    """Get loss function"""
    return nn.CrossEntropyLoss()

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float, Dict]:
    """
    Evaluate model on dataset
    
    Args:
        model: Neural network model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, accuracy, detailed_metrics)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # For detailed metrics
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Store predictions and targets for detailed metrics
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(all_targets, all_predictions)
    
    return avg_loss, accuracy, detailed_metrics

def calculate_detailed_metrics(targets: List[int], predictions: List[int]) -> Dict:
    """
    Calculate detailed classification metrics
    
    Args:
        targets: True labels
        predictions: Predicted labels
        
    Returns:
        Dictionary with detailed metrics
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    # Basic metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'confusion_matrix': cm.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist()
    }
    
    return metrics

def get_model_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Get model parameters as a list of tensors"""
    return [param.data.clone() for param in model.parameters()]

def set_model_parameters(model: nn.Module, parameters: List[torch.Tensor]):
    """Set model parameters from a list of tensors"""
    for param, new_param in zip(model.parameters(), parameters):
        param.data = new_param.data.clone()

def aggregate_parameters(parameters_list: List[List[torch.Tensor]], 
                        aggregation_method: str = "fedavg") -> List[torch.Tensor]:
    """
    Aggregate model parameters from multiple clients
    
    Args:
        parameters_list: List of parameter lists from different clients
        aggregation_method: Method to use for aggregation ("fedavg", "fedprox", etc.)
        
    Returns:
        Aggregated parameters
    """
    if aggregation_method == "fedavg":
        return _fedavg_aggregation(parameters_list)
    elif aggregation_method == "fedprox":
        return _fedprox_aggregation(parameters_list)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

def _fedavg_aggregation(parameters_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Federated Averaging (FedAvg) aggregation"""
    num_clients = len(parameters_list)
    
    # Initialize aggregated parameters
    aggregated_params = []
    for param in parameters_list[0]:
        aggregated_params.append(torch.zeros_like(param))
    
    # Average parameters
    for client_params in parameters_list:
        for i, param in enumerate(client_params):
            aggregated_params[i] += param / num_clients
    
    return aggregated_params

def _fedprox_aggregation(parameters_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """FedProx aggregation (same as FedAvg for now, can be extended)"""
    return _fedavg_aggregation(parameters_list)

def compute_parameter_difference(params1: List[torch.Tensor], 
                               params2: List[torch.Tensor]) -> float:
    """
    Compute the L2 norm difference between two parameter sets
    
    Args:
        params1: First parameter set
        params2: Second parameter set
        
    Returns:
        L2 norm of the difference
    """
    total_diff = 0.0
    
    for p1, p2 in zip(params1, params2):
        diff = p1 - p2
        total_diff += torch.sum(diff ** 2).item()
    
    return np.sqrt(total_diff)

def save_model(model: nn.Module, filepath: str):
    """Save model to file"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model_config.input_size,
            'hidden_sizes': model_config.hidden_sizes,
            'num_classes': model_config.num_classes,
            'dropout_rate': model_config.dropout_rate
        }
    }, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: str, device: torch.device = None) -> MNISTNet:
    """Load model from file"""
    if device is None:
        device = torch.device(training_config.device)
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model with saved configuration
    model_config_dict = checkpoint['model_config']
    model = MNISTNet(
        input_size=model_config_dict['input_size'],
        hidden_sizes=model_config_dict['hidden_sizes'],
        num_classes=model_config_dict['num_classes'],
        dropout_rate=model_config_dict['dropout_rate']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded from {filepath}")
    return model

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def create_model_summary(model: nn.Module) -> Dict:
    """Create a summary of the model"""
    return {
        'total_parameters': count_parameters(model),
        'model_size_mb': get_model_size_mb(model),
        'architecture': str(model),
        'input_size': model_config.input_size,
        'hidden_sizes': model_config.hidden_sizes,
        'num_classes': model_config.num_classes,
        'dropout_rate': model_config.dropout_rate
    } 