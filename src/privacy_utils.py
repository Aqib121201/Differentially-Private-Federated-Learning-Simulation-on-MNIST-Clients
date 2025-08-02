"""
Differential privacy utilities for federated learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.utils.batch_memory_manager import BatchMemoryManager
import warnings

from .config import privacy_config, federated_config, training_config

logger = logging.getLogger(__name__)

class PrivacyManager:
    """Manages differential privacy for federated learning"""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, 
                 noise_multiplier: float = None, max_grad_norm: float = None):
        """
        Initialize privacy manager
        
        Args:
            model: Neural network model
            dataloader: Training data loader
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.dataloader = dataloader
        
        # Use config values if not provided
        self.noise_multiplier = noise_multiplier or privacy_config.noise_multiplier
        self.max_grad_norm = max_grad_norm or privacy_config.max_grad_norm
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()
        self.accountant = RDPAccountant()
        
        # Privacy budget tracking
        self.epsilon = 0.0
        self.delta = privacy_config.delta
        self.alpha = privacy_config.alphas
        
        # Initialize privacy engine
        self._setup_privacy_engine()
    
    def _setup_privacy_engine(self):
        """Setup Opacus privacy engine"""
        try:
            self.privacy_engine = PrivacyEngine()
            
            # Make model compatible with Opacus
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=None,  # Will be set during training
                data_loader=self.dataloader,
                epochs=federated_config.local_epochs,
                target_epsilon=privacy_config.target_epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=False
            )
            
            logger.info(f"Privacy engine initialized with noise_multiplier={self.noise_multiplier}, "
                       f"max_grad_norm={self.max_grad_norm}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize privacy engine: {e}")
            logger.info("Continuing without differential privacy")
            self.privacy_engine = None
    
    def attach_optimizer(self, optimizer):
        """Attach optimizer to privacy engine"""
        if self.privacy_engine is not None:
            self.privacy_engine.attach(optimizer)
            logger.info("Optimizer attached to privacy engine")
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get current privacy budget spent"""
        if self.privacy_engine is not None:
            epsilon, delta = self.privacy_engine.get_privacy_spent(self.delta)
            return {
                'epsilon': epsilon,
                'delta': delta,
                'alpha': self.alpha
            }
        else:
            return {
                'epsilon': 0.0,
                'delta': 0.0,
                'alpha': self.alpha
            }
    
    def step(self):
        """Step the privacy accountant"""
        if self.privacy_engine is not None:
            self.privacy_engine.step()
    
    def get_noise_multiplier(self) -> float:
        """Get current noise multiplier"""
        if self.privacy_engine is not None:
            return self.privacy_engine.noise_multiplier
        return 0.0

def add_noise_to_gradients(model: nn.Module, noise_multiplier: float, 
                          max_grad_norm: float) -> None:
    """
    Add differential privacy noise to model gradients
    
    Args:
        model: Neural network model
        noise_multiplier: Noise multiplier for DP-SGD
        max_grad_norm: Maximum gradient norm for clipping
    """
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Add noise to gradients
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier * max_grad_norm
            param.grad += noise

def compute_epsilon_given_delta(sample_rate: float, noise_multiplier: float, 
                              num_steps: int, delta: float) -> float:
    """
    Compute epsilon given delta using RDP accountant
    
    Args:
        sample_rate: Sampling rate
        noise_multiplier: Noise multiplier
        num_steps: Number of training steps
        delta: Target delta
        
    Returns:
        Computed epsilon value
    """
    accountant = RDPAccountant()
    
    # Add composition
    for _ in range(num_steps):
        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate
        )
    
    # Get epsilon
    epsilon, _ = accountant.get_privacy_spent(delta=delta)
    return epsilon

def compute_noise_multiplier_given_epsilon(sample_rate: float, target_epsilon: float,
                                         num_steps: int, delta: float) -> float:
    """
    Compute noise multiplier given target epsilon
    
    Args:
        sample_rate: Sampling rate
        target_epsilon: Target epsilon value
        num_steps: Number of training steps
        delta: Target delta
        
    Returns:
        Computed noise multiplier
    """
    # Binary search for noise multiplier
    left, right = 0.1, 100.0
    tolerance = 0.01
    
    while right - left > tolerance:
        mid = (left + right) / 2
        epsilon = compute_epsilon_given_delta(sample_rate, mid, num_steps, delta)
        
        if epsilon > target_epsilon:
            left = mid
        else:
            right = mid
    
    return right

def create_privacy_report(privacy_manager: PrivacyManager, 
                         num_rounds: int, num_clients: int) -> Dict:
    """
    Create a comprehensive privacy report
    
    Args:
        privacy_manager: Privacy manager instance
        num_rounds: Number of federated rounds
        num_clients: Number of clients
        
    Returns:
        Privacy report dictionary
    """
    privacy_spent = privacy_manager.get_privacy_spent()
    
    report = {
        'total_epsilon': privacy_spent['epsilon'],
        'total_delta': privacy_spent['delta'],
        'noise_multiplier': privacy_manager.get_noise_multiplier(),
        'max_grad_norm': privacy_manager.max_grad_norm,
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'total_steps': num_rounds * federated_config.local_epochs,
        'sample_rate': federated_config.batch_size / len(privacy_manager.dataloader.dataset),
        'privacy_guarantee': f"({privacy_spent['epsilon']:.2f}, {privacy_spent['delta']:.2e})-DP"
    }
    
    return report

def analyze_privacy_utility_tradeoff(noise_multipliers: List[float], 
                                   accuracies: List[float],
                                   epsilons: List[float]) -> Dict:
    """
    Analyze the privacy-utility tradeoff
    
    Args:
        noise_multipliers: List of noise multipliers
        accuracies: List of corresponding accuracies
        epsilons: List of corresponding epsilon values
        
    Returns:
        Analysis results
    """
    if len(noise_multipliers) != len(accuracies) or len(noise_multipliers) != len(epsilons):
        raise ValueError("All input lists must have the same length")
    
    # Calculate correlations
    noise_acc_corr = np.corrcoef(noise_multipliers, accuracies)[0, 1]
    epsilon_acc_corr = np.corrcoef(epsilons, accuracies)[0, 1]
    
    # Find optimal points
    max_accuracy_idx = np.argmax(accuracies)
    min_epsilon_idx = np.argmin(epsilons)
    
    analysis = {
        'noise_accuracy_correlation': noise_acc_corr,
        'epsilon_accuracy_correlation': epsilon_acc_corr,
        'best_accuracy': {
            'value': accuracies[max_accuracy_idx],
            'noise_multiplier': noise_multipliers[max_accuracy_idx],
            'epsilon': epsilons[max_accuracy_idx]
        },
        'best_privacy': {
            'epsilon': epsilons[min_epsilon_idx],
            'accuracy': accuracies[min_epsilon_idx],
            'noise_multiplier': noise_multipliers[min_epsilon_idx]
        },
        'privacy_utility_ratio': accuracies[max_accuracy_idx] / epsilons[max_accuracy_idx]
    }
    
    return analysis

def validate_privacy_parameters(noise_multiplier: float, max_grad_norm: float, 
                              batch_size: int, dataset_size: int) -> bool:
    """
    Validate privacy parameters
    
    Args:
        noise_multiplier: Noise multiplier
        max_grad_norm: Maximum gradient norm
        batch_size: Batch size
        dataset_size: Size of dataset
        
    Returns:
        True if parameters are valid
    """
    sample_rate = batch_size / dataset_size
    
    # Check if sample rate is reasonable
    if sample_rate > 1.0:
        logger.warning(f"Sample rate {sample_rate:.3f} > 1.0")
        return False
    
    # Check noise multiplier
    if noise_multiplier <= 0:
        logger.warning(f"Noise multiplier {noise_multiplier} <= 0")
        return False
    
    # Check gradient norm
    if max_grad_norm <= 0:
        logger.warning(f"Max gradient norm {max_grad_norm} <= 0")
        return False
    
    # Estimate epsilon for validation
    try:
        estimated_epsilon = compute_epsilon_given_delta(
            sample_rate, noise_multiplier, 1, privacy_config.delta
        )
        logger.info(f"Estimated epsilon per step: {estimated_epsilon:.4f}")
    except Exception as e:
        logger.warning(f"Could not estimate epsilon: {e}")
        return False
    
    return True

def create_privacy_config_summary() -> Dict:
    """Create a summary of privacy configuration"""
    return {
        'enable_dp': privacy_config.enable_dp,
        'noise_multiplier': privacy_config.noise_multiplier,
        'max_grad_norm': privacy_config.max_grad_norm,
        'delta': privacy_config.delta,
        'target_epsilon': privacy_config.target_epsilon,
        'accountant_type': privacy_config.accountant_type,
        'num_alphas': len(privacy_config.alphas)
    } 