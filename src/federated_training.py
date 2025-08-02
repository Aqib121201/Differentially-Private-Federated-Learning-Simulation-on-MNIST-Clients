"""
Federated learning training module with differential privacy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
import time
from collections import defaultdict
import json
from pathlib import Path

from .config import federated_config, privacy_config, training_config
from .model_utils import (
    create_model, get_optimizer, get_criterion, train_epoch, evaluate_model,
    get_model_parameters, set_model_parameters, aggregate_parameters,
    save_model, create_model_summary
)
from .privacy_utils import PrivacyManager, create_privacy_report
from .data_preprocessing import prepare_federated_data, get_centralized_dataloaders

logger = logging.getLogger(__name__)

class FederatedClient:
    """Represents a client in federated learning"""
    
    def __init__(self, client_id: int, dataloader: DataLoader, 
                 enable_dp: bool = True):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            dataloader: Client's local data
            enable_dp: Whether to enable differential privacy
        """
        self.client_id = client_id
        self.dataloader = dataloader
        self.enable_dp = enable_dp
        
        # Initialize model and training components
        self.model = create_model()
        self.optimizer = get_optimizer(self.model)
        self.criterion = get_criterion()
        self.device = torch.device(training_config.device)
        
        # Privacy manager
        self.privacy_manager = None
        if enable_dp and privacy_config.enable_dp:
            self.privacy_manager = PrivacyManager(
                self.model, self.dataloader,
                privacy_config.noise_multiplier,
                privacy_config.max_grad_norm
            )
            self.privacy_manager.attach_optimizer(self.optimizer)
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'privacy_spent': []
        }
        
        logger.info(f"Initialized client {client_id} with {len(dataloader)} batches")
    
    def train_local(self, global_parameters: List[torch.Tensor]) -> Dict:
        """
        Train model locally for specified number of epochs
        
        Args:
            global_parameters: Global model parameters from server
            
        Returns:
            Training results dictionary
        """
        # Set global parameters
        set_model_parameters(self.model, global_parameters)
        
        # Training loop
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(federated_config.local_epochs):
            loss, accuracy = train_epoch(
                self.model, self.dataloader, self.optimizer, 
                self.criterion, self.device
            )
            
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
            
            # Step privacy accountant
            if self.privacy_manager:
                self.privacy_manager.step()
            
            logger.debug(f"Client {self.client_id}, Epoch {epoch+1}: "
                        f"Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
        
        # Get updated parameters
        updated_parameters = get_model_parameters(self.model)
        
        # Get privacy spent
        privacy_spent = {'epsilon': 0.0, 'delta': 0.0}
        if self.privacy_manager:
            privacy_spent = self.privacy_manager.get_privacy_spent()
        
        # Update training history
        self.training_history['loss'].extend(epoch_losses)
        self.training_history['accuracy'].extend(epoch_accuracies)
        self.training_history['privacy_spent'].append(privacy_spent)
        
        results = {
            'client_id': self.client_id,
            'parameters': updated_parameters,
            'num_samples': len(self.dataloader.dataset),
            'avg_loss': np.mean(epoch_losses),
            'avg_accuracy': np.mean(epoch_accuracies),
            'privacy_spent': privacy_spent,
            'training_time': time.time()
        }
        
        return results
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model on given dataloader
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Evaluation results
        """
        loss, accuracy, detailed_metrics = evaluate_model(
            self.model, dataloader, self.criterion, self.device
        )
        
        return {
            'client_id': self.client_id,
            'loss': loss,
            'accuracy': accuracy,
            'detailed_metrics': detailed_metrics
        }

class FederatedServer:
    """Central server for federated learning coordination"""
    
    def __init__(self, val_dataloader: DataLoader, test_dataloader: DataLoader,
                 aggregation_method: str = "fedavg"):
        """
        Initialize federated server
        
        Args:
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            aggregation_method: Method for aggregating client updates
        """
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.aggregation_method = aggregation_method
        
        # Global model
        self.global_model = create_model()
        self.criterion = get_criterion()
        self.device = torch.device(training_config.device)
        
        # Training history
        self.training_history = {
            'round': [],
            'val_loss': [],
            'val_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'privacy_budget': [],
            'communication_rounds': []
        }
        
        # Model summary
        self.model_summary = create_model_summary(self.global_model)
        
        logger.info(f"Initialized federated server with {self.aggregation_method} aggregation")
        logger.info(f"Global model: {self.model_summary['total_parameters']} parameters")
    
    def aggregate_updates(self, client_results: List[Dict]) -> List[torch.Tensor]:
        """
        Aggregate client updates
        
        Args:
            client_results: List of client training results
            
        Returns:
            Aggregated parameters
        """
        parameters_list = [result['parameters'] for result in client_results]
        aggregated_params = aggregate_parameters(parameters_list, self.aggregation_method)
        
        # Update global model
        set_model_parameters(self.global_model, aggregated_params)
        
        return aggregated_params
    
    def evaluate_global_model(self) -> Dict:
        """
        Evaluate global model on validation and test sets
        
        Returns:
            Evaluation results
        """
        # Validation evaluation
        val_loss, val_accuracy, val_metrics = evaluate_model(
            self.global_model, self.val_dataloader, self.criterion, self.device
        )
        
        # Test evaluation
        test_loss, test_accuracy, test_metrics = evaluate_model(
            self.global_model, self.test_dataloader, self.criterion, self.device
        )
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_metrics': val_metrics,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_metrics': test_metrics
        }
    
    def save_model(self, filepath: str):
        """Save global model"""
        save_model(self.global_model, filepath)
    
    def get_global_parameters(self) -> List[torch.Tensor]:
        """Get global model parameters"""
        return get_model_parameters(self.global_model)

class FederatedLearningSimulator:
    """Main federated learning simulator"""
    
    def __init__(self, enable_dp: bool = True, aggregation_method: str = "fedavg"):
        """
        Initialize federated learning simulator
        
        Args:
            enable_dp: Whether to enable differential privacy
            aggregation_method: Method for aggregating client updates
        """
        self.enable_dp = enable_dp
        self.aggregation_method = aggregation_method
        
        # Prepare data
        self.client_dataloaders, self.val_dataloader, self.test_dataloader = prepare_federated_data()
        
        # Initialize clients
        self.clients = []
        for client_id, dataloader in enumerate(self.client_dataloaders):
            client = FederatedClient(client_id, dataloader, enable_dp)
            self.clients.append(client)
        
        # Initialize server
        self.server = FederatedServer(self.val_dataloader, self.test_dataloader, aggregation_method)
        
        # Training metrics
        self.metrics = {
            'round_metrics': [],
            'client_metrics': defaultdict(list),
            'privacy_metrics': [],
            'communication_metrics': []
        }
        
        logger.info(f"Initialized federated learning simulator with {len(self.clients)} clients")
        logger.info(f"Differential privacy: {'enabled' if enable_dp else 'disabled'}")
    
    def run_federated_training(self, num_rounds: int = None) -> Dict:
        """
        Run federated learning training
        
        Args:
            num_rounds: Number of training rounds (uses config if None)
            
        Returns:
            Training results
        """
        if num_rounds is None:
            num_rounds = federated_config.num_rounds
        
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        start_time = time.time()
        
        for round_num in range(num_rounds):
            round_start_time = time.time()
            
            logger.info(f"Round {round_num + 1}/{num_rounds}")
            
            # Get global parameters
            global_parameters = self.server.get_global_parameters()
            
            # Client training
            client_results = []
            for client in self.clients:
                try:
                    result = client.train_local(global_parameters)
                    client_results.append(result)
                    
                    logger.debug(f"Client {client.client_id}: "
                               f"Loss={result['avg_loss']:.4f}, "
                               f"Accuracy={result['avg_accuracy']:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error training client {client.client_id}: {e}")
                    continue
            
            if not client_results:
                logger.error("No clients completed training successfully")
                break
            
            # Aggregate updates
            self.server.aggregate_updates(client_results)
            
            # Evaluate global model
            eval_results = self.server.evaluate_global_model()
            
            # Update training history
            self.server.training_history['round'].append(round_num + 1)
            self.server.training_history['val_loss'].append(eval_results['val_loss'])
            self.server.training_history['val_accuracy'].append(eval_results['val_accuracy'])
            self.server.training_history['test_loss'].append(eval_results['test_loss'])
            self.server.training_history['test_accuracy'].append(eval_results['test_accuracy'])
            
            # Privacy metrics
            if self.enable_dp:
                total_privacy_spent = self._calculate_total_privacy_spent()
                self.server.training_history['privacy_budget'].append(total_privacy_spent)
            
            # Communication metrics
            round_time = time.time() - round_start_time
            self.server.training_history['communication_rounds'].append(round_time)
            
            # Log progress
            logger.info(f"Round {round_num + 1} completed: "
                       f"Val Acc={eval_results['val_accuracy']:.2f}%, "
                       f"Test Acc={eval_results['test_accuracy']:.2f}%, "
                       f"Time={round_time:.2f}s")
            
            # Save model periodically
            if (round_num + 1) % training_config.save_interval == 0:
                model_path = training_config.models_dir / f"federated_model_round_{round_num + 1}.pth"
                self.server.save_model(str(model_path))
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_eval = self.server.evaluate_global_model()
        
        # Save final model
        final_model_path = training_config.models_dir / "federated_model_final.pth"
        self.server.save_model(str(final_model_path))
        
        # Compile results
        results = {
            'final_metrics': final_eval,
            'training_history': self.server.training_history,
            'model_summary': self.server.model_summary,
            'total_training_time': total_time,
            'num_rounds': num_rounds,
            'num_clients': len(self.clients),
            'enable_dp': self.enable_dp,
            'aggregation_method': self.aggregation_method
        }
        
        if self.enable_dp:
            results['privacy_report'] = self._create_privacy_report()
        
        logger.info(f"Federated training completed in {total_time:.2f}s")
        logger.info(f"Final test accuracy: {final_eval['test_accuracy']:.2f}%")
        
        return results
    
    def _calculate_total_privacy_spent(self) -> Dict:
        """Calculate total privacy budget spent across all clients"""
        total_epsilon = 0.0
        total_delta = 0.0
        
        for client in self.clients:
            if client.privacy_manager:
                privacy_spent = client.privacy_manager.get_privacy_spent()
                total_epsilon = max(total_epsilon, privacy_spent['epsilon'])
                total_delta = max(total_delta, privacy_spent['delta'])
        
        return {
            'epsilon': total_epsilon,
            'delta': total_delta
        }
    
    def _create_privacy_report(self) -> Dict:
        """Create comprehensive privacy report"""
        if not self.enable_dp:
            return {}
        
        # Use first client's privacy manager for report
        privacy_manager = self.clients[0].privacy_manager
        if privacy_manager:
            return create_privacy_report(
                privacy_manager,
                len(self.server.training_history['round']),
                len(self.clients)
            )
        
        return {}
    
    def save_results(self, results: Dict, filepath: str):
        """Save training results to file"""
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def run_centralized_baseline() -> Dict:
    """
    Run centralized training as baseline
    
    Returns:
        Training results
    """
    logger.info("Running centralized training baseline")
    
    # Prepare data
    train_dataloader, val_dataloader, test_dataloader = get_centralized_dataloaders()
    
    # Create model
    model = create_model()
    optimizer = get_optimizer(model)
    criterion = get_criterion()
    device = torch.device(training_config.device)
    
    # Training history
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(federated_config.num_rounds * federated_config.local_epochs):
        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_dataloader, optimizer, criterion, device
        )
        
        # Evaluate
        val_loss, val_accuracy, _ = evaluate_model(
            model, val_dataloader, criterion, device
        )
        
        # Update history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}: Train Acc={train_accuracy:.2f}%, "
                       f"Val Acc={val_accuracy:.2f}%")
    
    total_time = time.time() - start_time
    
    # Final evaluation
    test_loss, test_accuracy, test_metrics = evaluate_model(
        model, test_dataloader, criterion, device
    )
    
    # Save model
    model_path = training_config.models_dir / "centralized_model.pth"
    save_model(model, str(model_path))
    
    results = {
        'final_metrics': {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_metrics': test_metrics
        },
        'training_history': training_history,
        'total_training_time': total_time,
        'num_epochs': len(training_history['epoch'])
    }
    
    logger.info(f"Centralized training completed in {total_time:.2f}s")
    logger.info(f"Final test accuracy: {test_accuracy:.2f}%")
    
    return results 