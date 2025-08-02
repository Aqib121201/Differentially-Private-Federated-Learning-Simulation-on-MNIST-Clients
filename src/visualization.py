"""
Visualization module for federated learning results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from .config import training_config

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_curves(training_history: Dict, save_path: str = None, 
                        title: str = "Training Curves") -> None:
    """
    Plot training curves (loss and accuracy)
    
    Args:
        training_history: Training history dictionary
        save_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    rounds = training_history['round']
    
    # Loss curves
    if 'val_loss' in training_history:
        ax1.plot(rounds, training_history['val_loss'], 'b-', label='Validation Loss', linewidth=2)
    if 'test_loss' in training_history:
        ax1.plot(rounds, training_history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Communication Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'val_accuracy' in training_history:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(rounds, training_history['val_accuracy'], 'g--', 
                     label='Validation Accuracy', linewidth=2)
        ax1_twin.set_ylabel('Accuracy (%)', color='g')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        ax1_twin.legend(loc='upper right')
    
    # Accuracy plot
    if 'val_accuracy' in training_history:
        ax2.plot(rounds, training_history['val_accuracy'], 'g-', 
                label='Validation Accuracy', linewidth=2)
    if 'test_accuracy' in training_history:
        ax2.plot(rounds, training_history['test_accuracy'], 'm-', 
                label='Test Accuracy', linewidth=2)
    
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Communication Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_privacy_budget(privacy_budget: List[Dict], save_path: str = None) -> None:
    """
    Plot privacy budget consumption over rounds
    
    Args:
        privacy_budget: List of privacy budget dictionaries
        save_path: Path to save the plot
    """
    if not privacy_budget:
        logger.warning("No privacy budget data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    rounds = list(range(1, len(privacy_budget) + 1))
    epsilons = [budget['epsilon'] for budget in privacy_budget]
    deltas = [budget['delta'] for budget in privacy_budget]
    
    # Epsilon plot
    ax1.plot(rounds, epsilons, 'r-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Epsilon (ε)')
    ax1.set_title('Privacy Budget (ε) vs Communication Rounds')
    ax1.grid(True, alpha=0.3)
    
    # Delta plot
    ax2.semilogy(rounds, deltas, 'b-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Delta (δ)')
    ax2.set_title('Privacy Budget (δ) vs Communication Rounds')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Differential Privacy Budget Consumption', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Privacy budget plot saved to {save_path}")
    
    plt.show()

def plot_communication_metrics(communication_rounds: List[float], save_path: str = None) -> None:
    """
    Plot communication round timing metrics
    
    Args:
        communication_rounds: List of round completion times
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    rounds = list(range(1, len(communication_rounds) + 1))
    
    # Round timing
    ax1.plot(rounds, communication_rounds, 'purple', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Round Completion Time')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative time
    cumulative_time = np.cumsum(communication_rounds)
    ax2.plot(rounds, cumulative_time, 'orange', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Cumulative Time (seconds)')
    ax2.set_title('Cumulative Training Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Communication Metrics', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Communication metrics plot saved to {save_path}")
    
    plt.show()

def plot_comparison_curves(results_dict: Dict[str, Dict], metric: str = 'accuracy', 
                          save_path: str = None) -> None:
    """
    Plot comparison curves for different methods
    
    Args:
        results_dict: Dictionary of results for different methods
        metric: Metric to compare ('accuracy' or 'loss')
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (method_name, results) in enumerate(results_dict.items()):
        training_history = results['training_history']
        
        if metric == 'accuracy':
            if 'val_accuracy' in training_history:
                rounds = training_history['round']
                accuracies = training_history['val_accuracy']
                plt.plot(rounds, accuracies, color=colors[i % len(colors)], 
                        linewidth=2, marker='o', markersize=4, label=f'{method_name} (Val)')
            
            if 'test_accuracy' in training_history:
                rounds = training_history['round']
                accuracies = training_history['test_accuracy']
                plt.plot(rounds, accuracies, color=colors[i % len(colors)], 
                        linewidth=2, linestyle='--', marker='s', markersize=4, 
                        label=f'{method_name} (Test)')
        
        elif metric == 'loss':
            if 'val_loss' in training_history:
                rounds = training_history['round']
                losses = training_history['val_loss']
                plt.plot(rounds, losses, color=colors[i % len(colors)], 
                        linewidth=2, marker='o', markersize=4, label=f'{method_name} (Val)')
            
            if 'test_loss' in training_history:
                rounds = training_history['round']
                losses = training_history['test_loss']
                plt.plot(rounds, losses, color=colors[i % len(colors)], 
                        linewidth=2, linestyle='--', marker='s', markersize=4, 
                        label=f'{method_name} (Test)')
    
    plt.xlabel('Communication Rounds')
    plt.ylabel(f'{metric.capitalize()}')
    plt.title(f'{metric.capitalize()} Comparison Across Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.show()

def plot_privacy_utility_tradeoff(noise_multipliers: List[float], 
                                accuracies: List[float],
                                epsilons: List[float], save_path: str = None) -> None:
    """
    Plot privacy-utility tradeoff
    
    Args:
        noise_multipliers: List of noise multipliers
        accuracies: List of corresponding accuracies
        epsilons: List of corresponding epsilon values
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Noise multiplier vs Accuracy
    ax1.plot(noise_multipliers, accuracies, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Multiplier')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Noise Multiplier')
    ax1.grid(True, alpha=0.3)
    
    # Epsilon vs Accuracy
    ax2.plot(epsilons, accuracies, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Epsilon (ε)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Privacy Budget (ε)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Privacy-Utility Tradeoff Analysis', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Privacy-utility tradeoff plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(confusion_matrix: List[List[int]], class_names: List[str] = None,
                         save_path: str = None, title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix as 2D list
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_data_distribution(client_sizes: List[int], class_distributions: List[Dict],
                          save_path: str = None) -> None:
    """
    Plot data distribution across clients
    
    Args:
        client_sizes: List of client dataset sizes
        class_distributions: List of class distributions for each client
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Client sizes
    client_ids = list(range(len(client_sizes)))
    ax1.bar(client_ids, client_sizes, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Data Distribution Across Clients')
    ax1.grid(True, alpha=0.3)
    
    # Class distribution heatmap
    if class_distributions:
        # Create matrix for heatmap
        all_classes = set()
        for dist in class_distributions:
            all_classes.update(dist.keys())
        
        all_classes = sorted(list(all_classes))
        distribution_matrix = []
        
        for dist in class_distributions:
            row = [dist.get(cls, 0) for cls in all_classes]
            distribution_matrix.append(row)
        
        sns.heatmap(distribution_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=all_classes, yticklabels=client_ids, ax=ax2)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Client ID')
        ax2.set_title('Class Distribution Heatmap')
    
    plt.tight_layout()
    plt.suptitle('Data Distribution Analysis', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Data distribution plot saved to {save_path}")
    
    plt.show()

def create_comprehensive_report(results: Dict, save_dir: str = None) -> None:
    """
    Create comprehensive visualization report
    
    Args:
        results: Results dictionary from federated training
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = training_config.viz_dir
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    training_history = results['training_history']
    
    # Training curves
    plot_training_curves(
        training_history,
        save_path=save_dir / "training_curves.png",
        title="Federated Learning Training Curves"
    )
    
    # Privacy budget
    if 'privacy_budget' in training_history and training_history['privacy_budget']:
        plot_privacy_budget(
            training_history['privacy_budget'],
            save_path=save_dir / "privacy_budget.png"
        )
    
    # Communication metrics
    if 'communication_rounds' in training_history:
        plot_communication_metrics(
            training_history['communication_rounds'],
            save_path=save_dir / "communication_metrics.png"
        )
    
    # Confusion matrix for final test results
    if 'final_metrics' in results and 'test_metrics' in results['final_metrics']:
        test_metrics = results['final_metrics']['test_metrics']
        if 'confusion_matrix' in test_metrics:
            plot_confusion_matrix(
                test_metrics['confusion_matrix'],
                class_names=[f'Digit {i}' for i in range(10)],
                save_path=save_dir / "confusion_matrix.png",
                title="Final Test Confusion Matrix"
            )
    
    logger.info(f"Comprehensive report saved to {save_dir}")

def plot_experiment_summary(experiment_results: Dict[str, Dict], save_path: str = None) -> None:
    """
    Create summary plot comparing different experiments
    
    Args:
        experiment_results: Dictionary of experiment results
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = list(experiment_results.keys())
    final_accuracies = []
    final_losses = []
    training_times = []
    privacy_epsilons = []
    
    for method, results in experiment_results.items():
        final_metrics = results['final_metrics']
        final_accuracies.append(final_metrics['test_accuracy'])
        final_losses.append(final_metrics['test_loss'])
        training_times.append(results['total_training_time'])
        
        # Privacy epsilon
        if 'privacy_report' in results:
            privacy_epsilons.append(results['privacy_report']['total_epsilon'])
        else:
            privacy_epsilons.append(0.0)
    
    # Final accuracy comparison
    bars1 = ax1.bar(methods, final_accuracies, color='lightblue', alpha=0.7)
    ax1.set_ylabel('Final Test Accuracy (%)')
    ax1.set_title('Final Test Accuracy Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, final_accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Final loss comparison
    bars2 = ax2.bar(methods, final_losses, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Final Test Loss')
    ax2.set_title('Final Test Loss Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, loss in zip(bars2, final_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    bars3 = ax3.bar(methods, training_times, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars3, training_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{time_val:.0f}s', ha='center', va='bottom')
    
    # Privacy budget comparison
    bars4 = ax4.bar(methods, privacy_epsilons, color='lightyellow', alpha=0.7)
    ax4.set_ylabel('Privacy Budget (ε)')
    ax4.set_title('Privacy Budget Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, epsilon in zip(bars4, privacy_epsilons):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{epsilon:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.suptitle('Experiment Summary Comparison', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Experiment summary saved to {save_path}")
    
    plt.show() 