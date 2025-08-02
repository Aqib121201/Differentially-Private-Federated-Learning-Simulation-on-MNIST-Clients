#!/usr/bin/env python3
"""
Main orchestrator script for Differentially Private Federated Learning Simulation

This script runs the complete pipeline including:
1. Centralized training baseline
2. Federated learning without differential privacy
3. Federated learning with differential privacy
4. Comprehensive evaluation and visualization
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import time
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import (
    federated_config, privacy_config, training_config, 
    experiment_config, get_env_config
)
from src.federated_training import (
    FederatedLearningSimulator, run_centralized_baseline
)
from src.visualization import (
    create_comprehensive_report, plot_experiment_summary,
    plot_comparison_curves, plot_privacy_utility_tradeoff
)
from src.data_preprocessing import analyze_data_distribution, prepare_federated_data

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(training_config.logs_dir / "federated_learning.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_experiments(args: argparse.Namespace) -> Dict[str, Dict]:
    """
    Run all experiments based on configuration
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of experiment results
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    # Override config with command line arguments
    if args.num_rounds:
        federated_config.num_rounds = args.num_rounds
    if args.num_clients:
        federated_config.num_clients = args.num_clients
    if args.noise_multiplier:
        privacy_config.noise_multiplier = args.noise_multiplier
    if args.enable_dp is not None:
        privacy_config.enable_dp = args.enable_dp
    
    # Get environment config
    env_config = get_env_config()
    logger.info(f"Environment configuration: {env_config}")
    
    # 1. Centralized Training Baseline
    if experiment_config.run_centralized:
        logger.info("=" * 60)
        logger.info("RUNNING CENTRALIZED TRAINING BASELINE")
        logger.info("=" * 60)
        
        try:
            centralized_results = run_centralized_baseline()
            results['centralized'] = centralized_results
            
            logger.info(f"Centralized training completed: "
                       f"Test Accuracy = {centralized_results['final_metrics']['test_accuracy']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error in centralized training: {e}")
    
    # 2. Federated Learning without Differential Privacy
    if experiment_config.run_federated_no_dp:
        logger.info("=" * 60)
        logger.info("RUNNING FEDERATED LEARNING WITHOUT DIFFERENTIAL PRIVACY")
        logger.info("=" * 60)
        
        try:
            # Temporarily disable DP
            original_enable_dp = privacy_config.enable_dp
            privacy_config.enable_dp = False
            
            simulator_no_dp = FederatedLearningSimulator(
                enable_dp=False,
                aggregation_method=args.aggregation_method
            )
            
            federated_no_dp_results = simulator_no_dp.run_federated_training()
            results['federated_no_dp'] = federated_no_dp_results
            
            # Save results
            results_path = training_config.logs_dir / "federated_no_dp_results.json"
            simulator_no_dp.save_results(federated_no_dp_results, str(results_path))
            
            logger.info(f"Federated learning (no DP) completed: "
                       f"Test Accuracy = {federated_no_dp_results['final_metrics']['test_accuracy']:.2f}%")
            
            # Restore DP setting
            privacy_config.enable_dp = original_enable_dp
            
        except Exception as e:
            logger.error(f"Error in federated learning without DP: {e}")
    
    # 3. Federated Learning with Differential Privacy
    if experiment_config.run_federated_with_dp and privacy_config.enable_dp:
        logger.info("=" * 60)
        logger.info("RUNNING FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY")
        logger.info("=" * 60)
        
        try:
            simulator_with_dp = FederatedLearningSimulator(
                enable_dp=True,
                aggregation_method=args.aggregation_method
            )
            
            federated_with_dp_results = simulator_with_dp.run_federated_training()
            results['federated_with_dp'] = federated_with_dp_results
            
            # Save results
            results_path = training_config.logs_dir / "federated_with_dp_results.json"
            simulator_with_dp.save_results(federated_with_dp_results, str(results_path))
            
            logger.info(f"Federated learning (with DP) completed: "
                       f"Test Accuracy = {federated_with_dp_results['final_metrics']['test_accuracy']:.2f}%")
            
            if 'privacy_report' in federated_with_dp_results:
                privacy_report = federated_with_dp_results['privacy_report']
                logger.info(f"Privacy guarantee: ({privacy_report['total_epsilon']:.2f}, "
                           f"{privacy_report['total_delta']:.2e})-DP")
            
        except Exception as e:
            logger.error(f"Error in federated learning with DP: {e}")
    
    return results

def create_experiment_summary(results: Dict[str, Dict]) -> None:
    """
    Create comprehensive experiment summary
    
    Args:
        results: Dictionary of experiment results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'Test Accuracy':<15} {'Test Loss':<12} {'Training Time':<15} {'Privacy ε':<12}")
    print("=" * 80)
    
    for method, result in results.items():
        final_metrics = result['final_metrics']
        accuracy = final_metrics['test_accuracy']
        loss = final_metrics['test_loss']
        training_time = result['total_training_time']
        
        # Privacy epsilon
        epsilon = 0.0
        if 'privacy_report' in result:
            epsilon = result['privacy_report']['total_epsilon']
        
        print(f"{method:<25} {accuracy:<15.2f} {loss:<12.4f} {training_time:<15.1f} {epsilon:<12.2f}")
    
    print("=" * 80)
    
    # Create comparison plots
    if len(results) > 1:
        logger.info("Creating comparison visualizations...")
        
        # Experiment summary plot
        summary_path = training_config.viz_dir / "experiment_summary.png"
        plot_experiment_summary(results, str(summary_path))
        
        # Accuracy comparison
        accuracy_path = training_config.viz_dir / "accuracy_comparison.png"
        plot_comparison_curves(results, metric='accuracy', save_path=str(accuracy_path))
        
        # Loss comparison
        loss_path = training_config.viz_dir / "loss_comparison.png"
        plot_comparison_curves(results, metric='loss', save_path=str(loss_path))

def analyze_data_distribution_analysis() -> None:
    """Analyze and visualize data distribution across clients"""
    logger = logging.getLogger(__name__)
    
    logger.info("Analyzing data distribution...")
    
    # Prepare federated data to get client datasets
    client_dataloaders, _, _ = prepare_federated_data()
    
    # Extract client datasets
    client_datasets = [dataloader.dataset for dataloader in client_dataloaders]
    
    # Analyze distribution
    distribution_stats = analyze_data_distribution(client_datasets)
    
    # Save distribution analysis
    distribution_path = training_config.logs_dir / "data_distribution_analysis.json"
    with open(distribution_path, 'w') as f:
        json.dump(distribution_stats, f, indent=2)
    
    logger.info(f"Data distribution analysis saved to {distribution_path}")
    logger.info(f"Average client size: {distribution_stats['avg_client_size']:.1f} samples")
    logger.info(f"Client size std: {distribution_stats['std_client_size']:.1f} samples")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Differentially Private Federated Learning Simulation on MNIST"
    )
    
    parser.add_argument(
        "--num-rounds", type=int, default=None,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--num-clients", type=int, default=None,
        help="Number of federated learning clients"
    )
    parser.add_argument(
        "--noise-multiplier", type=float, default=None,
        help="Noise multiplier for differential privacy"
    )
    parser.add_argument(
        "--enable-dp", action="store_true", default=None,
        help="Enable differential privacy"
    )
    parser.add_argument(
        "--disable-dp", action="store_true", default=None,
        help="Disable differential privacy"
    )
    parser.add_argument(
        "--aggregation-method", type=str, default="fedavg",
        choices=["fedavg", "fedprox"],
        help="Aggregation method for federated learning"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--skip-centralized", action="store_true",
        help="Skip centralized training baseline"
    )
    parser.add_argument(
        "--skip-federated-no-dp", action="store_true",
        help="Skip federated learning without DP"
    )
    parser.add_argument(
        "--skip-federated-with-dp", action="store_true",
        help="Skip federated learning with DP"
    )
    parser.add_argument(
        "--data-analysis-only", action="store_true",
        help="Only run data distribution analysis"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Handle DP arguments
    if args.enable_dp is not None and args.disable_dp is not None:
        logger.error("Cannot specify both --enable-dp and --disable-dp")
        sys.exit(1)
    
    if args.enable_dp:
        privacy_config.enable_dp = True
    elif args.disable_dp:
        privacy_config.enable_dp = False
    
    # Handle experiment skipping
    if args.skip_centralized:
        experiment_config.run_centralized = False
    if args.skip_federated_no_dp:
        experiment_config.run_federated_no_dp = False
    if args.skip_federated_with_dp:
        experiment_config.run_federated_with_dp = False
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("DIFFERENTIALLY PRIVATE FEDERATED LEARNING SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Federated config: {federated_config.num_clients} clients, "
               f"{federated_config.num_rounds} rounds, {federated_config.local_epochs} local epochs")
    logger.info(f"Privacy config: DP={privacy_config.enable_dp}, "
               f"noise_multiplier={privacy_config.noise_multiplier}, "
               f"target_epsilon={privacy_config.target_epsilon}")
    logger.info(f"Model config: {model_config.hidden_sizes} hidden layers, "
               f"{model_config.num_classes} classes")
    
    # Data analysis only
    if args.data_analysis_only:
        analyze_data_distribution_analysis()
        return
    
    # Run experiments
    start_time = time.time()
    
    try:
        # Analyze data distribution
        analyze_data_distribution_analysis()
        
        # Run experiments
        results = run_experiments(args)
        
        # Create comprehensive visualizations for each experiment
        for method, result in results.items():
            logger.info(f"Creating visualizations for {method}...")
            create_comprehensive_report(result)
        
        # Create experiment summary
        if len(results) > 1:
            create_experiment_summary(results)
        
        total_time = time.time() - start_time
        logger.info(f"All experiments completed in {total_time:.2f} seconds")
        
        # Save overall results
        overall_results_path = training_config.logs_dir / "overall_results.json"
        with open(overall_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Overall results saved to {overall_results_path}")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main() 