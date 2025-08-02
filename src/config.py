"""
Configuration file for Differentially Private Federated Learning Simulation
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class FederatedConfig:
    """Configuration for federated learning setup"""
    num_clients: int = 5
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Client data distribution
    data_distribution: str = "iid"  # "iid" or "non_iid"
    non_iid_alpha: float = 0.5  # Dirichlet distribution parameter
    
    # Communication settings
    min_fit_clients: int = 3
    min_evaluate_clients: int = 3
    min_available_clients: int = 3

@dataclass
class PrivacyConfig:
    """Configuration for differential privacy"""
    enable_dp: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    target_epsilon: float = 10.0
    
    # Privacy accountant
    accountant_type: str = "rdp"  # "rdp" or "gdp"
    alphas: List[float] = None
    
    def __post_init__(self):
        if self.alphas is None:
            self.alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

@dataclass
class ModelConfig:
    """Configuration for neural network model"""
    input_size: int = 784  # 28x28 MNIST images
    hidden_sizes: List[int] = None
    num_classes: int = 10
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 256, 128]

@dataclass
class DataConfig:
    """Configuration for data handling"""
    dataset_name: str = "mnist"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    normalize: bool = True
    augment: bool = False
    
    # Data paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    
    def __post_init__(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    device: str = "cpu"  # "cpu" or "cuda"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 20
    eval_interval: int = 5
    
    # Paths
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    viz_dir: Path = PROJECT_ROOT / "visualizations"
    
    def __post_init__(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Configuration for experiments and evaluation"""
    # Baseline comparison
    run_centralized: bool = True
    run_federated_no_dp: bool = True
    run_federated_with_dp: bool = True
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "loss", "privacy_budget", "communication_rounds"]

# Global configuration instances
federated_config = FederatedConfig()
privacy_config = PrivacyConfig()
model_config = ModelConfig()
data_config = DataConfig()
training_config = TrainingConfig()
experiment_config = ExperimentConfig()

# Environment variables
def get_env_config():
    """Get configuration from environment variables"""
    return {
        "device": os.getenv("DEVICE", "cpu"),
        "num_clients": int(os.getenv("NUM_CLIENTS", "5")),
        "num_rounds": int(os.getenv("NUM_ROUNDS", "100")),
        "enable_dp": os.getenv("ENABLE_DP", "true").lower() == "true",
        "noise_multiplier": float(os.getenv("NOISE_MULTIPLIER", "1.0")),
    } 