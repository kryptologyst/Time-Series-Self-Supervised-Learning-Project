"""
Time Series Self-Supervised Learning Package.

This package provides comprehensive tools for self-supervised learning
on time series data, including data generation, model architectures,
training utilities, and visualization interfaces.
"""

__version__ = "1.0.0"
__author__ = "Time Series SSL Team"
__email__ = "contact@timeseriesssl.com"

from .data_generation import (
    TimeSeriesGenerator,
    TimeSeriesConfig,
    create_ssl_dataset,
    create_forecasting_dataset
)

from .models import (
    TransformerSSLModel,
    CNNSSLModel,
    HybridSSLModel,
    ContrastiveSSLModel,
    SSLTrainer,
    create_model
)

from .config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    Config,
    ConfigManager,
    Logger,
    CheckpointManager,
    create_default_config,
    setup_experiment
)

__all__ = [
    # Data generation
    "TimeSeriesGenerator",
    "TimeSeriesConfig",
    "create_ssl_dataset",
    "create_forecasting_dataset",
    
    # Models
    "TransformerSSLModel",
    "CNNSSLModel",
    "HybridSSLModel",
    "ContrastiveSSLModel",
    "SSLTrainer",
    "create_model",
    
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "Config",
    "ConfigManager",
    "Logger",
    "CheckpointManager",
    "create_default_config",
    "setup_experiment",
]
