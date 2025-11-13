"""
Configuration management and logging utilities.

This module provides configuration loading, logging setup, and checkpoint management
for the time series SSL project.
"""

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import torch
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for SSL models."""
    model_type: str = 'transformer'
    input_dim: int = 1
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 1000


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    mask_prob: float = 0.2
    seq_len: int = 50
    device: str = 'auto'
    save_every: int = 10
    eval_every: int = 5


@dataclass
class DataConfig:
    """Configuration for data generation."""
    length: int = 1000
    frequency: str = 'D'
    start_date: str = '2020-01-01'
    noise_level: float = 0.1
    trend_strength: float = 0.5
    seasonality_strength: float = 1.0
    anomaly_probability: float = 0.05


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = 'ssl_experiment'
    output_dir: str = './outputs'
    log_level: str = 'INFO'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment_name=config_dict.get('experiment_name', 'ssl_experiment'),
            output_dir=config_dict.get('output_dir', './outputs'),
            log_level=config_dict.get('log_level', 'INFO')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'log_level': self.log_level
        }


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Config:
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config.from_dict(config_dict)
    
    @staticmethod
    def save_yaml(config: Config, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Config:
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return Config.from_dict(config_dict)
    
    @staticmethod
    def save_json(config: Config, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)


class Logger:
    """Logger setup and management."""
    
    def __init__(self, 
                 name: str = 'ssl_project',
                 log_level: str = 'INFO',
                 log_dir: str = './logs'):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with file and console handlers."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{self.name}_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       config: Config,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: Union[str, Path],
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def list_checkpoints(self) -> list:
        """List available checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoint_files.sort()
        return checkpoint_files
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None


def create_default_config() -> Config:
    """Create default configuration."""
    return Config(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig()
    )


def setup_experiment(config: Config) -> tuple[Logger, CheckpointManager]:
    """Setup experiment with logging and checkpoint management."""
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / 'logs'
    logger_manager = Logger(
        name=config.experiment_name,
        log_level=config.log_level,
        log_dir=str(log_dir)
    )
    
    # Setup checkpoint management
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    # Save configuration
    config_path = output_dir / 'config.yaml'
    ConfigManager.save_yaml(config, config_path)
    
    logger = logger_manager.get_logger()
    logger.info(f"Experiment setup complete. Output directory: {output_dir}")
    logger.info(f"Configuration saved to: {config_path}")
    
    return logger_manager, checkpoint_manager


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    config.experiment_name = 'test_experiment'
    config.training.num_epochs = 50
    
    # Setup experiment
    logger_manager, checkpoint_manager = setup_experiment(config)
    logger = logger_manager.get_logger()
    
    logger.info("Configuration and logging setup complete")
    
    # Save and load config
    ConfigManager.save_yaml(config, 'test_config.yaml')
    loaded_config = ConfigManager.load_yaml('test_config.yaml')
    
    logger.info(f"Loaded config experiment name: {loaded_config.experiment_name}")
