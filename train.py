"""
Main training script for time series SSL models.

This script provides a complete training pipeline with configuration management,
logging, checkpointing, and evaluation.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import TimeSeriesGenerator, TimeSeriesConfig, create_ssl_dataset
from models import create_model, SSLTrainer
from config import create_default_config, ConfigManager, setup_experiment


def create_data_loaders(X: np.ndarray, Y: np.ndarray, M: np.ndarray, 
                       batch_size: int = 32, train_split: float = 0.8) -> tuple:
    """Create train and validation data loaders."""
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    
    # Split data
    train_X, val_X = X[:n_train], X[n_train:]
    train_Y, val_Y = Y[:n_train], Y[n_train:]
    train_M, val_M = M[:n_train], M[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_X),
        torch.FloatTensor(train_Y),
        torch.FloatTensor(train_M)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_X),
        torch.FloatTensor(val_Y),
        torch.FloatTensor(val_M)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model: nn.Module, trainer: SSLTrainer, 
               train_loader: DataLoader) -> dict:
    """Train model for one epoch."""
    epoch_metrics = {'loss': [], 'mask_ratio': []}
    
    for batch_X, batch_Y, batch_M in train_loader:
        metrics = trainer.train_step(batch_Y)
        epoch_metrics['loss'].append(metrics['loss'])
        epoch_metrics['mask_ratio'].append(metrics['mask_ratio'])
    
    return {
        'loss': np.mean(epoch_metrics['loss']),
        'mask_ratio': np.mean(epoch_metrics['mask_ratio'])
    }


def evaluate_model(model: nn.Module, trainer: SSLTrainer, 
                  val_loader: DataLoader) -> dict:
    """Evaluate model on validation set."""
    eval_metrics = {'loss': [], 'reconstruction_error': [], 'mask_ratio': []}
    
    for batch_X, batch_Y, batch_M in val_loader:
        metrics = trainer.evaluate(batch_Y)
        eval_metrics['loss'].append(metrics['loss'])
        eval_metrics['reconstruction_error'].append(metrics['reconstruction_error'])
        eval_metrics['mask_ratio'].append(metrics['mask_ratio'])
    
    return {
        'loss': np.mean(eval_metrics['loss']),
        'reconstruction_error': np.mean(eval_metrics['reconstruction_error']),
        'mask_ratio': np.mean(eval_metrics['mask_ratio'])
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SSL model for time series')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--experiment_name', type=str, default='ssl_experiment',
                       help='Name of the experiment')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = ConfigManager.load_yaml(args.config)
    else:
        config = create_default_config()
        config.experiment_name = args.experiment_name
        config.output_dir = args.output_dir
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    config.training.device = str(device)
    
    # Setup experiment
    logger_manager, checkpoint_manager = setup_experiment(config)
    logger = logger_manager.get_logger()
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Generate data
    logger.info("Generating synthetic time series data...")
    data_config = TimeSeriesConfig(
        length=config.data.length,
        noise_level=config.data.noise_level,
        trend_strength=config.data.trend_strength,
        seasonality_strength=config.data.seasonality_strength,
        anomaly_probability=config.data.anomaly_probability
    )
    
    generator = TimeSeriesGenerator(data_config)
    series = generator.generate_complex_series()
    
    # Create SSL dataset
    X, Y, M = create_ssl_dataset(
        series,
        seq_len=config.training.seq_len,
        mask_prob=config.training.mask_prob
    )
    
    logger.info(f"Generated dataset with {len(X)} samples")
    logger.info(f"Sequence length: {config.training.seq_len}")
    logger.info(f"Mask probability: {config.training.mask_prob}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X, Y, M, 
        batch_size=config.training.batch_size
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info(f"Creating {config.model.model_type} model...")
    model = create_model(
        model_type=config.model.model_type,
        input_dim=config.model.input_dim,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_len=config.model.max_len
    )
    
    # Create optimizer and trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    trainer = SSLTrainer(
        model, optimizer, device, config.training.mask_prob
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(config.training.num_epochs):
        # Training
        train_metrics = train_epoch(model, trainer, train_loader)
        train_losses.append(train_metrics['loss'])
        
        # Validation
        if epoch % config.training.eval_every == 0:
            val_metrics = evaluate_model(model, trainer, val_loader)
            val_losses.append(val_metrics['loss'])
            
            logger.info(
                f"Epoch {epoch+1}/{config.training.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Mask Ratio: {train_metrics['mask_ratio']:.3f}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{config.training.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Mask Ratio: {train_metrics['mask_ratio']:.3f}"
            )
        
        # Save checkpoint
        if epoch % config.training.save_every == 0:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'], config
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_val_metrics = evaluate_model(model, trainer, val_loader)
    
    logger.info(f"Final validation loss: {final_val_metrics['loss']:.4f}")
    logger.info(f"Final reconstruction error: {final_val_metrics['reconstruction_error']:.4f}")
    
    # Save final model
    final_checkpoint_path = checkpoint_manager.save_checkpoint(
        model, optimizer, config.training.num_epochs - 1, 
        final_val_metrics['loss'], config,
        additional_info={'final_metrics': final_val_metrics}
    )
    
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    logger.info("Training completed successfully!")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses + [None] * (len(train_losses) - len(val_losses))
    })
    
    history_path = Path(config.output_dir) / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved: {history_path}")


if __name__ == "__main__":
    main()
