"""
Unit tests for time series SSL project.

This module contains comprehensive tests for data generation, models,
configuration management, and training utilities.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import (
    TimeSeriesGenerator, TimeSeriesConfig, 
    create_ssl_dataset, create_forecasting_dataset
)
from models import (
    TransformerSSLModel, CNNSSLModel, HybridSSLModel,
    ContrastiveSSLModel, SSLTrainer, create_model
)
from config import (
    ModelConfig, TrainingConfig, DataConfig, Config,
    ConfigManager, Logger, CheckpointManager, create_default_config
)


class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_time_series_config(self):
        """Test TimeSeriesConfig dataclass."""
        config = TimeSeriesConfig()
        assert config.length == 1000
        assert config.noise_level == 0.1
        assert config.frequency == 'D'
    
    def test_time_series_generator(self):
        """Test TimeSeriesGenerator."""
        config = TimeSeriesConfig(length=100)
        generator = TimeSeriesGenerator(config)
        
        # Test sine wave generation
        sine_wave = generator.generate_sine_wave(amplitude=1.0, period=10)
        assert len(sine_wave) == 100
        assert np.allclose(sine_wave[0], sine_wave[10], atol=1e-10)  # Periodicity
        
        # Test trend generation
        trend = generator.generate_trend(slope=0.01)
        assert len(trend) == 100
        assert trend[0] == 0.0
        assert trend[-1] == 0.99
        
        # Test noise generation
        noise = generator.generate_noise('gaussian')
        assert len(noise) == 100
        assert np.abs(noise.mean()) < 0.5  # Should be close to zero
        
        # Test complex series generation
        series = generator.generate_complex_series()
        assert isinstance(series, pd.Series)
        assert len(series) == 100
        assert isinstance(series.index, pd.DatetimeIndex)
    
    def test_ssl_dataset_creation(self):
        """Test SSL dataset creation."""
        # Create simple time series
        t = np.linspace(0, 10, 100)
        series = pd.Series(np.sin(t), index=pd.date_range('2020-01-01', periods=100))
        
        # Create SSL dataset
        X, Y, M = create_ssl_dataset(series, seq_len=20, mask_prob=0.2)
        
        assert X.shape[0] == Y.shape[0] == M.shape[0]  # Same number of samples
        assert X.shape[1] == Y.shape[1] == M.shape[1] == 20  # Same sequence length
        assert X.shape[2] == Y.shape[2] == 1  # Single feature
        assert M.shape[2] == 1  # Mask has single dimension
        
        # Check that masked values are zero
        mask_positions = M > 0
        assert np.all(X[mask_positions] == 0.0)
        
        # Check that mask probability is approximately correct
        actual_mask_prob = M.mean()
        assert 0.1 <= actual_mask_prob <= 0.3  # Allow some variance
    
    def test_forecasting_dataset_creation(self):
        """Test forecasting dataset creation."""
        # Create simple time series
        t = np.linspace(0, 10, 100)
        series = pd.Series(np.sin(t), index=pd.date_range('2020-01-01', periods=100))
        
        # Create forecasting dataset
        X, Y = create_forecasting_dataset(series, input_len=20, output_len=5)
        
        assert X.shape[0] == Y.shape[0]  # Same number of samples
        assert X.shape[1] == 20  # Input length
        assert Y.shape[1] == 5   # Output length
        assert X.shape[2] == 1   # Single feature
        assert Y.shape[2] == 1   # Single feature


class TestModels:
    """Test model architectures."""
    
    def test_transformer_model(self):
        """Test TransformerSSLModel."""
        model = TransformerSSLModel(input_dim=1, d_model=64, nhead=4, num_layers=2)
        
        # Test forward pass
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 1)
        output = model(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_cnn_model(self):
        """Test CNNSSLModel."""
        model = CNNSSLModel(input_dim=1, hidden_dims=[16, 32], kernel_sizes=[3, 5])
        
        # Test forward pass
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 1)
        output = model(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_hybrid_model(self):
        """Test HybridSSLModel."""
        model = HybridSSLModel(
            input_dim=1, 
            cnn_hidden_dims=[16, 32], 
            cnn_kernel_sizes=[3, 5],
            d_model=64, 
            nhead=4, 
            num_layers=2
        )
        
        # Test forward pass
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 1)
        output = model(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_contrastive_model(self):
        """Test ContrastiveSSLModel."""
        model = ContrastiveSSLModel(input_dim=1, hidden_dim=64, projection_dim=32)
        
        # Test forward pass
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 1)
        output = model(x)
        
        assert output.shape == (batch_size, 32)  # Projection dimension
        assert not torch.isnan(output).any()
        
        # Check that output is normalized
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_model_factory(self):
        """Test model creation factory function."""
        model_types = ['transformer', 'cnn', 'hybrid', 'contrastive']
        
        for model_type in model_types:
            model = create_model(model_type, input_dim=1)
            assert isinstance(model, torch.nn.Module)
            
            # Test forward pass
            x = torch.randn(2, 10, 1)
            if model_type == 'contrastive':
                output = model(x)
                assert output.shape[0] == 2
            else:
                output = model(x)
                assert output.shape == x.shape
    
    def test_ssl_trainer(self):
        """Test SSLTrainer."""
        model = TransformerSSLModel(input_dim=1, d_model=32, nhead=2, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = SSLTrainer(model, optimizer, device='cpu', mask_prob=0.2)
        
        # Test training step
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 1)
        
        metrics = trainer.train_step(x)
        assert 'loss' in metrics
        assert 'mask_ratio' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['mask_ratio'] <= 1
        
        # Test evaluation
        eval_metrics = trainer.evaluate(x)
        assert 'loss' in eval_metrics
        assert 'reconstruction_error' in eval_metrics
        assert 'mask_ratio' in eval_metrics


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_dataclasses(self):
        """Test configuration dataclasses."""
        model_config = ModelConfig()
        training_config = TrainingConfig()
        data_config = DataConfig()
        
        assert model_config.model_type == 'transformer'
        assert training_config.batch_size == 32
        assert data_config.length == 1000
    
    def test_config_class(self):
        """Test main Config class."""
        config = Config(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig()
        )
        
        assert config.experiment_name == 'ssl_experiment'
        assert config.log_level == 'INFO'
        
        # Test conversion to/from dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict
        assert 'data' in config_dict
        
        # Test creation from dict
        new_config = Config.from_dict(config_dict)
        assert new_config.experiment_name == config.experiment_name
    
    def test_config_manager(self):
        """Test ConfigManager."""
        config = create_default_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test YAML save/load
            yaml_path = Path(temp_dir) / 'config.yaml'
            ConfigManager.save_yaml(config, yaml_path)
            assert yaml_path.exists()
            
            loaded_config = ConfigManager.load_yaml(yaml_path)
            assert loaded_config.experiment_name == config.experiment_name
            
            # Test JSON save/load
            json_path = Path(temp_dir) / 'config.json'
            ConfigManager.save_json(config, json_path)
            assert json_path.exists()
            
            loaded_config = ConfigManager.load_json(json_path)
            assert loaded_config.experiment_name == config.experiment_name
    
    def test_logger(self):
        """Test Logger class."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger_manager = Logger(
                name='test_logger',
                log_level='INFO',
                log_dir=temp_dir
            )
            
            logger = logger_manager.get_logger()
            assert isinstance(logger, logging.Logger)
            assert logger.name == 'test_logger'
    
    def test_checkpoint_manager(self):
        """Test CheckpointManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create dummy model and optimizer
            model = torch.nn.Linear(10, 1)
            optimizer = torch.optim.Adam(model.parameters())
            config = create_default_config()
            
            # Test save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, epoch=0, loss=0.5, config=config
            )
            assert Path(checkpoint_path).exists()
            
            # Test load checkpoint
            new_model = torch.nn.Linear(10, 1)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            checkpoint_info = checkpoint_manager.load_checkpoint(
                checkpoint_path, new_model, new_optimizer
            )
            
            assert checkpoint_info['epoch'] == 0
            assert checkpoint_info['loss'] == 0.5
            
            # Test list checkpoints
            checkpoints = checkpoint_manager.list_checkpoints()
            assert len(checkpoints) == 1
            
            # Test get latest checkpoint
            latest = checkpoint_manager.get_latest_checkpoint()
            assert latest is not None


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create simple data
        config = TimeSeriesConfig(length=100)
        generator = TimeSeriesGenerator(config)
        series = generator.generate_complex_series()
        
        X, Y, M = create_ssl_dataset(series, seq_len=20, mask_prob=0.2)
        
        # Create model and trainer
        model = TransformerSSLModel(input_dim=1, d_model=32, nhead=2, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = SSLTrainer(model, optimizer, device='cpu', mask_prob=0.2)
        
        # Train for a few steps
        X_tensor = torch.FloatTensor(X[:10])  # Use small subset
        Y_tensor = torch.FloatTensor(Y[:10])
        
        initial_loss = trainer.evaluate(Y_tensor)['loss']
        
        for _ in range(5):
            trainer.train_step(Y_tensor)
        
        final_loss = trainer.evaluate(Y_tensor)['loss']
        
        # Loss should generally decrease (though not guaranteed for small number of steps)
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
