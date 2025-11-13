"""
Enhanced self-supervised learning models for time series.

This module implements various SSL architectures including CNN-based models,
Transformer-based models, and hybrid approaches for time series representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerSSLModel(nn.Module):
    """Transformer-based SSL model for time series."""
    
    def __init__(self, 
                 input_dim: int = 1,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer expects (seq_len, batch_size, d_model) for positional encoding
        # but we use batch_first=True, so we need to transpose
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.output_projection(x)
        return x


class CNNSSLModel(nn.Module):
    """Enhanced CNN-based SSL model for time series."""
    
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dims: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                decoder_layers.extend([
                    nn.Conv1d(hidden_dims[i], input_dim, kernel_sizes[i], 
                             padding=kernel_sizes[i]//2),
                    nn.Tanh()
                ])
            else:
                decoder_layers.extend([
                    nn.Conv1d(hidden_dims[i], hidden_dims[i-1], kernel_sizes[i], 
                             padding=kernel_sizes[i]//2),
                    nn.BatchNorm1d(hidden_dims[i-1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded.transpose(1, 2)  # (batch_size, seq_len, input_dim)


class HybridSSLModel(nn.Module):
    """Hybrid CNN-Transformer SSL model."""
    
    def __init__(self, 
                 input_dim: int = 1,
                 cnn_hidden_dims: List[int] = [32, 64],
                 cnn_kernel_sizes: List[int] = [3, 5],
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # CNN feature extractor
        cnn_layers = []
        prev_dim = input_dim
        
        for hidden_dim, kernel_size in zip(cnn_hidden_dims, cnn_kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.cnn_encoder = nn.Sequential(*cnn_layers)
        
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(cnn_hidden_dims[-1], d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # CNN feature extraction
        cnn_features = self.cnn_encoder(x)  # (batch_size, cnn_hidden_dim, seq_len)
        cnn_features = cnn_features.transpose(1, 2)  # (batch_size, seq_len, cnn_hidden_dim)
        
        # Project to transformer dimension
        transformer_input = self.feature_projection(cnn_features)
        transformer_input = self.pos_encoding(transformer_input)
        
        # Transformer processing
        transformer_output = self.transformer(transformer_input)
        
        # Output projection
        output = self.output_projection(transformer_output)
        
        return output


class ContrastiveSSLModel(nn.Module):
    """Contrastive learning SSL model for time series."""
    
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 projection_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, 3, padding=1),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        features = self.encoder(x).squeeze(-1)  # (batch_size, hidden_dim)
        projections = self.projection_head(features)  # (batch_size, projection_dim)
        
        return F.normalize(projections, dim=1)


class SSLTrainer:
    """Trainer for self-supervised learning models."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu',
                 mask_prob: float = 0.2):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.mask_prob = mask_prob
        
    def mask_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked version of input sequence."""
        batch_size, seq_len, input_dim = x.shape
        mask = torch.rand(batch_size, seq_len, device=self.device) < self.mask_prob
        
        masked_x = x.clone()
        masked_x[mask] = 0.0
        
        return masked_x, mask.float()
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create masked input
        masked_x, mask = self.mask_sequence(x)
        
        # Forward pass
        predictions = self.model(masked_x)
        
        # Compute masked loss
        mse_loss = F.mse_loss(predictions, x, reduction='none')
        masked_loss = (mse_loss * mask.unsqueeze(-1)).sum() / mask.sum()
        
        # Backward pass
        masked_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': masked_loss.item(),
            'mask_ratio': mask.mean().item()
        }
    
    def evaluate(self, x: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on masked sequences."""
        self.model.eval()
        
        with torch.no_grad():
            masked_x, mask = self.mask_sequence(x)
            predictions = self.model(masked_x)
            
            # Compute metrics
            mse_loss = F.mse_loss(predictions, x, reduction='none')
            masked_loss = (mse_loss * mask.unsqueeze(-1)).sum() / mask.sum()
            
            # Reconstruction accuracy on masked positions
            masked_predictions = predictions[mask.bool()]
            masked_targets = x[mask.bool()]
            reconstruction_error = F.mse_loss(masked_predictions, masked_targets)
            
        return {
            'loss': masked_loss.item(),
            'reconstruction_error': reconstruction_error.item(),
            'mask_ratio': mask.mean().item()
        }


def create_model(model_type: str = 'transformer', **kwargs) -> nn.Module:
    """Factory function to create different types of SSL models."""
    
    if model_type == 'transformer':
        return TransformerSSLModel(**kwargs)
    elif model_type == 'cnn':
        return CNNSSLModel(**kwargs)
    elif model_type == 'hybrid':
        return HybridSSLModel(**kwargs)
    elif model_type == 'contrastive':
        return ContrastiveSSLModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model('transformer', input_dim=1, d_model=128, nhead=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = SSLTrainer(model, optimizer, device)
    
    # Create dummy data
    batch_size, seq_len, input_dim = 32, 50, 1
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Training step
    metrics = trainer.train_step(x)
    print(f"Training metrics: {metrics}")
    
    # Evaluation
    eval_metrics = trainer.evaluate(x)
    print(f"Evaluation metrics: {eval_metrics}")
