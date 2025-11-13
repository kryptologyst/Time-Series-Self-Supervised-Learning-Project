#!/usr/bin/env python3
"""
Demo script showing the time series SSL project in action.

This script demonstrates the complete pipeline from data generation
to model training and evaluation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data_generation import TimeSeriesGenerator, TimeSeriesConfig, create_ssl_dataset
from models import create_model, SSLTrainer
from config import create_default_config

def main():
    """Run the complete SSL demo."""
    print("Time Series Self-Supervised Learning Demo")
    print("=" * 50)
    
    # 1. Generate synthetic time series data
    print("\n1. Generating synthetic time series data...")
    config = TimeSeriesConfig(
        length=500,
        noise_level=0.1,
        trend_strength=0.3,
        seasonality_strength=0.8
    )
    
    generator = TimeSeriesGenerator(config)
    series = generator.generate_complex_series()
    
    print(f"   Generated series with {len(series)} points")
    print(f"   Mean: {series.mean():.3f}, Std: {series.std():.3f}")
    
    # 2. Create SSL dataset
    print("\n2. Creating SSL dataset...")
    X, Y, M = create_ssl_dataset(series, seq_len=30, mask_prob=0.2)
    
    print(f"   Created {len(X)} sequences")
    print(f"   Sequence length: {X.shape[1]}")
    print(f"   Mask ratio: {M.mean():.3f}")
    
    # 3. Create and train model
    print("\n3. Training SSL model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = create_model('transformer', input_dim=1, d_model=64, nhead=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = SSLTrainer(model, optimizer, device, mask_prob=0.2)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X[:50]).unsqueeze(-1)  # Add feature dimension
    Y_tensor = torch.FloatTensor(Y[:50]).unsqueeze(-1)  # Add feature dimension
    
    # Training loop
    print("   Training progress:")
    for epoch in range(10):
        metrics = trainer.train_step(Y_tensor)
        if epoch % 2 == 0:
            print(f"     Epoch {epoch+1:2d}: Loss = {metrics['loss']:.4f}")
    
    # 4. Evaluate model
    print("\n4. Evaluating model...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Calculate metrics
    mse = np.mean((Y_tensor.numpy() - predictions) ** 2)
    mae = np.mean(np.abs(Y_tensor.numpy() - predictions))
    
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # 5. Visualize results
    print("\n5. Creating visualization...")
    
    # Select a sample for visualization
    sample_idx = 0
    original = Y[sample_idx].flatten()
    predicted = predictions[sample_idx].flatten()
    mask = M[sample_idx].flatten()
    
    plt.figure(figsize=(12, 6))
    
    # Plot original vs predicted
    plt.subplot(1, 2, 1)
    plt.plot(original, 'b-', label='Original', linewidth=2)
    plt.plot(predicted, 'r--', label='Predicted', linewidth=2)
    
    # Highlight masked positions
    masked_positions = np.where(mask > 0)[0]
    if len(masked_positions) > 0:
        plt.scatter(masked_positions, original[masked_positions], 
                   color='orange', s=50, label='Masked Values', zorder=5)
    
    plt.title('SSL Reconstruction Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot masking pattern
    plt.subplot(1, 2, 2)
    plt.plot(mask, 'g-', linewidth=2)
    plt.fill_between(range(len(mask)), mask, alpha=0.3, color='green')
    plt.title('Masking Pattern')
    plt.xlabel('Time Steps')
    plt.ylabel('Mask (0/1)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'ssl_demo_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to: {plot_path}")
    
    # 6. Summary
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"✓ Generated {len(series)} time series points")
    print(f"✓ Created {len(X)} SSL training sequences")
    print(f"✓ Trained {model.__class__.__name__} model")
    print(f"✓ Achieved MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"✓ Saved visualization to {plot_path}")
    print("\nNext steps:")
    print("- Run 'python cli.py ui' to launch the interactive web interface")
    print("- Run 'python train.py' for full training with more epochs")
    print("- Check 'notebooks/ssl_analysis.ipynb' for detailed analysis")


if __name__ == "__main__":
    main()
