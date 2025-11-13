"""
Streamlit interface for time series SSL visualization and analysis.

This module provides an interactive web interface for exploring SSL models,
visualizing results, and comparing different approaches.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import TimeSeriesGenerator, TimeSeriesConfig, create_ssl_dataset
from models import create_model, SSLTrainer
from config import create_default_config, ConfigManager


def setup_page():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Time Series SSL Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Time Series Self-Supervised Learning Analysis")
    st.markdown("Explore and analyze self-supervised learning models for time series data")


def create_sidebar():
    """Create sidebar controls."""
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    data_length = st.sidebar.slider("Series Length", 100, 2000, 1000)
    noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 2.0, 0.5)
    seasonality_strength = st.sidebar.slider("Seasonality Strength", 0.0, 2.0, 1.0)
    anomaly_prob = st.sidebar.slider("Anomaly Probability", 0.0, 0.2, 0.05)
    
    # Model parameters
    st.sidebar.subheader("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["transformer", "cnn", "hybrid", "contrastive"]
    )
    seq_len = st.sidebar.slider("Sequence Length", 20, 100, 50)
    mask_prob = st.sidebar.slider("Mask Probability", 0.1, 0.5, 0.2)
    
    # Training parameters
    st.sidebar.subheader("Training")
    num_epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, 32)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
    
    return {
        'data_length': data_length,
        'noise_level': noise_level,
        'trend_strength': trend_strength,
        'seasonality_strength': seasonality_strength,
        'anomaly_prob': anomaly_prob,
        'model_type': model_type,
        'seq_len': seq_len,
        'mask_prob': mask_prob,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }


def generate_data(config: Dict) -> Tuple[pd.Series, Dict[str, np.ndarray]]:
    """Generate synthetic time series data."""
    data_config = TimeSeriesConfig(
        length=config['data_length'],
        noise_level=config['noise_level'],
        trend_strength=config['trend_strength'],
        seasonality_strength=config['seasonality_strength'],
        anomaly_probability=config['anomaly_prob']
    )
    
    generator = TimeSeriesGenerator(data_config)
    series = generator.generate_complex_series()
    
    # Create SSL dataset
    X, Y, M = create_ssl_dataset(
        series, 
        seq_len=config['seq_len'], 
        mask_prob=config['mask_prob']
    )
    
    return series, {'X': X, 'Y': Y, 'M': M}


def plot_time_series(series: pd.Series, title: str = "Time Series"):
    """Plot time series data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name='Time Series',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_ssl_reconstruction(X: np.ndarray, Y: np.ndarray, M: np.ndarray, 
                          predictions: np.ndarray, sample_idx: int = 0):
    """Plot SSL reconstruction results."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Original vs Predicted", "Masking Pattern"),
        vertical_spacing=0.1
    )
    
    # Get sample data
    original = Y[sample_idx].flatten()
    masked = X[sample_idx].flatten()
    predicted = predictions[sample_idx].flatten()
    mask = M[sample_idx].flatten()
    
    # Plot original vs predicted
    fig.add_trace(
        go.Scatter(
            y=original,
            mode='lines',
            name='Original',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # Highlight masked positions
    masked_positions = np.where(mask > 0)[0]
    if len(masked_positions) > 0:
        fig.add_trace(
            go.Scatter(
                x=masked_positions,
                y=original[masked_positions],
                mode='markers',
                name='Masked Values',
                marker=dict(color='orange', size=8, symbol='circle')
            ),
            row=1, col=1
        )
    
    # Plot masking pattern
    fig.add_trace(
        go.Scatter(
            y=mask,
            mode='lines',
            name='Mask Pattern',
            line=dict(color='green'),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"SSL Reconstruction Results (Sample {sample_idx})",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time Steps", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Mask (0/1)", row=2, col=1)
    
    return fig


def train_model(config: Dict, X: np.ndarray, Y: np.ndarray) -> Tuple[torch.nn.Module, List[float]]:
    """Train SSL model and return trained model with loss history."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_dim=1,
        d_model=128,
        nhead=8
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = SSLTrainer(model, optimizer, device, config['mask_prob'])
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    
    # Training loop
    loss_history = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        
        # Simple batch training (in practice, you'd use DataLoader)
        batch_size = config['batch_size']
        n_batches = len(X_tensor) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_tensor[start_idx:end_idx]
            batch_Y = Y_tensor[start_idx:end_idx]
            
            metrics = trainer.train_step(batch_Y)
            epoch_losses.append(metrics['loss'])
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        # Update progress
        progress = (epoch + 1) / config['num_epochs']
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return model, loss_history


def plot_training_loss(loss_history: List[float]):
    """Plot training loss curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_history,
        mode='lines',
        name='Training Loss',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit application."""
    setup_page()
    
    # Create sidebar controls
    config = create_sidebar()
    
    # Generate data
    st.header("📊 Data Generation")
    if st.button("Generate New Data"):
        with st.spinner("Generating time series data..."):
            series, ssl_data = generate_data(config)
            st.session_state['series'] = series
            st.session_state['ssl_data'] = ssl_data
    
    if 'series' in st.session_state:
        series = st.session_state['series']
        ssl_data = st.session_state['ssl_data']
        
        # Display data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Length", len(series))
        with col2:
            st.metric("Mean", f"{series.mean():.3f}")
        with col3:
            st.metric("Std", f"{series.std():.3f}")
        with col4:
            st.metric("SSL Samples", len(ssl_data['X']))
        
        # Plot time series
        fig = plot_time_series(series, "Generated Time Series")
        st.plotly_chart(fig, use_container_width=True)
        
        # Model training section
        st.header("🤖 Model Training")
        if st.button("Train Model"):
            with st.spinner("Training SSL model..."):
                model, loss_history = train_model(config, ssl_data['X'], ssl_data['Y'])
                st.session_state['model'] = model
                st.session_state['loss_history'] = loss_history
        
        if 'model' in st.session_state and 'loss_history' in st.session_state:
            model = st.session_state['model']
            loss_history = st.session_state['loss_history']
            
            # Plot training loss
            fig_loss = plot_training_loss(loss_history)
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Model evaluation section
            st.header("📈 Model Evaluation")
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(ssl_data['X'])
                predictions = model(X_tensor).numpy()
            
            # Sample selection
            sample_idx = st.selectbox(
                "Select Sample for Visualization",
                range(min(10, len(ssl_data['X'])))
            )
            
            # Plot reconstruction
            fig_recon = plot_ssl_reconstruction(
                ssl_data['X'], ssl_data['Y'], ssl_data['M'], 
                predictions, sample_idx
            )
            st.plotly_chart(fig_recon, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                mse = np.mean((ssl_data['Y'] - predictions) ** 2)
                st.metric("MSE", f"{mse:.4f}")
            with col2:
                mae = np.mean(np.abs(ssl_data['Y'] - predictions))
                st.metric("MAE", f"{mae:.4f}")
            with col3:
                # Masked reconstruction error
                mask = ssl_data['M'] > 0
                if mask.sum() > 0:
                    masked_mse = np.mean((ssl_data['Y'][mask] - predictions[mask]) ** 2)
                    st.metric("Masked MSE", f"{masked_mse:.4f}")
    
    else:
        st.info("Click 'Generate New Data' to start the analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit for Time Series Self-Supervised Learning Analysis")


if __name__ == "__main__":
    main()
