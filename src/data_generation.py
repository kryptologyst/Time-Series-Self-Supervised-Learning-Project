"""
Data generation module for creating realistic synthetic time series.

This module provides functions to generate various types of time series data
including trends, seasonality, noise, and anomalies for self-supervised learning.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series generation."""
    length: int = 1000
    frequency: str = 'D'  # Daily frequency
    start_date: str = '2020-01-01'
    noise_level: float = 0.1
    trend_strength: float = 0.5
    seasonality_strength: float = 1.0
    anomaly_probability: float = 0.05


class TimeSeriesGenerator:
    """Generator for synthetic time series data."""
    
    def __init__(self, config: TimeSeriesConfig):
        """Initialize the generator with configuration."""
        self.config = config
        self.rng = np.random.RandomState(42)
        
    def generate_sine_wave(self, amplitude: float = 1.0, period: float = 365) -> np.ndarray:
        """Generate a sine wave component."""
        t = np.arange(self.config.length)
        return amplitude * np.sin(2 * np.pi * t / period)
    
    def generate_trend(self, slope: float = 0.001) -> np.ndarray:
        """Generate a linear trend component."""
        t = np.arange(self.config.length)
        return slope * t
    
    def generate_noise(self, noise_type: str = 'gaussian') -> np.ndarray:
        """Generate noise component."""
        if noise_type == 'gaussian':
            return self.rng.normal(0, self.config.noise_level, self.config.length)
        elif noise_type == 'student_t':
            # Heavy-tailed noise
            return self.rng.standard_t(df=3, size=self.config.length) * self.config.noise_level
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def generate_anomalies(self, anomaly_type: str = 'spike') -> np.ndarray:
        """Generate anomaly component."""
        anomalies = np.zeros(self.config.length)
        n_anomalies = int(self.config.length * self.config.anomaly_probability)
        
        if n_anomalies > 0:
            anomaly_indices = self.rng.choice(
                self.config.length, 
                size=n_anomalies, 
                replace=False
            )
            
            if anomaly_type == 'spike':
                anomaly_values = self.rng.normal(0, 3, n_anomalies)
            elif anomaly_type == 'level_shift':
                anomaly_values = self.rng.choice([-2, 2], n_anomalies)
            else:
                anomaly_values = self.rng.normal(0, 2, n_anomalies)
                
            anomalies[anomaly_indices] = anomaly_values
            
        return anomalies
    
    def generate_complex_series(self, 
                              components: List[str] = None,
                              **kwargs) -> pd.Series:
        """Generate a complex time series with multiple components."""
        if components is None:
            components = ['trend', 'seasonality', 'noise']
            
        series = np.zeros(self.config.length)
        
        if 'trend' in components:
            trend = self.generate_trend(kwargs.get('trend_slope', 0.001))
            series += trend * self.config.trend_strength
            
        if 'seasonality' in components:
            # Multiple seasonal components
            daily_seasonality = self.generate_sine_wave(0.5, 7)  # Weekly
            yearly_seasonality = self.generate_sine_wave(1.0, 365)  # Yearly
            series += (daily_seasonality + yearly_seasonality) * self.config.seasonality_strength
            
        if 'noise' in components:
            noise = self.generate_noise(kwargs.get('noise_type', 'gaussian'))
            series += noise
            
        if 'anomalies' in components:
            anomalies = self.generate_anomalies(kwargs.get('anomaly_type', 'spike'))
            series += anomalies
            
        # Create pandas Series with datetime index
        date_range = pd.date_range(
            start=self.config.start_date,
            periods=self.config.length,
            freq=self.config.frequency
        )
        
        return pd.Series(series, index=date_range, name='value')
    
    def generate_multiple_series(self, n_series: int = 5) -> Dict[str, pd.Series]:
        """Generate multiple time series with different characteristics."""
        series_dict = {}
        
        # Different types of series
        series_types = [
            {'components': ['trend', 'seasonality', 'noise'], 'name': 'trend_seasonal'},
            {'components': ['seasonality', 'noise'], 'name': 'seasonal_only'},
            {'components': ['trend', 'noise'], 'name': 'trend_only'},
            {'components': ['seasonality', 'noise', 'anomalies'], 'name': 'seasonal_anomalies'},
            {'components': ['trend', 'seasonality', 'noise', 'anomalies'], 'name': 'complex'}
        ]
        
        for i, series_config in enumerate(series_types[:n_series]):
            series = self.generate_complex_series(**series_config)
            series_dict[f"series_{i+1}_{series_config['name']}"] = series
            
        return series_dict


def create_ssl_dataset(series: pd.Series, 
                      seq_len: int = 50, 
                      mask_prob: float = 0.2,
                      stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create dataset for self-supervised learning with masked values.
    
    Args:
        series: Input time series
        seq_len: Length of sequences
        mask_prob: Probability of masking each value
        stride: Step size for creating sequences
        
    Returns:
        Tuple of (masked_sequences, original_sequences, mask_indicators)
    """
    data = series.values
    X, Y, M = [], [], []
    
    for i in range(0, len(data) - seq_len, stride):
        seq = data[i:i+seq_len]
        
        # Create random mask
        mask = np.random.rand(seq_len) < mask_prob
        masked_seq = seq.copy()
        masked_seq[mask] = 0.0  # Simple masking strategy
        
        X.append(masked_seq)
        Y.append(seq)
        M.append(mask.astype(float))
        
    return np.array(X), np.array(Y), np.array(M)


def create_forecasting_dataset(series: pd.Series,
                             input_len: int = 50,
                             output_len: int = 10,
                             stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset for forecasting tasks.
    
    Args:
        series: Input time series
        input_len: Length of input sequences
        output_len: Length of output sequences to predict
        stride: Step size for creating sequences
        
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    data = series.values
    X, Y = [], []
    
    for i in range(0, len(data) - input_len - output_len + 1, stride):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+output_len])
        
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    # Example usage
    config = TimeSeriesConfig(length=1000, noise_level=0.1)
    generator = TimeSeriesGenerator(config)
    
    # Generate a complex time series
    series = generator.generate_complex_series()
    print(f"Generated series with {len(series)} points")
    print(f"Series statistics: mean={series.mean():.3f}, std={series.std():.3f}")
    
    # Create SSL dataset
    X, Y, M = create_ssl_dataset(series, seq_len=50, mask_prob=0.2)
    print(f"SSL dataset shapes: X={X.shape}, Y={Y.shape}, M={M.shape}")
    
    # Create forecasting dataset
    X_fcst, Y_fcst = create_forecasting_dataset(series, input_len=50, output_len=10)
    print(f"Forecasting dataset shapes: X={X_fcst.shape}, Y={Y_fcst.shape}")
