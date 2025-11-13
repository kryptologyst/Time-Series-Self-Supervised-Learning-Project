# Time Series Self-Supervised Learning Project

A comprehensive implementation of self-supervised learning (SSL) models for time series analysis, featuring state-of-the-art architectures including Transformers, CNNs, and hybrid approaches.

## Overview

This project implements various SSL techniques for time series data, focusing on masked value prediction (similar to BERT for text). The models learn meaningful temporal representations that can be fine-tuned for downstream tasks like forecasting, anomaly detection, or classification.

### Key Features

- **Multiple SSL Architectures**: Transformer, CNN, Hybrid CNN-Transformer, and Contrastive models
- **Realistic Data Generation**: Synthetic time series with trends, seasonality, noise, and anomalies
- **Interactive Visualization**: Streamlit web interface for exploring results
- **Comprehensive Configuration**: YAML-based configuration management
- **Checkpoint Management**: Model saving and loading with experiment tracking
- **Unit Testing**: Comprehensive test suite for all components
- **Modern Python Practices**: Type hints, docstrings, PEP8 compliance

## Project Structure

```
├── src/                    # Source code
│   ├── data_generation.py  # Synthetic time series generation
│   ├── models.py           # SSL model architectures
│   ├── config.py           # Configuration management
│   └── streamlit_app.py    # Interactive web interface
├── config/                 # Configuration files
│   └── default.yaml       # Default configuration
├── notebooks/              # Jupyter notebooks
│   └── ssl_analysis.ipynb # Analysis notebook
├── tests/                  # Unit tests
│   └── test_ssl.py        # Test suite
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Time-Series-Self-Supervised-Learning-Project.git
   cd Time-Series-Self-Supervised-Learning-Project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Basic Training

Train a Transformer-based SSL model with default settings:

```bash
python train.py --experiment_name "my_experiment"
```

### 2. Custom Configuration

Create a custom configuration file and train:

```bash
python train.py --config config/custom.yaml
```

### 3. Interactive Analysis

Launch the Streamlit web interface:

```bash
streamlit run src/streamlit_app.py
```

### 4. Jupyter Notebook

Open the analysis notebook:

```bash
jupyter notebook notebooks/ssl_analysis.ipynb
```

## Usage Examples

### Data Generation

```python
from src.data_generation import TimeSeriesGenerator, TimeSeriesConfig

# Create configuration
config = TimeSeriesConfig(
    length=1000,
    noise_level=0.1,
    trend_strength=0.5,
    seasonality_strength=1.0
)

# Generate time series
generator = TimeSeriesGenerator(config)
series = generator.generate_complex_series()

# Create SSL dataset
from src.data_generation import create_ssl_dataset
X, Y, M = create_ssl_dataset(series, seq_len=50, mask_prob=0.2)
```

### Model Training

```python
from src.models import create_model, SSLTrainer
import torch

# Create model
model = create_model('transformer', input_dim=1, d_model=128, nhead=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = SSLTrainer(model, optimizer, device='cpu', mask_prob=0.2)

# Training loop
for epoch in range(100):
    metrics = trainer.train_step(batch_data)
    print(f"Epoch {epoch+1}, Loss: {metrics['loss']:.4f}")
```

### Configuration Management

```python
from src.config import create_default_config, ConfigManager

# Create and save configuration
config = create_default_config()
config.experiment_name = "my_experiment"
config.training.num_epochs = 200

ConfigManager.save_yaml(config, "my_config.yaml")

# Load configuration
loaded_config = ConfigManager.load_yaml("my_config.yaml")
```

## Model Architectures

### 1. Transformer SSL Model

- **Architecture**: Multi-head self-attention with positional encoding
- **Best for**: Long-range dependencies, complex temporal patterns
- **Parameters**: `d_model`, `nhead`, `num_layers`, `dim_feedforward`

### 2. CNN SSL Model

- **Architecture**: Convolutional encoder-decoder with batch normalization
- **Best for**: Local patterns, computational efficiency
- **Parameters**: `hidden_dims`, `kernel_sizes`, `dropout`

### 3. Hybrid SSL Model

- **Architecture**: CNN feature extractor + Transformer processing
- **Best for**: Combining local and global patterns
- **Parameters**: CNN and Transformer parameters

### 4. Contrastive SSL Model

- **Architecture**: Encoder with projection head for contrastive learning
- **Best for**: Learning robust representations
- **Parameters**: `hidden_dim`, `projection_dim`

## Configuration Options

### Model Configuration

```yaml
model:
  model_type: transformer  # transformer, cnn, hybrid, contrastive
  input_dim: 1
  d_model: 128
  nhead: 8
  num_layers: 6
  dropout: 0.1
```

### Training Configuration

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  mask_prob: 0.2
  seq_len: 50
  device: auto  # cpu, cuda, auto
```

### Data Configuration

```yaml
data:
  length: 1000
  noise_level: 0.1
  trend_strength: 0.5
  seasonality_strength: 1.0
  anomaly_probability: 0.05
```

## Running Tests

Execute the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_ssl.py::TestDataGeneration -v
pytest tests/test_ssl.py::TestModels -v
pytest tests/test_ssl.py::TestConfiguration -v
```

## Performance Tips

1. **GPU Usage**: Set `device: cuda` in configuration for faster training
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Sequence Length**: Longer sequences capture more temporal patterns but require more memory
4. **Model Size**: Larger models (higher `d_model`, more layers) may perform better but train slower

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Import Errors**: Ensure all dependencies are installed and src/ is in Python path
3. **Configuration Errors**: Check YAML syntax and parameter values

### Debug Mode

Enable debug logging:

```python
from src.config import Logger
logger_manager = Logger(log_level='DEBUG')
logger = logger_manager.get_logger()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{time_series_ssl,
  title={Time Series Self-Supervised Learning},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Time-Series-Self-Supervised-Learning-Project}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework
- The time series analysis community for inspiration and best practices
# Time-Series-Self-Supervised-Learning-Project
