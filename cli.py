#!/usr/bin/env python3
"""
Command-line interface for time series SSL project.

This script provides easy access to common operations like training models,
generating data, and launching the web interface.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import create_default_config, ConfigManager
from data_generation import TimeSeriesGenerator, TimeSeriesConfig
from models import create_model


def generate_data(args):
    """Generate synthetic time series data."""
    print("Generating synthetic time series data...")
    
    config = TimeSeriesConfig(
        length=args.length,
        noise_level=args.noise_level,
        trend_strength=args.trend_strength,
        seasonality_strength=args.seasonality_strength
    )
    
    generator = TimeSeriesGenerator(config)
    series = generator.generate_complex_series()
    
    # Save data
    output_path = Path(args.output) / "synthetic_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(output_path)
    
    print(f"Generated {len(series)} data points")
    print(f"Data saved to: {output_path}")
    print(f"Statistics: mean={series.mean():.3f}, std={series.std():.3f}")


def train_model(args):
    """Train SSL model."""
    print("Training SSL model...")
    
    # Load or create configuration
    if args.config and Path(args.config).exists():
        config = ConfigManager.load_yaml(args.config)
    else:
        config = create_default_config()
        config.experiment_name = args.experiment_name
        config.training.num_epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.learning_rate
    
    print(f"Configuration: {config.to_dict()}")
    
    # Import training function
    from train import main as train_main
    
    # Override sys.argv for training script
    sys.argv = ['train.py', '--experiment_name', config.experiment_name]
    if args.config:
        sys.argv.extend(['--config', args.config])
    
    train_main()


def launch_ui(args):
    """Launch Streamlit web interface."""
    print("Launching Streamlit web interface...")
    
    import subprocess
    streamlit_path = Path(__file__).parent / "src" / "streamlit_app.py"
    
    cmd = ["streamlit", "run", str(streamlit_path)]
    if args.port:
        cmd.extend(["--server.port", str(args.port)])
    
    subprocess.run(cmd)


def run_tests(args):
    """Run test suite."""
    print("Running test suite...")
    
    import subprocess
    test_path = Path(__file__).parent / "tests" / "test_ssl.py"
    
    cmd = ["pytest", str(test_path), "-v"]
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html"])
    
    subprocess.run(cmd)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Time Series Self-Supervised Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate --length 1000 --output ./data
  %(prog)s train --epochs 50 --experiment_name my_experiment
  %(prog)s ui --port 8501
  %(prog)s test --coverage
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--length', type=int, default=1000, help='Data length')
    gen_parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level')
    gen_parser.add_argument('--trend-strength', type=float, default=0.5, help='Trend strength')
    gen_parser.add_argument('--seasonality-strength', type=float, default=1.0, help='Seasonality strength')
    gen_parser.add_argument('--output', type=str, default='./data', help='Output directory')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train SSL model')
    train_parser.add_argument('--config', type=str, help='Configuration file')
    train_parser.add_argument('--experiment-name', type=str, default='ssl_experiment', help='Experiment name')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    # Launch UI command
    ui_parser = subparsers.add_parser('ui', help='Launch web interface')
    ui_parser.add_argument('--port', type=int, default=8501, help='Port number')
    
    # Run tests command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'generate':
            generate_data(args)
        elif args.command == 'train':
            train_model(args)
        elif args.command == 'ui':
            launch_ui(args)
        elif args.command == 'test':
            run_tests(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
