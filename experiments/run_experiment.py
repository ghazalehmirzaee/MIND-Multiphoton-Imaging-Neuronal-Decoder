# experiments/run_experiment.py
"""
Single model and signal type experiment runner.
"""

import argparse
import logging
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import random

from mind.data.loader import load_and_align_data
from mind.data.processor import create_datasets
from mind.training.trainer import train_model
from mind.visualization.comprehensive_viz import create_all_visualizations
from mind.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

def run_experiment(model_type, signal_type, config):
    """Run experiment for a single model and signal type."""
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Load data with binary classification
    logger.info(f"Loading data from {config['data']['mat_file']} and {config['data']['xlsx_file']}")
    try:
        calcium_signals, frame_labels = load_and_align_data(
            mat_file_path=config["data"]["mat_file"],
            xlsx_file_path=config["data"]["xlsx_file"],
            binary_classification=True
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

    # Create datasets
    logger.info("Creating datasets")
    try:
        datasets = create_datasets(
            calcium_signals=calcium_signals,
            frame_labels=frame_labels,
            window_size=config["data"]["window_size"],
            step_size=config["data"]["step_size"],
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            random_state=config["models"][model_type]["random_state"]
        )
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return None

    # Train model
    logger.info(f"Training {model_type} on {signal_type}")
    try:
        # Get dimensions
        window_size = config["data"]["window_size"]
        n_neurons = datasets[signal_type]['train'][0][0].shape[1]

        # Get model parameters
        model_params = config["models"].get(model_type, {})

        # Train model
        results = train_model(
            model_type=model_type,
            model_params=model_params,
            datasets=datasets,
            signal_type=signal_type,
            window_size=window_size,
            n_neurons=n_neurons,
            output_dir=config["training"]["output_dir"],
            device=config["training"]["device"],
            optimize_hyperparams=config["training"]["optimize_hyperparams"]
        )

        return results

    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def main():
    """Main function to run experiments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run neural decoding experiments")
    parser.add_argument("--model", type=str, choices=['random_forest', 'svm', 'mlp', 'fcnn', 'cnn', 'all'],
                        default='all', help="Model to run (default: all)")
    parser.add_argument("--signal", type=str, choices=['calcium_signal', 'deltaf_signal', 'deconv_signal', 'all'],
                        default='all', help="Signal type to use (default: all)")
    parser.add_argument("--data", type=str, help="Path to .mat file")
    parser.add_argument("--behavior", type=str, help="Path to .xlsx file")
    parser.add_argument("--output", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Get configuration
    config = get_config()

    # Update configuration from command line
    if args.data:
        config["data"]["mat_file"] = args.data
    if args.behavior:
        config["data"]["xlsx_file"] = args.behavior
    config["training"]["output_dir"] = args.output
    config["training"]["optimize_hyperparams"] = args.optimize
    config["seed"] = args.seed

    # Set seed once for experiment
    set_seed(args.seed)

    # Determine which models and signals to run
    if args.model == 'all':
        models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    else:
        models = [args.model]

    if args.signal == 'all':
        signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    else:
        signals = [args.signal]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {}
    for model in models:
        all_results[model] = {}
        for signal in signals:
            logger.info(f"Running experiment: {model} on {signal}")
            results = run_experiment(model, signal, config)
            all_results[model][signal] = results

    # Save results
    results_path = output_dir / "all_results.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        elif hasattr(obj, 'numpy'):
            return obj.numpy().tolist()
        else:
            return str(obj)

    json_results = convert_to_json_serializable(all_results)

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Create visualizations if requested
    if args.visualize:
        try:
            logger.info("Creating visualizations")

            # Load calcium signals for visualization
            calcium_signals, _ = load_and_align_data(
                mat_file_path=config["data"]["mat_file"],
                xlsx_file_path=config["data"]["xlsx_file"],
                binary_classification=True
            )

            viz_dir = output_dir / "visualizations"
            create_all_visualizations(
                results=json_results,
                calcium_signals=calcium_signals,
                output_dir=str(viz_dir),
                mat_file_path=config["data"]["mat_file"]
            )

            logger.info(f"Visualizations saved to {viz_dir}")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    main()

    
