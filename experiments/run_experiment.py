"""
Simplified experiment runner.
"""
import argparse
import os
import sys
import numpy as np
import logging
from pathlib import Path
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mind.data.loader import load_and_align_data
from mind.data.processor import create_datasets
from mind.training.trainer import train_model
from mind.config import get_config
from mind.utils.logging import setup_logging

# Set up logging
setup_logging(log_level='INFO', console=True)
logger = logging.getLogger(__name__)


def main():
    """
    Run a single experiment with specified model and signal type.
    """
    # Get default configuration
    config = get_config()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run neural decoding experiment")
    parser.add_argument("--model", type=str, required=True,
                        choices=['random_forest', 'svm', 'mlp', 'fcnn', 'cnn'],
                        help="Model type")
    parser.add_argument("--signal", type=str, required=True,
                        choices=['calcium_signal', 'deltaf_signal', 'deconv_signal'],
                        help="Signal type")
    parser.add_argument("--data", type=str, default=config["data"]["mat_file"],
                        help="Path to .mat file")
    parser.add_argument("--behavior", type=str, default=config["data"]["xlsx_file"],
                        help="Path to behavior Excel file")
    parser.add_argument("--output", type=str, default=config["training"]["output_dir"],
                        help="Output directory")
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize hyperparameters")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B tracking")

    args = parser.parse_args()

    # Update configuration with CLI arguments
    config["data"]["mat_file"] = args.data
    config["data"]["xlsx_file"] = args.behavior
    config["training"]["output_dir"] = args.output
    config["training"]["optimize_hyperparams"] = args.optimize
    config["wandb"]["use_wandb"] = not args.no_wandb

    # Initialize W&B
    if config["wandb"]["use_wandb"]:
        wandb.init(
            project=config["wandb"]["project_name"],
            entity=config["wandb"]["entity"],
            config={
                "model": args.model,
                "signal_type": args.signal,
                "optimize_hyperparams": config["training"]["optimize_hyperparams"],
                "window_size": config["data"]["window_size"],
                "step_size": config["data"]["step_size"],
                "model_params": config["models"][args.model]
            }
        )

    try:
        # Load data
        logger.info(f"Loading data from {args.data} and {args.behavior}")
        calcium_signals, frame_labels = load_and_align_data(
            mat_file_path=args.data,
            xlsx_file_path=args.behavior,
            binary_classification=config["data"]["binary_classification"]
        )

        # Create datasets
        logger.info("Creating datasets")
        datasets = create_datasets(
            calcium_signals=calcium_signals,
            frame_labels=frame_labels,
            window_size=config["data"]["window_size"],
            step_size=config["data"]["step_size"],
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            random_state=config["models"][args.model]["random_state"]
        )

        # Get window size and number of neurons
        window_size = config["data"]["window_size"]
        if args.signal in datasets and 'train' in datasets[args.signal]:
            sample, _ = datasets[args.signal]['train'][0]
            n_neurons = sample.shape[1]
        else:
            n_neurons = calcium_signals[args.signal].shape[1]

        logger.info(f"Window size: {window_size}, Number of neurons: {n_neurons}")

        # Train model
        results = train_model(
            model_type=args.model,
            model_params=config["models"][args.model],
            datasets=datasets,
            signal_type=args.signal,
            window_size=window_size,
            n_neurons=n_neurons,
            output_dir=args.output,
            device=config["training"]["device"],
            optimize_hyperparams=config["training"]["optimize_hyperparams"],
            use_wandb=config["wandb"]["use_wandb"]
        )

        logger.info(f"Experiment completed successfully")

    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)
        raise

    finally:
        # Finish W&B
        if config["wandb"]["use_wandb"]:
            wandb.finish()


if __name__ == "__main__":
    main()

