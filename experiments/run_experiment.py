
"""
Enhanced experiment runner with NumPy 2.0 compatibility and improved reproducibility.
"""
import argparse
import logging
import json
from pathlib import Path
import time
from typing import Dict, Any
import torch
import wandb
import numpy as np
import random
import os

from mind.data.loader import load_and_align_data, find_most_active_neurons
from mind.data.processor import create_datasets
from mind.training.trainer import train_model
from mind.visualization.comprehensive_viz import create_all_visualizations
from mind.config import get_config
from mind.utils.logging import setup_logging

# Import the modified CNN model if available
try:
    from mind.models.deep.modified_cnn import ModifiedCNNWrapper

    HAS_MODIFIED_CNN = True
except ImportError:
    HAS_MODIFIED_CNN = False

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42):
    """
    Set all random seeds for complete reproducibility across all libraries.

    This function ensures consistent results by setting seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and GPU seeds
    - CUDA deterministic operations

    Parameters
    ----------
    seed : int
        Random seed to use across all libraries
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for some operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"All random seeds set to {seed} for reproducibility")


def deep_convert_to_json_serializable(obj):
    """
    Recursively convert numpy arrays and other non-JSON serializable objects to JSON-serializable format.
    This version is compatible with both NumPy 1.x and 2.x.

    The function handles:
    - NumPy arrays → Python lists
    - NumPy scalars → Python scalars
    - Complex nested structures
    - Custom objects with __dict__ attributes

    Parameters
    ----------
    obj : any
        Object to convert

    Returns
    -------
    any
        JSON-serializable object
    """
    # Check if it's a numpy array first
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Check if it's a numpy scalar using numpy's own type checking
    # This approach is compatible with both NumPy 1.x and 2.x
    elif isinstance(obj, np.generic):
        # numpy.generic is the base class for all numpy scalar types
        return obj.item()

    # Standard Python types
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float)):
        return obj

    # Collections
    elif isinstance(obj, dict):
        return {key: deep_convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [deep_convert_to_json_serializable(item) for item in obj]

    # None
    elif obj is None:
        return None

    # Custom objects
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting their __dict__
        return deep_convert_to_json_serializable(obj.__dict__)

    # Last resort: try to convert to string
    else:
        try:
            return str(obj)
        except:
            logger.warning(f"Could not convert object of type {type(obj)} to JSON-serializable format")
            return None


def run_single_experiment(model_type: str, signal_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment for one model and signal type with reproducibility guarantees.

    This function:
    1. Loads data with consistent binary classification
    2. Creates datasets with consistent random splits
    3. Trains the model with fixed random seeds
    4. Returns results in a JSON-serializable format

    Parameters
    ----------
    model_type : str
        Type of model to train ('random_forest', 'svm', 'mlp', 'fcnn', 'cnn', 'modified_cnn')
    signal_type : str
        Type of signal to use ('calcium_signal', 'deltaf_signal', 'deconv_signal')
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Results dictionary with all arrays converted to lists
    """
    logger.info(f"Running experiment: {model_type} on {signal_type}")

    # Load data with binary classification
    calcium_signals, frame_labels = load_and_align_data(
        mat_file_path=config["data"]["mat_file"],
        xlsx_file_path=config["data"]["xlsx_file"],
        binary_classification=True  # Force binary classification
    )

    # Verify we have only binary labels
    unique_labels = np.unique(frame_labels)
    logger.info(f"Unique labels in data: {unique_labels}")
    if len(unique_labels) > 2 or max(unique_labels) > 1:
        logger.error("Data contains more than binary labels! Check binary classification handling.")
        return {}

    # Create datasets with consistent random seed
    random_state = config["models"].get(model_type, {}).get("random_state", 42)
    datasets = create_datasets(
        calcium_signals=calcium_signals,
        frame_labels=frame_labels,
        window_size=config["data"]["window_size"],
        step_size=config["data"]["step_size"],
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=random_state
    )

    # Check if signal type exists
    if signal_type not in datasets:
        logger.error(f"Signal type {signal_type} not found in datasets")
        return {}

    # Get dimensions
    window_size = config["data"]["window_size"]
    sample, _ = datasets[signal_type]['train'][0]
    n_neurons = sample.shape[1]

    logger.info(f"Data dimensions - Window size: {window_size}, Neurons: {n_neurons}")

    # Train model (use modified CNN if requested and available)
    if model_type == 'modified_cnn' and HAS_MODIFIED_CNN:
        logger.info("Using modified CNN model")

        # Extract data from datasets
        X_train = torch.stack([X for X, _ in datasets[signal_type]['train']])
        y_train = torch.tensor([y.item() for _, y in datasets[signal_type]['train']])
        X_val = torch.stack([X for X, _ in datasets[signal_type]['val']])
        y_val = torch.tensor([y.item() for _, y in datasets[signal_type]['val']])
        X_test = torch.stack([X for X, _ in datasets[signal_type]['test']])
        y_test = torch.tensor([y.item() for _, y in datasets[signal_type]['test']])

        # Initialize and train the modified CNN model
        device = config["training"]["device"]
        model = ModifiedCNNWrapper(
            window_size=window_size,
            n_neurons=n_neurons,
            device=device,
            random_state=config.get("seed", 42)
        )

        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate model
        from mind.evaluation.metrics import evaluate_model
        from mind.evaluation.feature_importance import extract_feature_importance, create_importance_summary

        eval_results = evaluate_model(model, X_test, y_test)

        # Extract feature importance
        importance_matrix = extract_feature_importance(model, window_size, n_neurons)
        importance_summary = create_importance_summary(importance_matrix, window_size, n_neurons)

        # Get top 100 contributing neurons
        top_100_neurons = model.get_top_contributing_neurons(n_top=100)
        importance_summary['top_100_neurons'] = top_100_neurons.tolist()

        # Create results dictionary
        results = {
            'metrics': eval_results['metrics'],
            'confusion_matrix': eval_results['confusion_matrix'].tolist(),
            'importance_summary': importance_summary,
            'top_100_neurons': top_100_neurons.tolist()
        }

    else:
        # Use standard training pipeline
        model_params = config["models"].get(model_type, {})
        results = train_model(
            model_type=model_type,
            model_params=model_params,
            datasets=datasets,
            signal_type=signal_type,
            window_size=window_size,
            n_neurons=n_neurons,
            output_dir=config["training"]["output_dir"],
            device=config["training"]["device"],
            optimize_hyperparams=config["training"]["optimize_hyperparams"],
            use_wandb=config["wandb"]["use_wandb"]
        )

    return results


def run_all_experiments(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Run all model-signal combinations with consistent random seeds.

    This ensures that the entire experiment suite is reproducible when using
    the same seed value.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dictionary of results
    """
    # Include modified_cnn if available
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    if HAS_MODIFIED_CNN:
        models.append('modified_cnn')

    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    results = {}

    for model in models:
        results[model] = {}
        for signal in signals:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Starting: {model} on {signal}")
            logger.info(f"{'=' * 50}\n")

            if config["wandb"]["use_wandb"]:
                wandb.init(
                    project=config["wandb"]["project_name"],
                    name=f"{model}_{signal}",
                    config={
                        "model": model,
                        "signal": signal,
                        **config
                    },
                    reinit=True
                )

            try:
                result = run_single_experiment(model, signal, config)
                results[model][signal] = result

                # Log summary metrics to W&B
                if config["wandb"]["use_wandb"] and 'metrics' in result:
                    wandb.log(result['metrics'])
                    wandb.finish()

            except Exception as e:
                logger.error(f"Error running {model} on {signal}: {e}")
                results[model][signal] = {}  # Add empty result instead of skipping
                if config["wandb"]["use_wandb"]:
                    wandb.finish()
                continue

    return results


def main():
    """
    Main entry point for experiments with reproducibility and robust JSON handling.

    This function:
    1. Parses command-line arguments
    2. Sets up logging and random seeds
    3. Runs experiments with consistent configuration
    4. Saves results in JSON format
    5. Optionally creates visualizations
    """
    # Set up argument parser with additional options
    model_choices = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn', 'all']
    if HAS_MODIFIED_CNN:
        model_choices.append('modified_cnn')

    parser = argparse.ArgumentParser(description="Run neural decoding experiments")
    parser.add_argument("--model", type=str,
                        choices=model_choices,
                        default='all', help="Model to run (or 'all')")
    parser.add_argument("--signal", type=str,
                        choices=['calcium_signal', 'deltaf_signal', 'deconv_signal', 'all'],
                        default='all', help="Signal type to use (or 'all')")
    parser.add_argument("--data", type=str, help="Path to .mat file")
    parser.add_argument("--behavior", type=str, help="Path to .xlsx file")
    parser.add_argument("--output", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--viz-only", action="store_true", help="Only create visualizations from existing results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top neurons to visualize")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level='INFO', console=True)

    # Set all random seeds for reproducibility
    set_all_seeds(args.seed)

    # Get configuration
    config = get_config()

    # Add the seed to the config
    config["seed"] = args.seed

    # Update configuration from command line
    if args.data:
        config["data"]["mat_file"] = args.data
    if args.behavior:
        config["data"]["xlsx_file"] = args.behavior
    config["training"]["output_dir"] = args.output
    config["training"]["optimize_hyperparams"] = args.optimize
    config["wandb"]["use_wandb"] = not args.no_wandb

    # Update all model configurations with the same seed for consistency
    for model in config["models"]:
        config["models"][model]["random_state"] = args.seed

    # Check if we're only visualizing
    if args.viz_only:
        # Load existing results
        results_path = Path(args.output) / "all_results.json"
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            return

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Load calcium signals for visualization
        calcium_signals, _ = load_and_align_data(
            mat_file_path=config["data"]["mat_file"],
            xlsx_file_path=config["data"]["xlsx_file"],
            binary_classification=True
        )

        # Create visualizations
        viz_dir = Path(args.output) / "visualizations"

        # Pass the mat_file_path for neuron bubble charts
        create_all_visualizations(
            results=results,
            calcium_signals=calcium_signals,
            output_dir=viz_dir,
            mat_file_path=config["data"]["mat_file"],
            top_n=args.top_n
        )
        return

    # Run experiments
    if args.model == 'all' and args.signal == 'all':
        # Run all experiments
        results = run_all_experiments(config)

        # Save all results
        output_dir = Path(config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the enhanced deep conversion function to handle all nested numpy arrays
        json_results = deep_convert_to_json_serializable(results)

        # Save to JSON
        try:
            with open(output_dir / "all_results.json", 'w') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved all results to {output_dir / 'all_results.json'}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            # If there's still an issue, try to identify the problematic data
            logger.error("Attempting to identify non-serializable data...")
            for model, model_results in results.items():
                for signal, signal_results in model_results.items():
                    for key, value in signal_results.items():
                        try:
                            json.dumps(deep_convert_to_json_serializable(value))
                        except Exception as inner_e:
                            logger.error(f"Issue with {model}/{signal}/{key}: {inner_e}")
                            logger.error(f"Type: {type(value)}")

        # Create visualizations if requested
        if args.visualize:
            viz_dir = output_dir / "visualizations"

            # Load calcium signals directly from the data files
            calcium_signals, _ = load_and_align_data(
                mat_file_path=config["data"]["mat_file"],
                xlsx_file_path=config["data"]["xlsx_file"],
                binary_classification=True
            )

            # Pass the mat_file_path for neuron bubble charts
            create_all_visualizations(
                results=json_results,
                calcium_signals=calcium_signals,
                output_dir=viz_dir,
                mat_file_path=config["data"]["mat_file"],
                top_n=args.top_n
            )

    elif args.model == 'all' or args.signal == 'all':
        # Run one model on all signals or all models on one signal
        results = {}

        if args.model == 'all':
            # Run all models on one signal
            models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
            if HAS_MODIFIED_CNN:
                models.append('modified_cnn')
            signals = [args.signal]
        else:
            # Run one model on all signals
            models = [args.model]
            signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

        # Initialize results structure
        for model in models:
            results[model] = {}

        # Run experiments
        for model in models:
            for signal in signals:
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Starting: {model} on {signal}")
                logger.info(f"{'=' * 50}\n")

                try:
                    result = run_single_experiment(model, signal, config)
                    results[model][signal] = result
                except Exception as e:
                    logger.error(f"Error running {model} on {signal}: {e}")
                    results[model][signal] = {}

        # Save results
        output_dir = Path(config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        json_results = deep_convert_to_json_serializable(results)

        with open(output_dir / "results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved results to {output_dir / 'results.json'}")

        # Create visualizations if requested
        if args.visualize:
            viz_dir = output_dir / "visualizations"

            # Load calcium signals
            calcium_signals, _ = load_and_align_data(
                mat_file_path=config["data"]["mat_file"],
                xlsx_file_path=config["data"]["xlsx_file"],
                binary_classification=True
            )

            # Create visualizations
            create_all_visualizations(
                results=json_results,
                calcium_signals=calcium_signals,
                output_dir=viz_dir,
                mat_file_path=config["data"]["mat_file"],
                top_n=args.top_n
            )

    else:
        # Run a single experiment
        result = run_single_experiment(args.model, args.signal, config)

        # Save result
        output_dir = Path(config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{args.model}_{args.signal}_{timestamp}.json"

        # Convert to JSON-serializable format
        json_result = deep_convert_to_json_serializable(result)

        with open(output_dir / filename, 'w') as f:
            json.dump(json_result, f, indent=2)

        logger.info(f"Saved result to {output_dir / filename}")


if __name__ == "__main__":
    main()

