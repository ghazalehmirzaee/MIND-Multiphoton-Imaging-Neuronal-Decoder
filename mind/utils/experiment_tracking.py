"""Experiment tracking utility functions."""
from typing import Dict, Any, List, Optional, Union
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import wandb

logger = logging.getLogger(__name__)


def log_metrics(
        wandb_run: Any,
        metrics: Dict[str, float]
) -> None:
    """
    Log metrics to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    metrics : Dict[str, float]
        Dictionary containing metrics to log
    """
    wandb_run.log(metrics)


def log_figures(
        wandb_run: Any,
        figures: Dict[str, plt.Figure]
) -> None:
    """
    Log figures to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    figures : Dict[str, plt.Figure]
        Dictionary containing figures to log
    """
    for name, fig in figures.items():
        wandb_run.log({name: wandb.Image(fig)})


def log_model(
        wandb_run: Any,
        model_path: str,
        model_name: str
) -> None:
    """
    Log model to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    model_path : str
        Path to model file
    model_name : str
        Name of the model
    """
    artifact = wandb.Artifact(
        name=model_name,
        type='model',
        description=f'Trained {model_name} model'
    )
    artifact.add_file(model_path)
    wandb_run.log_artifact(artifact)


def log_results(
        wandb_run: Any,
        results_path: str,
        results_name: str
) -> None:
    """
    Log results to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    results_path : str
        Path to results file
    results_name : str
        Name of the results
    """
    artifact = wandb.Artifact(
        name=results_name,
        type='results',
        description=f'{results_name} results'
    )
    artifact.add_file(results_path)
    wandb_run.log_artifact(artifact)


def init_wandb(
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Initialize Weights & Biases run.

    Parameters
    ----------
    project_name : str
        Project name
    experiment_name : Optional[str], optional
        Experiment name, by default None
    config : Optional[Dict[str, Any]], optional
        Configuration dictionary, by default None

    Returns
    -------
    Any
        Weights & Biases run
    """
    # Generate a unique experiment name if not provided
    if experiment_name is None:
        experiment_name = f"experiment_{int(time.time())}"

    # Initialize run
    wandb_run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=config
    )

    return wandb_run


def save_config(
        config: Dict[str, Any],
        output_file: str = 'config.json'
) -> None:
    """
    Save configuration to JSON file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    output_file : str, optional
        Output file path, by default 'config.json'
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save configuration to JSON file
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(
        input_file: str = 'config.json'
) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Parameters
    ----------
    input_file : str, optional
        Input file path, by default 'config.json'

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    # Check if file exists
    if not os.path.exists(input_file):
        logger.warning(f"Configuration file {input_file} not found")
        return {}

    # Load configuration from JSON file
    with open(input_file, 'r') as f:
        config = json.load(f)

    return config
