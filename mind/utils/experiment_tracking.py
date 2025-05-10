"""Experiment tracking utility functions."""
from typing import Dict, Any, List, Optional, Union
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import datetime
import wandb
from omegaconf import DictConfig, OmegaConf

# Set up logger
logger = logging.getLogger(__name__)


def log_metrics(
        wandb_run: Any,
        metrics: Dict[str, float],
        step: Optional[int] = None
) -> None:
    """
    Log metrics to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    metrics : Dict[str, float]
        Dictionary containing metrics to log
    step : Optional[int], optional
        Step for logging metrics, by default None
    """
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def log_figures(
        wandb_run: Any,
        figures: Dict[str, plt.Figure],
        step: Optional[int] = None
) -> None:
    """
    Log figures to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    figures : Dict[str, plt.Figure]
        Dictionary containing figures to log
    step : Optional[int], optional
        Step for logging figures, by default None
    """
    if wandb_run is not None:
        for name, fig in figures.items():
            wandb_run.log({name: wandb.Image(fig)}, step=step)


def log_artifact(
        wandb_run: Any,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str = 'model'
) -> None:
    """
    Log artifact to Weights & Biases.

    Parameters
    ----------
    wandb_run : Any
        Weights & Biases run
    artifact_path : str
        Path to artifact file
    artifact_name : str
        Name of the artifact
    artifact_type : str, optional
        Type of the artifact, by default 'model'
    """
    if wandb_run is not None:
        try:
            # Add timestamp to artifact name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            full_artifact_name = f"{artifact_name}_{timestamp}"

            # Create artifact
            artifact = wandb.Artifact(
                name=full_artifact_name,
                type=artifact_type,
                description=f'{artifact_name} {artifact_type}'
            )

            # Add file to artifact
            artifact.add_file(artifact_path)

            # Log artifact
            wandb_run.log_artifact(artifact)

            logger.info(f"Logged artifact {full_artifact_name} to W&B")
        except Exception as e:
            logger.error(f"Error logging artifact to W&B: {e}")


def init_wandb(
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], DictConfig]] = None,
        tags: Optional[List[str]] = None
) -> Any:
    """
    Initialize Weights & Biases run with improved configuration handling.

    Parameters
    ----------
    project_name : str
        Project name
    experiment_name : Optional[str], optional
        Experiment name, by default None
    config : Optional[Union[Dict[str, Any], DictConfig]], optional
        Configuration dictionary, by default None
    tags : Optional[List[str]], optional
        Tags for the run, by default None

    Returns
    -------
    Any
        Weights & Biases run object or None if initialization fails
    """
    try:
        # Convert OmegaConf DictConfig to dict if needed
        if config is not None and isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        # Generate a unique experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        # Initialize run
        wandb_run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config_dict,
            tags=tags
        )

        logger.info(f"Initialized W&B run: {experiment_name}")
        return wandb_run
    except Exception as e:
        logger.error(f"Error initializing W&B: {e}")
        logger.warning("Continuing without W&B tracking")
        return None


def save_config(
        config: Union[Dict[str, Any], DictConfig],
        output_file: str = 'config.json',
        timestamp: bool = True
) -> None:
    """
    Save configuration to JSON file with optional timestamp.

    Parameters
    ----------
    config : Union[Dict[str, Any], DictConfig]
        Configuration dictionary or OmegaConf DictConfig
    output_file : str, optional
        Output file path, by default 'config.json'
    timestamp : bool, optional
        Whether to add timestamp to filename, by default True
    """
    # Add timestamp to filename if requested
    if timestamp:
        dirname = os.path.dirname(output_file)
        basename = os.path.basename(output_file)
        name, ext = os.path.splitext(basename)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_basename = f"{name}_{timestamp_str}{ext}"

        output_file = os.path.join(dirname, new_basename)

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert OmegaConf DictConfig to dict if needed
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config

    # Convert numpy types to Python native types
    def convert_np_to_python(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_np_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np_to_python(item) for item in obj]
        else:
            return obj

    config_json = convert_np_to_python(config_dict)

    # Save configuration to JSON file
    with open(output_file, 'w') as f:
        json.dump(config_json, f, indent=4)

    logger.info(f"Configuration saved to {output_file}")


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

    logger.info(f"Configuration loaded from {input_file}")
    return config

