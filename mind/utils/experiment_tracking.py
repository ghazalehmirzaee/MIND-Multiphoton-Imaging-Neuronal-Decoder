"""
Experiment tracking with Weights & Biases.
"""
import os
import wandb
from typing import Dict, Optional, Any, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def init_wandb(project_name: str = "mind-calcium-imaging",
               entity: Optional[str] = None,
               api_key: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None,
               log_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Initialize Weights & Biases for experiment tracking.
    """
    try:
        # Check if API key is provided or in environment
        if api_key is not None:
            os.environ["WANDB_API_KEY"] = api_key

        # Check if log directory is provided
        if log_dir is not None:
            os.environ["WANDB_DIR"] = str(log_dir)

        # Initialize W&B
        wandb.init(
            project=project_name,
            entity=entity,
            config=config
        )

        logger.info(f"W&B initialized (project: {project_name}, entity: {entity})")
        return True

    except Exception as e:
        logger.error(f"Error initializing W&B: {e}")
        return False


def log_artifact(artifact_name: str,
                 artifact_type: str,
                 file_path: Union[str, Path],
                 description: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Log an artifact to W&B.
    """
    try:
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata
        )

        # Add file to artifact
        artifact.add_file(str(file_path))

        # Log artifact
        wandb.log_artifact(artifact)

        logger.info(f"Artifact {artifact_name} logged to W&B")
        return True

    except Exception as e:
        logger.error(f"Error logging artifact to W&B: {e}")
        return False


def log_results(results: Dict[str, Any],
                prefix: Optional[str] = None,
                step: Optional[int] = None) -> bool:
    """
    Log results to W&B.
    """
    try:
        # Apply prefix if provided
        if prefix is not None:
            results = {f"{prefix}/{k}": v for k, v in results.items()}

        # Log results
        wandb.log(results, step=step)

        logger.info(f"Results logged to W&B")
        return True

    except Exception as e:
        logger.error(f"Error logging results to W&B: {e}")
        return False


def log_model(model, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Log a model to W&B.
    """
    try:
        # Log model
        wandb.log({f"models/{model_name}": model})

        # Log metadata if provided
        if metadata is not None:
            wandb.log({f"model_metadata/{model_name}": metadata})

        logger.info(f"Model {model_name} logged to W&B")
        return True

    except Exception as e:
        logger.error(f"Error logging model to W&B: {e}")
        return False


def finish_wandb() -> bool:
    """
    Finish the current W&B run.
    """
    try:
        wandb.finish()
        logger.info("W&B run finished")
        return True

    except Exception as e:
        logger.error(f"Error finishing W&B run: {e}")
        return False

