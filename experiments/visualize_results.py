"""Script to visualize results from model comparison with Hydra configuration."""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Dict, Any, List, Optional
from datetime import datetime
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from mind.data.loader import load_processed_data
from mind.utils.logging import setup_logging
from mind.utils.experiment_tracking import init_wandb
from mind.visualization.performance import (
    plot_performance_comparison,
    plot_signal_type_comparison,
    plot_model_comparison,
    plot_binary_confusion_matrices,
    plot_binary_roc_curves,
    create_comparative_performance_grid,
    plot_performance_radar,
    plot_cross_signal_comparison
)
from mind.visualization.feature_importance import (
    analyze_feature_importance,
    plot_comparative_feature_importance
)
from mind.visualization.signal_visualization import create_signal_visualizations
from mind.evaluation.metrics import generate_metrics_report


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Parameters
    ----------
    file_path : str
        Path to the results JSON file

    Returns
    -------
    Dict[str, Any]
        Loaded results dictionary
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


@hydra.main(config_path="../mind/config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function to visualize model comparison results with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Access the working directory (set by Hydra)
    work_dir = os.getcwd()

    # Setup logging
    log_file = os.path.join(work_dir, 'logs', f'visualize_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting results visualization with Hydra configuration")

    # Initialize W&B
    wandb_run = init_wandb(
        project_name=cfg.wandb.project,
        experiment_name=cfg.experiment.name or 'results_visualization',
        config=cfg
    )

    # Set plotting style
    plt.style.use(cfg.visualization.style)

    # Load data
    logger.info(f"Loading data from {cfg.data.matlab_file}")
    data = load_processed_data(cfg.data.matlab_file)

    # Load results
    classical_results_path = os.path.join(cfg.experiment.output_dir, 'metrics', 'classical_ml_results.json')
    deep_results_path = os.path.join(cfg.experiment.output_dir, 'metrics', 'deep_learning_results.json')

    logger.info(f"Loading classical ML results from {classical_results_path}")
    classical_results = load_results(classical_results_path)

    logger.info(f"Loading deep learning results from {deep_results_path}")
    deep_results = load_results(deep_results_path)

    # Create output directories
    figures_dir = os.path.join(work_dir, 'figures')
    metrics_dir = os.path.join(work_dir, 'metrics')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Combine results
    all_results = {}

    if 'test_results' in classical_results:
        all_results.update(classical_results['test_results'])

    if 'test_results' in deep_results:
        for signal_type, signal_results in deep_results['test_results'].items():
            if signal_type not in all_results:
                all_results[signal_type] = {}

            all_results[signal_type].update(signal_results)

    # Create performance visualizations
    if all_results:
        # Create DataFrame for performance comparison
        rows = []
        for signal_type, signal_results in all_results.items():
            for model_type, metrics in signal_results.items():
                row = {
                    'Signal Type': signal_type,
                    'Model': model_type,
                    'Accuracy': metrics['accuracy'],
                    'Precision (Macro)': metrics['precision_macro'],
                    'Recall (Macro)': metrics['recall_macro'],
                    'F1 (Macro)': metrics['f1_macro']
                }
                rows.append(row)

        performance_df = pd.DataFrame(rows)

        # Plot performance comparison
        fig = plot_performance_comparison(
            performance_df,
            output_file=os.path.join(figures_dir, 'performance_comparison.png')
        )
        if wandb_run:
            wandb_run.log({"performance_comparison": wandb.Image(fig)})

        # Plot signal type comparison
        fig = plot_signal_type_comparison(
            performance_df,
            output_file=os.path.join(figures_dir, 'signal_type_comparison.png')
        )
        if wandb_run:
            wandb_run.log({"signal_type_comparison": wandb.Image(fig)})

        # Plot model comparison
        fig = plot_model_comparison(
            performance_df,
            output_file=os.path.join(figures_dir, 'model_comparison.png')
        )
        if wandb_run:
            wandb_run.log({"model_comparison": wandb.Image(fig)})

        # Plot binary confusion matrices
        cm_figure = plot_binary_confusion_matrices(
            all_results,
            output_dir=figures_dir
        )
        if wandb_run:
            wandb_run.log({"binary_confusion_matrices": wandb.Image(cm_figure)})

        # Plot binary ROC curves
        roc_figures = plot_binary_roc_curves(
            all_results,
            output_dir=figures_dir
        )
        if wandb_run:
            for name, fig in roc_figures.items():
                wandb_run.log({name: wandb.Image(fig)})

        # Create comparative performance grid
        fig = create_comparative_performance_grid(
            all_results,
            output_file=os.path.join(figures_dir, 'comparative_performance_grid.png')
        )
        if wandb_run:
            wandb_run.log({"comparative_performance_grid": wandb.Image(fig)})

        # Plot performance radar
        fig = plot_performance_radar(
            all_results,
            output_file=os.path.join(figures_dir, 'performance_radar.png')
        )
        if wandb_run:
            wandb_run.log({"performance_radar": wandb.Image(fig)})

        # Plot cross-signal comparison
        fig = plot_cross_signal_comparison(
            all_results,
            output_file=os.path.join(figures_dir, 'cross_signal_comparison.png')
        )
        if wandb_run:
            wandb_run.log({"cross_signal_comparison": wandb.Image(fig)})

        # Generate metrics report
        metrics_report = generate_metrics_report(
            all_results,
            output_file=os.path.join(metrics_dir, 'metrics_report.json')
        )
        logger.info(f"Generated metrics report saved to {os.path.join(metrics_dir, 'metrics_report.json')}")

    # Feature importance visualizations
    if 'feature_importance' in classical_results:
        # Analyze feature importance
        fi_analysis = analyze_feature_importance(
            classical_results['feature_importance'],
            data['window_size'],
            {
                'calcium': data['n_calcium_neurons'],
                'deltaf': data['n_deltaf_neurons'],
                'deconv': data['n_deconv_neurons']
            },
            output_dir=figures_dir
        )

        # Log figures to W&B
        if wandb_run:
            for name, fig in fi_analysis['figures'].items():
                wandb_run.log({name: wandb.Image(fig)})

        # Create comparative feature importance visualizations
        fi_figures = plot_comparative_feature_importance(
            classical_results['feature_importance'],
            output_dir=figures_dir
        )

        # Log figures to W&B
        if wandb_run:
            for name, fig in fi_figures.items():
                wandb_run.log({name: wandb.Image(fig)})

    # Signal visualizations
    signal_figures = create_signal_visualizations(
        data,
        output_dir=figures_dir
    )

    # Log figures to W&B
    if wandb_run:
        for name, fig in signal_figures.items():
            wandb_run.log({name: wandb.Image(fig)})

    # Finish W&B run
    if wandb_run:
        wandb_run.finish()

    logger.info("Results visualization completed successfully.")


if __name__ == "__main__":
    main()

