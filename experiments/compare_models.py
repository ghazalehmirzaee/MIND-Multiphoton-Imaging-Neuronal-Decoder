"""Script to compare all models on all signal types with Hydra configuration."""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from typing import Dict, Any, Optional
from datetime import datetime
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from mind.data.loader import load_processed_data
from mind.training.trainer_classical import (
    train_all_classical_models,
    test_classical_models,
    save_classical_models,
    save_results as save_classical_results
)
from mind.training.trainer_deep import (
    train_all_deep_models,
    test_deep_models,
    save_deep_models,
    save_results as save_deep_results
)
from mind.utils.logging import setup_logging
from mind.utils.experiment_tracking import init_wandb, save_config
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


@hydra.main(config_path="../mind/config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main function to compare models on different signal types with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Access the working directory (set by Hydra)
    work_dir = os.getcwd()

    # Setup logging
    log_file = os.path.join(work_dir, 'logs', f'compare_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting model comparison with Hydra configuration")

    # Save configuration to JSON
    config_dir = os.path.join(work_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    save_config(cfg, os.path.join(config_dir, 'model_comparison.json'))

    # Initialize W&B
    wandb_run = init_wandb(
        project_name=cfg.wandb.project,
        experiment_name=cfg.experiment.name,
        config=cfg
    )

    # Set plotting style
    plt.style.use(cfg.visualization.style)

    # Load data
    logger.info(f"Loading data from {cfg.data.matlab_file}")
    data = load_processed_data(cfg.data.matlab_file)

    # Create output directories
    os.makedirs(os.path.join(work_dir, 'models', 'classical'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'models', 'deep'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'figures'), exist_ok=True)

    # Train and evaluate classical ML models
    classical_results = None
    if 'classical_types' in cfg.models and cfg.models.classical_types:
        logger.info("Training classical ML models")
        classical_results = train_all_classical_models(data, cfg, wandb_run)

        logger.info("Testing classical ML models")
        classical_test_results = test_classical_models(
            classical_results['models'], data, wandb_run
        )
        classical_results['test_results'] = classical_test_results

        logger.info("Saving classical ML models and results")
        save_classical_models(
            classical_results['models'],
            os.path.join(work_dir, 'models', 'classical')
        )
        save_classical_results(
            classical_results,
            os.path.join(work_dir, 'metrics', 'classical_ml_results.json')
        )

    # Train and evaluate deep learning models
    deep_results = None
    if 'deep_types' in cfg.models and cfg.models.deep_types:
        logger.info("Training deep learning models")
        deep_results = train_all_deep_models(data, cfg, wandb_run)

        logger.info("Testing deep learning models")
        deep_test_results = test_deep_models(
            deep_results['models'], data, cfg, wandb_run
        )
        deep_results['test_results'] = deep_test_results

        logger.info("Saving deep learning models and results")
        save_deep_models(
            deep_results['models'],
            os.path.join(work_dir, 'models', 'deep')
        )
        save_deep_results(
            deep_results,
            os.path.join(work_dir, 'metrics', 'deep_learning_results.json')
        )

    # Visualize results
    if not cfg.get('skip_visualization', False):
        logger.info("Visualizing results")

        # Combine results
        all_results = {}

        if classical_results and 'test_results' in classical_results:
            all_results.update(classical_results['test_results'])

        if deep_results and 'test_results' in deep_results:
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
                output_file=os.path.join(work_dir, 'figures', 'performance_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"performance_comparison": wandb.Image(fig)})

            # Plot signal type comparison
            fig = plot_signal_type_comparison(
                performance_df,
                output_file=os.path.join(work_dir, 'figures', 'signal_type_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"signal_type_comparison": wandb.Image(fig)})

            # Plot model comparison
            fig = plot_model_comparison(
                performance_df,
                output_file=os.path.join(work_dir, 'figures', 'model_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"model_comparison": wandb.Image(fig)})

            # Plot binary confusion matrices
            cm_figure = plot_binary_confusion_matrices(
                all_results,
                output_dir=os.path.join(work_dir, 'figures')
            )
            if wandb_run:
                wandb_run.log({"binary_confusion_matrices": wandb.Image(cm_figure)})

            # Plot binary ROC curves
            roc_figures = plot_binary_roc_curves(
                all_results,
                output_dir=os.path.join(work_dir, 'figures')
            )
            if wandb_run:
                for name, fig in roc_figures.items():
                    wandb_run.log({name: wandb.Image(fig)})

            # Create comparative performance grid
            fig = create_comparative_performance_grid(
                all_results,
                output_file=os.path.join(work_dir, 'figures', 'comparative_performance_grid.png')
            )
            if wandb_run:
                wandb_run.log({"comparative_performance_grid": wandb.Image(fig)})

            # Plot performance radar
            fig = plot_performance_radar(
                all_results,
                output_file=os.path.join(work_dir, 'figures', 'performance_radar.png')
            )
            if wandb_run:
                wandb_run.log({"performance_radar": wandb.Image(fig)})

            # Plot cross-signal comparison
            fig = plot_cross_signal_comparison(
                all_results,
                output_file=os.path.join(work_dir, 'figures', 'cross_signal_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"cross_signal_comparison": wandb.Image(fig)})

            # Generate metrics report
            metrics_report = generate_metrics_report(
                all_results,
                output_file=os.path.join(work_dir, 'metrics', 'metrics_report.json')
            )
            logger.info(
                f"Generated metrics report saved to {os.path.join(work_dir, 'metrics', 'metrics_report.json')}")

        # Feature importance visualizations
        if classical_results and 'feature_importance' in classical_results:
            # Analyze feature importance
            fi_analysis = analyze_feature_importance(
                classical_results['feature_importance'],
                data['window_size'],
                {
                    'calcium': data['n_calcium_neurons'],
                    'deltaf': data['n_deltaf_neurons'],
                    'deconv': data['n_deconv_neurons']
                },
                output_dir=os.path.join(work_dir, 'figures')
            )

            # Log figures to W&B
            if wandb_run:
                for name, fig in fi_analysis['figures'].items():
                    wandb_run.log({name: wandb.Image(fig)})

            # Create comparative feature importance visualizations
            fi_figures = plot_comparative_feature_importance(
                classical_results['feature_importance'],
                output_dir=os.path.join(work_dir, 'figures')
            )

            # Log figures to W&B
            if wandb_run:
                for name, fig in fi_figures.items():
                    wandb_run.log({name: wandb.Image(fig)})

        # Signal visualizations
        signal_figures = create_signal_visualizations(
            data,
            output_dir=os.path.join(work_dir, 'figures')
        )

        # Log figures to W&B
        if wandb_run:
            for name, fig in signal_figures.items():
                wandb_run.log({name: wandb.Image(fig)})

    # Finish W&B run
    if wandb_run:
        wandb_run.finish()

    logger.info("Model comparison completed successfully.")


if __name__ == "__main__":
    main()

