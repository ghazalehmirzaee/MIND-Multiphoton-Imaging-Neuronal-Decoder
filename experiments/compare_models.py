"""Script to compare all models on all signal types."""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import torch
import wandb
from typing import Dict, Any

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
    plot_confusion_matrices,
    plot_roc_curves,
    create_comparative_performance_grid,
    plot_performance_radar,
    plot_cross_signal_comparison
)
from mind.visualization.feature_importance import (
    analyze_feature_importance,
    plot_comparative_feature_importance
)
from mind.visualization.signal_visualization import create_signal_visualizations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare models on calcium imaging data.")

    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to processed data file')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for deep learning models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for deep learning models')

    # Execution flags
    parser.add_argument('--skip_classical', action='store_true',
                        help='Skip classical ML models')
    parser.add_argument('--skip_deep_learning', action='store_true',
                        help='Skip deep learning models')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip results visualization')

    # W&B parameters
    parser.add_argument('--project_name', type=str, default='MIND',
                        help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the W&B experiment')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')

    return parser.parse_args()


def create_config(args):
    """Create configuration dictionary."""
    config = {
        'experiment': {
            'name': args.experiment_name or 'model_comparison',
            'seed': 42,
            'output_dir': args.output_dir
        },
        'data': {
            'file': args.data_file
        },
        'training': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping': {
                'patience': 15,
                'min_delta': 0.001
            }
        },
        'models': {
            'classical': {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 30,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced'
                },
                'svm': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'class_weight': 'balanced',
                    'probability': True,
                    'pca': True,
                    'pca_components': 0.95
                },
                'mlp': {
                    'hidden_layer_sizes': [128, 64],
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive',
                    'early_stopping': True
                }
            },
            'deep': {
                'fcnn': {
                    'hidden_sizes': [256, 128, 64],
                    'dropout_rates': [0.4, 0.4, 0.3],
                    'batch_norm': True
                },
                'cnn': {
                    'channels': [64, 128, 256],
                    'kernel_size': 3,
                    'dropout_rate': 0.5,
                    'batch_norm': True
                }
            }
        },
        'visualization': {
            'style': 'seaborn-whitegrid',
            'figsize': [12, 8],
            'dpi': 300,
            'save_format': 'png',
            'colors': {
                'calcium': "#1f77b4",
                'deltaf': "#ff7f0e",
                'deconv': "#2ca02c"
            }
        },
        'wandb': {
            'project': args.project_name,
            'entity': 'your_username',
            'log_artifacts': True
        }
    }

    return config


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_file=os.path.join(args.output_dir, 'logs', 'compare_models.log'))
    logger = logging.getLogger(__name__)
    logger.info("Starting model comparison")

    # Create configuration
    config = create_config(args)

    # Save configuration
    os.makedirs(os.path.join(args.output_dir, 'config'), exist_ok=True)
    save_config(config, os.path.join(args.output_dir, 'config', 'model_comparison.json'))

    # Initialize W&B
    wandb_run = init_wandb(
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        config=config
    )

    # Set plotting style
    # plt.style.use(config['visualization']['style'])
    plt.style.use('ggplot')

    # Load data
    logger.info(f"Loading data from {args.data_file}")
    data = load_processed_data(args.data_file)

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'models', 'classical'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models', 'deep'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    # Train and evaluate classical ML models
    classical_results = None
    if not args.skip_classical:
        logger.info("Training classical ML models")
        classical_results = train_all_classical_models(data, config, wandb_run)

        logger.info("Testing classical ML models")
        classical_test_results = test_classical_models(
            classical_results['models'], data, wandb_run
        )
        classical_results['test_results'] = classical_test_results

        logger.info("Saving classical ML models and results")
        save_classical_models(
            classical_results['models'],
            os.path.join(args.output_dir, 'models', 'classical')
        )
        save_classical_results(
            classical_results,
            os.path.join(args.output_dir, 'metrics', 'classical_ml_results.json')
        )

    # Train and evaluate deep learning models
    deep_results = None
    if not args.skip_deep_learning:
        logger.info("Training deep learning models")
        deep_results = train_all_deep_models(data, config, wandb_run)

        logger.info("Testing deep learning models")
        deep_test_results = test_deep_models(
            deep_results['models'], data, config, wandb_run
        )
        deep_results['test_results'] = deep_test_results

        logger.info("Saving deep learning models and results")
        save_deep_models(
            deep_results['models'],
            os.path.join(args.output_dir, 'models', 'deep')
        )
        save_deep_results(
            deep_results,
            os.path.join(args.output_dir, 'metrics', 'deep_learning_results.json')
        )

    # Visualize results
    if not args.skip_visualization:
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
                output_file=os.path.join(args.output_dir, 'figures', 'performance_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"performance_comparison": wandb.Image(fig)})

            # Plot signal type comparison
            fig = plot_signal_type_comparison(
                performance_df,
                output_file=os.path.join(args.output_dir, 'figures', 'signal_type_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"signal_type_comparison": wandb.Image(fig)})

            # Plot model comparison
            fig = plot_model_comparison(
                performance_df,
                output_file=os.path.join(args.output_dir, 'figures', 'model_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"model_comparison": wandb.Image(fig)})

            # Plot confusion matrices
            cm_figures = plot_confusion_matrices(
                all_results,
                output_dir=os.path.join(args.output_dir, 'figures')
            )
            if wandb_run:
                for name, fig in cm_figures.items():
                    wandb_run.log({name: wandb.Image(fig)})

            # Plot ROC curves
            roc_figures = plot_roc_curves(
                all_results,
                output_dir=os.path.join(args.output_dir, 'figures')
            )
            if wandb_run:
                for name, fig in roc_figures.items():
                    wandb_run.log({name: wandb.Image(fig)})

            # Create comparative performance grid
            fig = create_comparative_performance_grid(
                all_results,
                output_file=os.path.join(args.output_dir, 'figures', 'comparative_performance_grid.png')
            )
            if wandb_run:
                wandb_run.log({"comparative_performance_grid": wandb.Image(fig)})

            # Plot performance radar
            fig = plot_performance_radar(
                all_results,
                output_file=os.path.join(args.output_dir, 'figures', 'performance_radar.png')
            )
            if wandb_run:
                wandb_run.log({"performance_radar": wandb.Image(fig)})

            # Plot cross-signal comparison
            fig = plot_cross_signal_comparison(
                all_results,
                output_file=os.path.join(args.output_dir, 'figures', 'cross_signal_comparison.png')
            )
            if wandb_run:
                wandb_run.log({"cross_signal_comparison": wandb.Image(fig)})

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
                output_dir=os.path.join(args.output_dir, 'figures')
            )

            # Log figures to W&B
            if wandb_run:
                for name, fig in fi_analysis['figures'].items():
                    wandb_run.log({name: wandb.Image(fig)})

            # Create comparative feature importance visualizations
            fi_figures = plot_comparative_feature_importance(
                classical_results['feature_importance'],
                output_dir=os.path.join(args.output_dir, 'figures')
            )

            # Log figures to W&B
            if wandb_run:
                for name, fig in fi_figures.items():
                    wandb_run.log({name: wandb.Image(fig)})

        # Signal visualizations
        signal_figures = create_signal_visualizations(
            data,
            output_dir=os.path.join(args.output_dir, 'figures')
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

