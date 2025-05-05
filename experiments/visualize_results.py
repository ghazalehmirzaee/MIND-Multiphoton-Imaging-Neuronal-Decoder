"""Script to visualize results from model comparison."""
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import wandb
from typing import Dict, Any, List, Optional, Tuple, Callable

from mind.data.loader import load_processed_data
from mind.utils.logging import setup_logging
from mind.utils.experiment_tracking import init_wandb
from mind.visualization.performance import (
    plot_performance_comparison,
    plot_signal_type_comparison,
    plot_model_comparison,
    plot_binary_confusion_matrices,  # Changed from plot_confusion_matrices
    plot_binary_roc_curves,  # Changed from plot_roc_curves
    create_comparative_performance_grid,
    plot_performance_radar,
    plot_cross_signal_comparison
)
from mind.visualization.feature_importance import (
    analyze_feature_importance,
    plot_comparative_feature_importance
)
from mind.visualization.signal_visualization import create_signal_visualizations
from mind.evaluation.metrics import generate_metrics_report  # Added for metrics report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize results from model comparison.")

    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to processed data file')
    parser.add_argument('--classical_results', type=str, required=True,
                        help='Path to classical ML results JSON file')
    parser.add_argument('--deep_results', type=str, required=True,
                        help='Path to deep learning results JSON file')

    # W&B parameters
    parser.add_argument('--project_name', type=str, default='MIND',
                        help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the W&B experiment')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')

    return parser.parse_args()


def load_results(file_path):
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_file=os.path.join(args.output_dir, 'logs', 'visualize_results.log'))
    logger = logging.getLogger(__name__)
    logger.info("Starting results visualization")

    # Initialize W&B
    wandb_run = init_wandb(
        project_name=args.project_name,
        experiment_name=args.experiment_name or 'results_visualization'
    )

    # Set plotting style
    plt.style.use('seaborn-whitegrid')

    # Load data
    logger.info(f"Loading data from {args.data_file}")
    data = load_processed_data(args.data_file)

    # Load results
    logger.info(f"Loading classical ML results from {args.classical_results}")
    classical_results = load_results(args.classical_results)

    logger.info(f"Loading deep learning results from {args.deep_results}")
    deep_results = load_results(args.deep_results)

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)

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

        # Plot binary confusion matrices
        cm_figure = plot_binary_confusion_matrices(
            all_results,
            output_dir=os.path.join(args.output_dir, 'figures')
        )
        if wandb_run:
            wandb_run.log({"binary_confusion_matrices": wandb.Image(cm_figure)})

        # Plot binary ROC curves
        roc_figures = plot_binary_roc_curves(
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

        # Generate metrics report
        metrics_report = generate_metrics_report(
            all_results,
            output_file=os.path.join(args.output_dir, 'metrics', 'metrics_report.json')
        )
        logger.info(
            f"Generated metrics report saved to {os.path.join(args.output_dir, 'metrics', 'metrics_report.json')}")

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

    logger.info("Results visualization completed successfully.")


if __name__ == "__main__":
    main()

    