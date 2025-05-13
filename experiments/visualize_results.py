# experiments/visualize_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import json
import logging

# Import custom modules
from mind.data.loader import load_dataset
from mind.visualization.signal_visualization import (
    visualize_signals, visualize_signal_comparison, visualize_signals_heatmap
)


@hydra.main(config_path="../mind/config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Script to visualize results from the model comparison experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Convert relative paths to absolute paths
    neural_path = to_absolute_path(cfg.data.neural_path)
    behavior_path = to_absolute_path(cfg.data.behavior_path)
    output_dir = to_absolute_path(cfg.output.dir)
    viz_dir = to_absolute_path(cfg.output.viz_dir)
    os.makedirs(viz_dir, exist_ok=True)

    # Load results
    results_path = os.path.join(output_dir, "performance_metrics.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded results from {results_path}")
    else:
        logger.error(f"Results file not found: {results_path}")
        results = {}

    # Create visualizations
    if results:
        # Create bar plots for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        signal_types = list(results.keys())
        model_names = list(results[signal_types[0]].keys())

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Prepare data for plotting
            metric_data = []
            for signal_type in signal_types:
                for model_name in model_names:
                    metric_data.append({
                        'Signal Type': signal_type,
                        'Model': model_name,
                        metric.capitalize(): results[signal_type][model_name][metric]
                    })
            df = pd.DataFrame(metric_data)

            # Plot grouped bar chart
            sns.barplot(x='Model', y=metric.capitalize(), hue='Signal Type', data=df, ax=ax)

            # Set labels and title
            ax.set_title(f'Model {metric.capitalize()} Comparison', fontsize=16)
            ax.set_xlabel('Model', fontsize=14)
            ax.set_ylabel(metric.capitalize(), fontsize=14)

            # Adjust legend
            ax.legend(title='Signal Type', fontsize=12)

            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"model_{metric}_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # Create heatmap of all metrics
        for signal_type in signal_types:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Prepare data for heatmap
            heatmap_data = []
            for model_name in model_names:
                model_metrics = results[signal_type][model_name]
                heatmap_data.append([
                    model_metrics[metric] for metric in metrics
                ])

            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                        xticklabels=[m.capitalize() for m in metrics],
                        yticklabels=model_names, ax=ax)

            # Set title
            ax.set_title(f'Performance Metrics Heatmap - {signal_type}', fontsize=16)

            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"metrics_heatmap_{signal_type}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # Create additional signal visualizations
    logger.info("Creating signal visualizations...")

    # Load data
    data = load_dataset(neural_path, behavior_path, cfg.data.binary_task)

    # Visualize signals
    visualize_signals(data,
                      num_neurons=cfg.visualization.signals.num_neurons,
                      output_dir=viz_dir,
                      save_filename="raw_signals_detailed.png",
                      figsize=(25, 20))

    # Visualize signal comparison with more neurons
    visualize_signal_comparison(data,
                                num_neurons=10,
                                output_dir=viz_dir,
                                save_filename="signal_comparison_detailed.png",
                                figsize=(25, 20))

    # Visualize heatmap with more neurons
    visualize_signals_heatmap(data,
                              num_neurons=250,
                              output_dir=viz_dir,
                              save_filename="signals_heatmap_detailed.png",
                              figsize=(25, 20))

    logger.info("Visualization complete.")


if __name__ == "__main__":
    main()

