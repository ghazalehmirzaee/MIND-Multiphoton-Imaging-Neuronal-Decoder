import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import time
import logging
import json
import gc
import torch
from datetime import datetime

# Import custom modules
from mind.data.loader import load_dataset
from mind.data.processor import create_sliding_windows, split_data
from mind.training.train import train_multiple_models, evaluate_multiple_models, clean_memory
from mind.visualization.signal_visualization import (
    visualize_signals, visualize_signal_comparison, visualize_signals_heatmap
)
from mind.visualization.performance import (
    plot_confusion_matrix, plot_model_performance_comparison,
    plot_roc_curves, plot_precision_recall_curves, plot_all_confusion_matrices,
    plot_radar_chart
)
from mind.visualization.feature_importance import (
    plot_feature_importance_heatmap, plot_temporal_feature_importance,
    plot_neuron_feature_importance, plot_all_feature_importances
)
from mind.utils.experiment_tracking import (
    init_wandb, log_metrics, log_model, log_figure,
    log_confusion_matrix, save_results_json, finish_run
)


@hydra.main(config_path="../mind/config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main experiment script for comparing different models on calcium signal data.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize W&B
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    init_wandb(project_name="MIND",
               experiment_name=f"compare_models_{timestamp}",
               config=OmegaConf.to_container(cfg, resolve=True))

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Convert relative paths to absolute paths
    neural_path = to_absolute_path(cfg.data.neural_path)
    behavior_path = to_absolute_path(cfg.data.behavior_path)

    # Create output directories
    output_dir = to_absolute_path(cfg.output.dir)
    model_dir = to_absolute_path(cfg.output.model_dir)
    viz_dir = to_absolute_path(cfg.output.viz_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Record start time
    start_time = time.time()

    # Step 1: Load and process data
    logger.info("Step 1: Loading and processing data...")
    data = load_dataset(neural_path, behavior_path, cfg.data.binary_task)

    # Log data information to W&B
    unique, counts = np.unique(data['labels'], return_counts=True)
    label_dist = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
    log_metrics({"data/label_distribution": label_dist})

    # Step 2: Visualize raw signals
    logger.info("Step 2: Visualizing raw signals...")
    signal_fig = visualize_signals(data,
                                   num_neurons=cfg.visualization.signals.num_neurons,
                                   output_dir=viz_dir,
                                   save_filename="raw_signals.png",
                                   figsize=cfg.visualization.signals.figsize)
    log_figure("raw_signals", signal_fig)

    comparison_fig = visualize_signal_comparison(data,
                                                 num_neurons=5,
                                                 output_dir=viz_dir,
                                                 save_filename="signal_comparison.png")
    log_figure("signal_comparison", comparison_fig)

    heatmap_fig = visualize_signals_heatmap(data,
                                            num_neurons=250,
                                            output_dir=viz_dir,
                                            save_filename="signals_heatmap.png")
    log_figure("signals_heatmap", heatmap_fig)

    # Step 3: Create sliding windows
    logger.info("Step 3: Creating sliding windows...")
    windowed_data = create_sliding_windows(data,
                                           window_size=cfg.data.window_size,
                                           step_size=cfg.data.step_size)

    # Log windowed data information
    windowed_label_dist = {}
    unique, counts = np.unique(windowed_data['labels'], return_counts=True)
    for u, c in zip(unique, counts):
        windowed_label_dist[int(u)] = int(c)
    log_metrics({"data/windowed_label_distribution": windowed_label_dist})

    # Step 4: Split data into train, validation, and test sets
    logger.info("Step 4: Splitting data...")
    splits = split_data(windowed_data,
                        test_size=cfg.data.test_size,
                        val_size=cfg.data.val_size,
                        random_state=cfg.data.random_state)

    # Log split information
    for split_name, split_data in splits.items():
        if 'labels' in split_data:
            unique, counts = np.unique(split_data['labels'], return_counts=True)
            split_label_dist = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
            log_metrics({f"data/{split_name}_label_distribution": split_label_dist})

    # Step 5: Train and evaluate models for each signal type
    logger.info("Step 5: Training and evaluating models...")

    # Define signal types to process
    signal_types = ['calcium_signal', 'deltaf', 'deconv']

    # Define model configurations
    model_configs = [
        {'name': 'random_forest', 'params': dict(cfg.models.random_forest)},
        {'name': 'svm', 'params': dict(cfg.models.svm)},
        {'name': 'mlp', 'params': dict(cfg.models.mlp)},
        {'name': 'fcnn', 'params': dict(cfg.models.fcnn)},
        {'name': 'cnn', 'params': dict(cfg.models.cnn)}
    ]

    # Initialize results dictionaries
    all_results = {}

    # Train and evaluate models for each signal type
    for signal_type in signal_types:
        if signal_type not in splits['train'] or splits['train'][signal_type] is None:
            logger.warning(f"Skipping {signal_type} as it is not available in the data.")
            continue

        logger.info(f"Processing {signal_type}...")
        all_results[signal_type] = {}

        # Create a W&B run group for this signal type
        log_metrics({f"current_signal": signal_type})

        # Extract data for this signal type
        X_train = splits['train'][signal_type]
        y_train = splits['train']['labels']
        X_val = splits['val'][signal_type]
        y_val = splits['val']['labels']
        X_test = splits['test'][signal_type]
        y_test = splits['test']['labels']

        # Create signal-specific output directories
        signal_model_dir = os.path.join(model_dir, signal_type)
        signal_viz_dir = os.path.join(viz_dir, signal_type)
        os.makedirs(signal_model_dir, exist_ok=True)
        os.makedirs(signal_viz_dir, exist_ok=True)

        # Train models
        training_results = train_multiple_models(
            model_configs, X_train, y_train, X_val, y_val, signal_model_dir
        )

        # Extract trained models
        trained_models = {}
        for name, results in training_results.items():
            if 'model' in results:
                trained_models[name] = results['model']

        # Evaluate on test set
        evaluation_results = evaluate_multiple_models(
            trained_models, X_test, y_test, "Test"
        )

        # Collect feature importances
        feature_importances = {}
        for model_name, model in trained_models.items():
            if hasattr(model, 'get_feature_importances'):
                try:
                    feature_importances[model_name] = model.get_feature_importances()
                except Exception as e:
                    print(f"Could not get feature importances for {model_name}: {e}")

        # Log results to W&B and save to JSON
        for model_name in trained_models.keys():
            # Create run name
            run_name = f"{signal_type}_{model_name}"

            # Log training metrics
            if model_name in training_results:
                metrics = training_results[model_name]['metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float, np.number)):
                        log_metrics({f"{run_name}/train/{metric_name}": float(metric_value)})
                    elif isinstance(metric_value, (list, np.ndarray)) and len(metric_value) > 0:
                        # For lists/arrays, log the final value
                        log_metrics({f"{run_name}/train/{metric_name}_final": float(metric_value[-1])})

                # Save training metrics as JSON
                save_results_json(
                    metrics,
                    os.path.join(signal_model_dir, model_name),
                    "training_metrics.json"
                )

            # Log evaluation metrics
            if model_name in evaluation_results:
                metrics = evaluation_results[model_name]
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ['confusion_matrix', 'y_true', 'y_pred', 'y_pred_proba'] and isinstance(
                            metric_value, (int, float, np.number)):
                        log_metrics({f"{run_name}/test/{metric_name}": float(metric_value)})

                # Log confusion matrix if available
                if 'y_true' in metrics and 'y_pred' in metrics:
                    log_confusion_matrix(
                        metrics['y_true'],
                        metrics['y_pred'],
                        ["No Footstep", "Contralateral Footstep"]
                    )

                # Save evaluation metrics as JSON
                save_results_json(
                    {k: v for k, v in metrics.items() if
                     k not in ['confusion_matrix', 'y_true', 'y_pred', 'y_pred_proba']},
                    os.path.join(signal_model_dir, model_name),
                    "evaluation_metrics.json"
                )

        # Store results for later visualization
        all_results[signal_type] = {
            'training': training_results,
            'evaluation': evaluation_results,
            'feature_importances': feature_importances
        }

        # Clean up memory between signal types
        clean_memory()

    # Step 6: Create comparison visualizations
    logger.info("Step 6: Creating comparison visualizations...")

    # Prepare data for visualization
    performance_comparison = {}
    for signal_type, results in all_results.items():
        performance_comparison[signal_type] = {}
        for model_name, metrics in results['evaluation'].items():
            if all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
                performance_comparison[signal_type][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                }

    # Plot performance comparison across signal types
    if performance_comparison:
        # Accuracy comparison
        acc_fig = plot_model_performance_comparison(
            performance_comparison, metric='accuracy',
            title='Model Accuracy Comparison',
            output_dir=viz_dir,
            save_filename="model_accuracy_comparison.png"
        )
        log_figure("model_accuracy_comparison", acc_fig)

        # F1 score comparison
        f1_fig = plot_model_performance_comparison(
            performance_comparison, metric='f1_score',
            title='Model F1 Score Comparison',
            output_dir=viz_dir,
            save_filename="model_f1_comparison.png"
        )
        log_figure("model_f1_comparison", f1_fig)

        # Radar chart for all metrics
        radar_fig = plot_radar_chart(
            performance_comparison,
            metrics=['accuracy', 'precision', 'recall', 'f1_score'],
            title='Model Performance Radar Chart',
            output_dir=viz_dir,
            save_filename="model_performance_radar.png"
        )
        log_figure("model_performance_radar", radar_fig)

    # Plot ROC curves for each signal type if prediction probabilities are available
    for signal_type, results in all_results.items():
        pred_probas = {}
        y_true = None

        for model_name, metrics in results['evaluation'].items():
            if 'y_pred_proba' in metrics and 'y_true' in metrics:
                pred_probas[model_name] = metrics['y_pred_proba']
                if y_true is None:
                    y_true = metrics['y_true']

        if pred_probas and y_true is not None:
            # Plot ROC curves
            roc_fig = plot_roc_curves(
                y_true, pred_probas,
                title=f'ROC Curves - {signal_type}',
                output_dir=viz_dir,
                save_filename=f"roc_curves_{signal_type}.png"
            )
            log_figure(f"roc_curves_{signal_type}", roc_fig)

            # Plot precision-recall curves
            pr_fig = plot_precision_recall_curves(
                y_true, pred_probas,
                title=f'Precision-Recall Curves - {signal_type}',
                output_dir=viz_dir,
                save_filename=f"pr_curves_{signal_type}.png"
            )
            log_figure(f"pr_curves_{signal_type}", pr_fig)

    # Plot confusion matrices
    for signal_type, results in all_results.items():
        confusion_data = {}

        for model_name, metrics in results['evaluation'].items():
            if 'y_true' in metrics and 'y_pred' in metrics:
                confusion_data[model_name] = {
                    'y_true': metrics['y_true'],
                    'y_pred': metrics['y_pred']
                }

        if confusion_data:
            # Plot all confusion matrices
            cm_figs = plot_all_confusion_matrices(
                confusion_data,
                ["No Footstep", "Contralateral Footstep"],
                title_prefix=f'Confusion Matrix - {signal_type}',
                output_dir=os.path.join(viz_dir, signal_type),
                save_filename_prefix="confusion_matrix"
            )

            # Log individual figures
            for model_name, fig in cm_figs.items():
                log_figure(f"confusion_matrix_{signal_type}_{model_name}", fig)

    # Plot feature importances
    for signal_type, results in all_results.items():
        if 'feature_importances' in results and results['feature_importances']:
            # Calculate window size and num_neurons
            window_size = cfg.data.window_size
            num_neurons = splits['train'][signal_type].shape[1] // window_size

            # Plot feature importances
            plot_all_feature_importances(
                results['feature_importances'],
                window_size=window_size,
                num_neurons=num_neurons,
                output_dir=os.path.join(viz_dir, signal_type),
                save_filename_prefix="feature_importance"
            )

            # Log individual figures
            for model_name, feature_imp in results['feature_importances'].items():
                # Feature importance heatmap
                fi_heatmap = plot_feature_importance_heatmap(
                    feature_imp, window_size, num_neurons,
                    title=f'Feature Importance Heatmap - {model_name} - {signal_type}',
                    output_dir=None
                )
                log_figure(f"feature_importance_heatmap_{signal_type}_{model_name}", fi_heatmap)

                # Temporal importance
                temp_imp = plot_temporal_feature_importance(
                    feature_imp, window_size, num_neurons,
                    title=f'Temporal Feature Importance - {model_name} - {signal_type}',
                    output_dir=None
                )
                log_figure(f"temporal_importance_{signal_type}_{model_name}", temp_imp)

                # Neuron importance
                neuron_imp = plot_neuron_feature_importance(
                    feature_imp, window_size, num_neurons,
                    top_n=cfg.visualization.feature_importance.top_n,
                    title=f'Top Neuron Feature Importance - {model_name} - {signal_type}',
                    output_dir=None
                )
                log_figure(f"neuron_importance_{signal_type}_{model_name}", neuron_imp)

    # Step 7: Save overall results
    logger.info("Step 7: Saving overall results...")

    # Create serializable results
    serializable_results = {}
    for signal_type, results in performance_comparison.items():
        serializable_results[signal_type] = {}
        for model_name, metrics in results.items():
            serializable_results[signal_type][model_name] = {
                metric: float(value) for metric, value in metrics.items()
            }

    # Save to JSON
    with open(os.path.join(output_dir, "performance_metrics.json"), 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Record end time and print total runtime
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Experiment completed in {total_time:.2f} seconds.")

    # Log total runtime to W&B
    log_metrics({"total_runtime_seconds": total_time})

    # Finish W&B run
    finish_run()


if __name__ == "__main__":
    main()

