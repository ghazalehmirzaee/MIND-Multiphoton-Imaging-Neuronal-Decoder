# """
# Comprehensive visualization module for calcium imaging analysis.
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Union, Any
# import logging
# from mind.data.loader import find_most_active_neurons
# from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
#
# logger = logging.getLogger(__name__)
#
# # Set style for publication-quality figures
# plt.style.use('seaborn-v0_8-white')
# sns.set_palette("husl")
#
#
# def plot_signal_comparison_vertical(calcium_signals: Dict[str, np.ndarray],
#                                     neuron_indices: np.ndarray,
#                                     output_dir: Optional[Union[str, Path]] = None,
#                                     time_range: Optional[Tuple[int, int]] = None) -> plt.Figure:
#     """
#     Plot vertical comparison of three signal types for top neurons.
#
#     Parameters
#     ----------
#     calcium_signals : Dict[str, np.ndarray]
#         Dictionary containing all three signal types
#     neuron_indices : np.ndarray
#         Indices of neurons to plot
#     output_dir : Optional[Union[str, Path]]
#         Directory to save the figure
#     time_range : Optional[Tuple[int, int]]
#         Time range to plot (start, end)
#     """
#     # Signal types to plot
#     signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Raw Calcium Signal', 'ΔF/F Signal', 'Deconvolved Signal']
#
#     # Create figure with subplots
#     n_neurons = len(neuron_indices)
#     fig, axes = plt.subplots(n_neurons, 3, figsize=(15, 2 * n_neurons),
#                              gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
#
#     if n_neurons == 1:
#         axes = axes.reshape(1, -1)
#
#     # Get the actual frame count from the first available signal
#     max_frames = 0
#     for signal_type in signal_types:
#         signal = calcium_signals.get(signal_type)
#         if signal is not None:
#             max_frames = signal.shape[0]
#             break
#
#     # Determine time range, making sure it doesn't exceed available frames
#     if time_range is None:
#         time_range = (0, max_frames)
#     else:
#         # Clamp the range to actual available frames
#         time_range = (max(0, time_range[0]), min(max_frames, time_range[1]))
#
#     # Create time indices for slicing
#     time_indices = slice(time_range[0], time_range[1])
#
#     # Create time points array that matches the actual data length
#     time_points = np.arange(time_range[0], time_range[1])
#
#     # Plot each neuron and signal type
#     for i, neuron_idx in enumerate(neuron_indices):
#         for j, (signal_type, signal_name) in enumerate(zip(signal_types, signal_names)):
#             ax = axes[i, j]
#
#             signal = calcium_signals.get(signal_type)
#             if signal is None:
#                 ax.text(0.5, 0.5, 'Signal not available',
#                         ha='center', va='center', transform=ax.transAxes)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 continue
#
#             # Check if neuron index is valid for this signal
#             if neuron_idx >= signal.shape[1]:
#                 ax.text(0.5, 0.5, 'Neuron not in this signal',
#                         ha='center', va='center', transform=ax.transAxes)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 continue
#
#             # Get the signal data for this neuron and time range
#             signal_data = signal[time_indices, neuron_idx]
#
#             # Make sure time_points and signal_data have the same length
#             if len(time_points) != len(signal_data):
#                 # Adjust time_points to match signal_data length
#                 time_points_adjusted = np.arange(time_range[0], time_range[0] + len(signal_data))
#             else:
#                 time_points_adjusted = time_points
#
#             # Plot the signal with appropriate color
#             if signal_type == 'calcium_signal':
#                 ax.plot(time_points_adjusted, signal_data, 'b-', linewidth=0.5)
#             elif signal_type == 'deltaf_signal':
#                 ax.plot(time_points_adjusted, signal_data, 'g-', linewidth=0.5)
#             else:  # deconv_signal
#                 ax.plot(time_points_adjusted, signal_data, 'r-', linewidth=0.5)
#
#             # Set labels
#             if i == 0:
#                 ax.set_title(signal_name, fontsize=12, fontweight='bold')
#             if i == n_neurons - 1:
#                 ax.set_xlabel('Time (frames)')
#             if j == 0:
#                 ax.set_ylabel(f'Neuron {neuron_idx}')
#
#             # Remove top and right spines
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#
#     fig.suptitle('Signal Comparison for Top Active Neurons', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'signal_comparison_vertical.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved signal comparison to {output_path}")
#
#     return fig
#
#
# def plot_confusion_matrix_grid(results: Dict[str, Dict[str, Any]],
#                                output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot 5x3 grid of confusion matrices for all models and signal types.
#     """
#     # Order of models and signals as in the paper
#     models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
#     model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     fig, axes = plt.subplots(5, 3, figsize=(12, 20),
#                              gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
#
#     for i, (model, model_name) in enumerate(zip(models, model_names)):
#         for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#             ax = axes[i, j]
#
#             try:
#                 # Get confusion matrix from results
#                 cm = np.array(results[model][signal]['confusion_matrix'])
#
#                 # Calculate percentages
#                 cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
#
#                 # Plot confusion matrix
#                 sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
#                             cbar=False, ax=ax,
#                             xticklabels=['No footstep', 'Contralateral'],
#                             yticklabels=['No footstep', 'Contralateral'])
#
#                 # Add title
#                 title = f'{model_name} - {signal_name}'
#                 accuracy = results[model][signal]['metrics']['accuracy']
#                 ax.set_title(f'{title}\nAccuracy: {accuracy:.3f}')
#
#             except (KeyError, TypeError, ValueError) as e:
#                 ax.text(0.5, 0.5, 'No data available',
#                         ha='center', va='center', transform=ax.transAxes)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#
#     fig.suptitle('Binary Classification Confusion Matrices', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'confusion_matrix_grid.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved confusion matrix grid to {output_path}")
#
#     return fig
#
#
# def plot_roc_curve_grid(results: Dict[str, Dict[str, Any]],
#                         output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot 5x3 grid of ROC curves for all models and signal types.
#     """
#     models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
#     model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     fig, axes = plt.subplots(5, 3, figsize=(12, 20),
#                              gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
#
#     for i, (model, model_name) in enumerate(zip(models, model_names)):
#         for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#             ax = axes[i, j]
#
#             try:
#                 # Get ROC curve data
#                 curve_data = results[model][signal].get('curve_data', {})
#                 if 'roc' in curve_data:
#                     fpr = np.array(curve_data['roc']['fpr'])
#                     tpr = np.array(curve_data['roc']['tpr'])
#
#                     # Plot ROC curve
#                     ax.plot(fpr, tpr, 'b-', linewidth=2)
#                     ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#
#                     # Calculate AUC
#                     auc_score = auc(fpr, tpr)
#                     ax.set_title(f'{model_name} - {signal_name}\nAUC: {auc_score:.3f}')
#                 else:
#                     ax.text(0.5, 0.5, 'No ROC data',
#                             ha='center', va='center', transform=ax.transAxes)
#
#                 ax.set_xlabel('False Positive Rate')
#                 ax.set_ylabel('True Positive Rate')
#                 ax.grid(True, alpha=0.3)
#                 ax.set_xlim([0, 1])
#                 ax.set_ylim([0, 1])
#
#             except (KeyError, TypeError) as e:
#                 ax.text(0.5, 0.5, 'No data available',
#                         ha='center', va='center', transform=ax.transAxes)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#
#     fig.suptitle('ROC Curves - Binary Classification', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'roc_curve_grid.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved ROC curve grid to {output_path}")
#
#     return fig
#
#
# def plot_performance_radar_grid(results: Dict[str, Dict[str, Any]],
#                                 output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot radar charts comparing model performance across signal types.
#     """
#     models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
#     model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconv']
#     metrics = ['accuracy', 'precision', 'recall', 'f1_score']
#     metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
#
#     for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#         ax = axes[j]
#
#         # Number of metrics
#         num_vars = len(metrics)
#         angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#         angles += angles[:1]
#
#         # Plot for each model
#         for model, model_name in zip(models, model_names):
#             try:
#                 values = []
#                 for metric in metrics:
#                     value = results[model][signal]['metrics'][metric]
#                     values.append(value)
#                 values += values[:1]
#
#                 ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
#                 ax.fill(angles, values, alpha=0.25)
#
#             except KeyError:
#                 continue
#
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(metric_names)
#         ax.set_ylim(0, 1)
#         ax.set_title(signal_name, size=14, weight='bold', pad=20)
#         ax.grid(True)
#
#         if j == 2:
#             ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9))
#
#     fig.suptitle('Performance Radar Plots by Signal Type', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'performance_radar_grid.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved performance radar to {output_path}")
#
#     return fig
#
#
# def plot_feature_importance_heatmaps(results: Dict[str, Dict[str, Any]],
#                                      output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot feature importance heatmaps for each signal type.
#     """
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#         ax = axes[j]
#
#         # Use Random Forest importance as it's most reliable
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             importance_matrix = np.array(importance_summary['importance_matrix'])
#
#             # Plot heatmap
#             im = ax.imshow(importance_matrix.T, aspect='auto', cmap='viridis')
#             ax.set_xlabel('Time Step')
#             ax.set_ylabel('Neuron')
#             ax.set_title(f'{signal_name} - Feature Importance')
#
#             # Add colorbar
#             plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No importance data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Feature Importance Heatmaps', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'feature_importance_heatmaps.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved feature importance heatmaps to {output_path}")
#
#     return fig
#
#
# def plot_temporal_importance_patterns(results: Dict[str, Dict[str, Any]],
#                                       output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot temporal importance patterns for each signal type.
#     """
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#         ax = axes[j]
#
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             temporal_importance = np.array(importance_summary['temporal_importance'])
#
#             # Plot bar chart
#             ax.bar(range(len(temporal_importance)), temporal_importance, color='skyblue')
#             ax.set_xlabel('Time Step')
#             ax.set_ylabel('Importance')
#             ax.set_title(f'{signal_name} - Temporal Importance')
#             ax.grid(True, alpha=0.3)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No temporal data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Temporal Importance Patterns', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'temporal_importance_patterns.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved temporal importance patterns to {output_path}")
#
#     return fig
#
#
# def plot_top_neuron_importance(results: Dict[str, Dict[str, Any]],
#                                output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot top neuron importance for each signal type.
#     """
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#         ax = axes[j]
#
#         try:
#             importance_summary = results['random_forest'][signal]['importance_summary']
#             neuron_importance = np.array(importance_summary['neuron_importance'])
#             top_indices = np.array(importance_summary['top_neuron_indices'])[:20]
#
#             # Get importance values for top neurons
#             top_importance = neuron_importance[top_indices]
#
#             # Plot bar chart
#             ax.bar(range(len(top_importance)), top_importance, color='coral')
#             ax.set_xlabel('Neuron Rank')
#             ax.set_ylabel('Mean Feature Importance')
#             ax.set_title(f'{signal_name} - Top 20 Neurons')
#             ax.set_xticks(range(0, 20, 5))
#             ax.set_xticklabels([f'N{i}' for i in top_indices[::5]])
#             ax.grid(True, alpha=0.3)
#
#         except (KeyError, TypeError, ValueError):
#             ax.text(0.5, 0.5, 'No neuron data',
#                     ha='center', va='center', transform=ax.transAxes)
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     fig.suptitle('Top 20 Neuron Importance', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'top_neuron_importance.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved top neuron importance to {output_path}")
#
#     return fig
#
#
# def plot_model_performance_heatmap(results: Dict[str, Dict[str, Any]],
#                                    output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot heatmap of F1 scores for all model-signal combinations.
#     """
#     models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
#     model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     # Create matrix of F1 scores
#     f1_matrix = np.zeros((len(models), len(signals)))
#
#     for i, model in enumerate(models):
#         for j, signal in enumerate(signals):
#             try:
#                 f1_score = results[model][signal]['metrics']['f1_score']
#                 f1_matrix[i, j] = f1_score
#             except KeyError:
#                 f1_matrix[i, j] = np.nan
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     # Plot heatmap
#     sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
#                 xticklabels=signal_names, yticklabels=model_names,
#                 cbar_kws={'label': 'F1 Score'}, ax=ax)
#
#     ax.set_title('Model Performance Heatmap (F1 Scores)', fontsize=14, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'model_performance_heatmap.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved model performance heatmap to {output_path}")
#
#     return fig
#
#
# def plot_precision_recall_grid(results: Dict[str, Dict[str, Any]],
#                                output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot 5x3 grid of precision-recall curves for all models and signal types.
#     """
#     models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
#     model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
#     signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
#     signal_names = ['Calcium', 'ΔF/F', 'Deconvolved']
#
#     fig, axes = plt.subplots(5, 3, figsize=(12, 20),
#                              gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
#
#     for i, (model, model_name) in enumerate(zip(models, model_names)):
#         for j, (signal, signal_name) in enumerate(zip(signals, signal_names)):
#             ax = axes[i, j]
#
#             try:
#                 # Get precision-recall curve data
#                 curve_data = results[model][signal].get('curve_data', {})
#                 if 'precision_recall' in curve_data:
#                     precision = np.array(curve_data['precision_recall']['precision'])
#                     recall = np.array(curve_data['precision_recall']['recall'])
#
#                     # Plot precision-recall curve
#                     ax.plot(recall, precision, 'g-', linewidth=2)
#
#                     # Calculate AUC
#                     pr_auc = auc(recall, precision)
#                     ax.set_title(f'{model_name} - {signal_name}\nAUC: {pr_auc:.3f}')
#                 else:
#                     ax.text(0.5, 0.5, 'No PR data',
#                             ha='center', va='center', transform=ax.transAxes)
#
#                 ax.set_xlabel('Recall')
#                 ax.set_ylabel('Precision')
#                 ax.grid(True, alpha=0.3)
#                 ax.set_xlim([0, 1])
#                 ax.set_ylim([0, 1])
#
#             except (KeyError, TypeError) as e:
#                 ax.text(0.5, 0.5, 'No data available',
#                         ha='center', va='center', transform=ax.transAxes)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#
#     fig.suptitle('Precision-Recall Curves - Binary Classification', fontsize=16, fontweight='bold')
#
#     if output_dir:
#         output_path = Path(output_dir) / 'precision_recall_grid.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved precision-recall curve grid to {output_path}")
#
#     return fig
#
#
# def plot_activity_heatmap_top_neurons(calcium_signals: Dict[str, np.ndarray],
#                                       n_neurons: int = 250,
#                                       signal_type: str = 'deconv_signal',
#                                       output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
#     """
#     Plot heatmap of activity for top n most active neurons.
#     """
#     # Find most active neurons
#     top_indices = find_most_active_neurons(calcium_signals, n_neurons, signal_type)
#
#     # Get signal data
#     signal = calcium_signals.get(signal_type)
#     if signal is None:
#         logger.error(f"Signal type {signal_type} not available")
#         return None
#
#     # Extract data for top neurons
#     data = signal[:, top_indices].T
#
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Plot heatmap
#     im = ax.imshow(data, aspect='auto', cmap='hot', interpolation='nearest')
#
#     ax.set_xlabel('Time (frames)')
#     ax.set_ylabel('Neuron')
#     ax.set_title(f'Activity Heatmap - Top {n_neurons} Active Neurons ({signal_type})')
#
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Activity Level')
#
#     # Set y-axis to show subset of neuron labels
#     n_labels = 10
#     label_indices = np.linspace(0, n_neurons - 1, n_labels, dtype=int)
#     ax.set_yticks(label_indices)
#     ax.set_yticklabels([f'N{top_indices[i]}' for i in label_indices])
#
#     if output_dir:
#         output_path = Path(output_dir) / f'activity_heatmap_top{n_neurons}.png'
#         fig.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Saved activity heatmap to {output_path}")
#
#     return fig
#
#
# def create_all_visualizations(results: Dict[str, Dict[str, Any]],
#                               calcium_signals: Dict[str, np.ndarray],
#                               output_dir: Union[str, Path]) -> None:
#     """
#     Create all required visualizations for the paper.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     logger.info("Creating all visualizations...")
#
#     try:
#         # 1. Signal comparisons for top 20 neurons
#         top_20_indices = find_most_active_neurons(calcium_signals, 20)
#         plot_signal_comparison_vertical(calcium_signals, top_20_indices, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create signal comparison: {e}")
#
#     try:
#         # 2. Confusion matrix grid
#         plot_confusion_matrix_grid(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create confusion matrix grid: {e}")
#
#     try:
#         # 3. ROC curve grid
#         plot_roc_curve_grid(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create ROC curve grid: {e}")
#
#     try:
#         # 4. Performance radar plots
#         plot_performance_radar_grid(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create performance radar: {e}")
#
#     try:
#         # 5. Feature importance heatmaps
#         plot_feature_importance_heatmaps(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create feature importance heatmaps: {e}")
#
#     try:
#         # 6. Temporal importance patterns
#         plot_temporal_importance_patterns(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create temporal importance patterns: {e}")
#
#     try:
#         # 7. Top neuron importance
#         plot_top_neuron_importance(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create top neuron importance: {e}")
#
#     try:
#         # 8. Model performance heatmap
#         plot_model_performance_heatmap(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create model performance heatmap: {e}")
#
#     try:
#         # 9. Precision-recall curves
#         plot_precision_recall_grid(results, output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create precision-recall curves: {e}")
#
#     try:
#         # 10. Activity heatmap for top 250 neurons
#         plot_activity_heatmap_top_neurons(calcium_signals, 250, 'deconv_signal', output_dir)
#     except Exception as e:
#         logger.error(f"Failed to create activity heatmap: {e}")
#
#     logger.info("Visualization creation completed!")
#


"""
Enhanced visualization module with consistent color coding and improved plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from mind.data.loader import find_most_active_neurons
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

logger = logging.getLogger(__name__)

# Define consistent color scheme for signal types
SIGNAL_COLORS = {
    'calcium_signal': '#3498db',  # Blue
    'deltaf_signal': '#2ecc71',  # Green
    'deconv_signal': '#e74c3c'  # Red
}

# Define consistent display names
SIGNAL_DISPLAY_NAMES = {
    'calcium_signal': 'Calcium',
    'deltaf_signal': 'ΔF/F',
    'deconv_signal': 'Deconvolved'
}

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-white')
sns.set_palette("husl")


def plot_signal_comparison_vertical(calcium_signals: Dict[str, np.ndarray],
                                    neuron_indices: np.ndarray,
                                    output_dir: Optional[Union[str, Path]] = None,
                                    time_range: Optional[Tuple[int, int]] = None,
                                    n_neurons: int = 10) -> plt.Figure:
    """
    Plot vertical comparison of three signal types for top neurons.
    Now with consistent color coding and configurable number of neurons.

    Parameters
    ----------
    calcium_signals : Dict[str, np.ndarray]
        Dictionary containing all three signal types
    neuron_indices : np.ndarray
        Indices of neurons to plot
    output_dir : Optional[Union[str, Path]]
        Directory to save the figure
    time_range : Optional[Tuple[int, int]]
        Time range to plot (start, end)
    n_neurons : int
        Number of neurons to display (default 10)
    """
    # Signal types to plot with consistent colors
    signal_types = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # Use only the requested number of neurons
    neuron_indices = neuron_indices[:n_neurons]

    # Create figure with subplots
    fig, axes = plt.subplots(n_neurons, 3, figsize=(15, 2 * n_neurons),
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.2})

    if n_neurons == 1:
        axes = axes.reshape(1, -1)

    # Get the actual frame count
    max_frames = 0
    for signal_type in signal_types:
        signal = calcium_signals.get(signal_type)
        if signal is not None:
            max_frames = signal.shape[0]
            break

    # Determine time range
    if time_range is None:
        time_range = (0, min(1000, max_frames))  # Show first 1000 frames by default
    else:
        time_range = (max(0, time_range[0]), min(max_frames, time_range[1]))

    time_indices = slice(time_range[0], time_range[1])
    time_points = np.arange(time_range[0], time_range[1])

    # Plot each neuron and signal type
    for i, neuron_idx in enumerate(neuron_indices):
        for j, signal_type in enumerate(signal_types):
            ax = axes[i, j]

            signal = calcium_signals.get(signal_type)
            if signal is None:
                ax.text(0.5, 0.5, 'Signal not available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Check if neuron index is valid
            if neuron_idx >= signal.shape[1]:
                ax.text(0.5, 0.5, 'Neuron not in this signal',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Get the signal data
            signal_data = signal[time_indices, neuron_idx]

            # Plot with consistent color
            color = SIGNAL_COLORS[signal_type]
            ax.plot(time_points, signal_data, color=color, linewidth=0.8)

            # Set labels
            if i == 0:
                ax.set_title(SIGNAL_DISPLAY_NAMES[signal_type],
                             fontsize=12, fontweight='bold')
            if i == n_neurons - 1:
                ax.set_xlabel('Time (frames)')
            if j == 0:
                ax.set_ylabel(f'Neuron {neuron_idx}')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'Signal Comparison for Top {n_neurons} Active Neurons',
                 fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / f'signal_comparison_top{n_neurons}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved signal comparison to {output_path}")

    return fig


def plot_confusion_matrix_grid(results: Dict[str, Dict[str, Any]],
                               output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot confusion matrices with improved visualization and class balance awareness.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(5, 3, figsize=(12, 20),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        for j, signal in enumerate(signals):
            ax = axes[i, j]

            # Get color for this signal type
            signal_color = SIGNAL_COLORS[signal]

            try:
                # Get confusion matrix and metrics
                cm = np.array(results[model][signal]['confusion_matrix'])
                metrics = results[model][signal]['metrics']

                # Calculate percentages row-wise (important for imbalanced data)
                cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

                # Create custom colormap based on signal color
                cmap = sns.light_palette(signal_color, as_cmap=True)

                # Plot confusion matrix
                sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap=cmap,
                            cbar=False, ax=ax,
                            xticklabels=['No footstep', 'Contralateral'],
                            yticklabels=['No footstep', 'Contralateral'])

                # Add title with metrics
                accuracy = metrics.get('accuracy', 0)
                f1_score = metrics.get('f1_score', 0)
                ax.set_title(f'{model_name} - {SIGNAL_DISPLAY_NAMES[signal]}\n'
                             f'Acc: {accuracy:.3f}, F1: {f1_score:.3f}',
                             fontsize=11)

                # Add border in signal color
                for spine in ax.spines.values():
                    spine.set_edgecolor(signal_color)
                    spine.set_linewidth(2)

            except (KeyError, TypeError, ValueError) as e:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle('Binary Classification Confusion Matrices', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'confusion_matrix_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix grid to {output_path}")

    return fig


def plot_performance_radar_grid(results: Dict[str, Dict[str, Any]],
                                output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot radar charts with consistent colors for each signal type.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

    for j, signal in enumerate(signals):
        ax = axes[j]
        signal_color = SIGNAL_COLORS[signal]

        # Number of metrics
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Create color variations for different models
        color_variations = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(models)))

        # Plot for each model
        for idx, (model, model_name) in enumerate(zip(models, model_names)):
            try:
                values = []
                for metric in metrics:
                    value = results[model][signal]['metrics'][metric]
                    values.append(value)
                values += values[:1]

                # Use different line styles for different models
                line_style = ['-', '--', '-.', ':', '-'][idx]
                ax.plot(angles, values, line_style, linewidth=2,
                        label=model_name, color=color_variations[idx])
                ax.fill(angles, values, alpha=0.1, color=color_variations[idx])

            except KeyError:
                continue

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(f'{SIGNAL_DISPLAY_NAMES[signal]}',
                     size=14, weight='bold', pad=20, color=signal_color)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')

        # Add colored border
        ax.spines['polar'].set_color(signal_color)
        ax.spines['polar'].set_linewidth(2)

        if j == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0.9))

    fig.suptitle('Performance Radar Plots by Signal Type', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'performance_radar_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance radar to {output_path}")

    return fig


def plot_roc_curve_grid(results: Dict[str, Dict[str, Any]],
                        output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot ROC curves with consistent colors and improved styling.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(5, 3, figsize=(12, 20),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        for j, signal in enumerate(signals):
            ax = axes[i, j]
            signal_color = SIGNAL_COLORS[signal]

            try:
                # Get ROC curve data
                curve_data = results[model][signal].get('curve_data', {})
                if 'roc' in curve_data:
                    fpr = np.array(curve_data['roc']['fpr'])
                    tpr = np.array(curve_data['roc']['tpr'])

                    # Plot ROC curve with signal color
                    ax.plot(fpr, tpr, color=signal_color, linewidth=2.5)
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

                    # Calculate and display AUC
                    auc_score = auc(fpr, tpr)
                    ax.set_title(f'{model_name} - {SIGNAL_DISPLAY_NAMES[signal]}\n'
                                 f'AUC: {auc_score:.3f}', fontsize=11)

                    # Fill area under curve with transparent color
                    ax.fill_between(fpr, tpr, alpha=0.2, color=signal_color)
                else:
                    ax.text(0.5, 0.5, 'No ROC data',
                            ha='center', va='center', transform=ax.transAxes)

                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(signal_color)
                    spine.set_linewidth(1.5)

            except (KeyError, TypeError) as e:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle('ROC Curves - Binary Classification', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'roc_curve_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve grid to {output_path}")

    return fig


def plot_precision_recall_grid(results: Dict[str, Dict[str, Any]],
                               output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot precision-recall curves with consistent colors.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    fig, axes = plt.subplots(5, 3, figsize=(12, 20),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.3})

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        for j, signal in enumerate(signals):
            ax = axes[i, j]
            signal_color = SIGNAL_COLORS[signal]

            try:
                # Get precision-recall curve data
                curve_data = results[model][signal].get('curve_data', {})
                if 'precision_recall' in curve_data:
                    precision = np.array(curve_data['precision_recall']['precision'])
                    recall = np.array(curve_data['precision_recall']['recall'])

                    # Plot precision-recall curve with signal color
                    ax.plot(recall, precision, color=signal_color, linewidth=2.5)

                    # Calculate and display AUC
                    pr_auc = auc(recall, precision)
                    ax.set_title(f'{model_name} - {SIGNAL_DISPLAY_NAMES[signal]}\n'
                                 f'AUC: {pr_auc:.3f}', fontsize=11)

                    # Fill area under curve
                    ax.fill_between(recall, precision, alpha=0.2, color=signal_color)

                    # Add baseline (for balanced dataset would be ~0.28)
                    baseline = 0.2814  # Percentage of positive class
                    ax.axhline(y=baseline, color='gray', linestyle='--',
                               alpha=0.7, label=f'Baseline: {baseline:.3f}')
                else:
                    ax.text(0.5, 0.5, 'No PR data',
                            ha='center', va='center', transform=ax.transAxes)

                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(signal_color)
                    spine.set_linewidth(1.5)

            except (KeyError, TypeError) as e:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle('Precision-Recall Curves - Binary Classification', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'precision_recall_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve grid to {output_path}")

    return fig


def plot_model_performance_heatmap(results: Dict[str, Dict[str, Any]],
                                   output_dir: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot heatmap of model performance with consistent signal colors.
    """
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    model_names = ['Random Forest', 'SVM', 'MLP', 'FCNN', 'CNN']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    # Create matrices for different metrics
    metrics_to_plot = ['f1_score', 'accuracy', 'roc_auc']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        matrix = np.zeros((len(models), len(signals)))

        for i, model in enumerate(models):
            for j, signal in enumerate(signals):
                try:
                    if metric == 'roc_auc':
                        matrix[i, j] = results[model][signal]['metrics'].get('roc_auc', np.nan)
                    else:
                        matrix[i, j] = results[model][signal]['metrics'][metric]
                except KeyError:
                    matrix[i, j] = np.nan

        # Create custom colormap
        if metric == 'f1_score':
            cmap = 'YlOrRd'
            title = 'F1 Score'
        elif metric == 'accuracy':
            cmap = 'Blues'
            title = 'Accuracy'
        else:
            cmap = 'Greens'
            title = 'ROC AUC'

        # Plot heatmap
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap=cmap,
                    xticklabels=[SIGNAL_DISPLAY_NAMES[s] for s in signals],
                    yticklabels=model_names,
                    cbar_kws={'label': title}, ax=ax)

        ax.set_title(f'{title} Heatmap', fontsize=12, fontweight='bold')

        # Color the x-axis labels according to signal type
        for ticklabel, signal in zip(ax.get_xticklabels(), signals):
            ticklabel.set_color(SIGNAL_COLORS[signal])
            ticklabel.set_weight('bold')

    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    if output_dir:
        output_path = Path(output_dir) / 'model_performance_heatmap.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model performance heatmap to {output_path}")

    return fig


def create_all_visualizations(results: Dict[str, Dict[str, Any]],
                              calcium_signals: Dict[str, np.ndarray],
                              output_dir: Union[str, Path]) -> None:
    """
    Create all required visualizations with consistent styling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating all visualizations with consistent color scheme...")

    try:
        # 1. Signal comparisons for top 10 neurons (as requested)
        top_10_indices = find_most_active_neurons(calcium_signals, 10)
        plot_signal_comparison_vertical(calcium_signals, top_10_indices, output_dir, n_neurons=10)
    except Exception as e:
        logger.error(f"Failed to create signal comparison: {e}")

    try:
        # 2. Confusion matrix grid
        plot_confusion_matrix_grid(results, output_dir)
    except Exception as e:
        logger.error(f"Failed to create confusion matrix grid: {e}")

    try:
        # 3. ROC curve grid
        plot_roc_curve_grid(results, output_dir)
    except Exception as e:
        logger.error(f"Failed to create ROC curve grid: {e}")

    try:
        # 4. Performance radar plots
        plot_performance_radar_grid(results, output_dir)
    except Exception as e:
        logger.error(f"Failed to create performance radar: {e}")

    try:
        # 5. Precision-recall curves
        plot_precision_recall_grid(results, output_dir)
    except Exception as e:
        logger.error(f"Failed to create precision-recall curves: {e}")

    try:
        # 6. Model performance heatmap
        plot_model_performance_heatmap(results, output_dir)
    except Exception as e:
        logger.error(f"Failed to create model performance heatmap: {e}")

    logger.info("Visualization creation completed!")

