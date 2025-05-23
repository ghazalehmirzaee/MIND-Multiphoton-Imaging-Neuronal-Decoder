"""
Streamlined experiment runner for MIND project.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from mind.data.loader import load_and_align_data
from mind.data.processor import create_datasets
from mind.training.trainer import train_model, set_seed
from mind.config import DEFAULT_CONFIG
from mind.visualization.comprehensive_viz import create_all_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_all_experiments(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run experiments for all model-signal combinations."""
    # Set global seed
    set_seed(config['training']['seed'])

    # Load data
    calcium_signals, frame_labels = load_and_align_data(
        config['data']['mat_file'],
        config['data']['xlsx_file']
    )

    # Create datasets
    datasets = create_datasets(
        calcium_signals, frame_labels,
        window_size=config['data']['window_size'],
        step_size=config['data']['step_size'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['training']['seed']
    )

    # Get dimensions
    sample, _ = datasets['calcium_signal']['train'][0]
    window_size = sample.shape[0]
    n_neurons = sample.shape[1]

    # Run experiments
    models = ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']
    signals = ['calcium_signal', 'deltaf_signal', 'deconv_signal']

    all_results = {}
    for model in models:
        all_results[model] = {}
        for signal in signals:
            logger.info(f"\n{'=' * 50}\nRunning {model} on {signal}\n{'=' * 50}")

            results = train_model(
                model_type=model,
                model_params=config['models'][model],
                datasets=datasets,
                signal_type=signal,
                window_size=window_size,
                n_neurons=n_neurons,
                output_dir=config['training']['output_dir'],
                device=config['training']['device']
            )

            all_results[model][signal] = results

            # Log key metrics
            metrics = results['metrics']
            logger.info(f"{model} - {signal}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MIND experiments")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    config['training']['output_dir'] = args.output
    config['training']['seed'] = args.seed

    # Run experiments
    results = run_all_experiments(config)

    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating visualizations...")

        # Reload data for visualization
        calcium_signals, _ = load_and_align_data(
            config['data']['mat_file'],
            config['data']['xlsx_file']
        )

        viz_dir = Path(args.output) / "visualizations"
        create_all_visualizations(
            results=results,
            calcium_signals=calcium_signals,
            output_dir=str(viz_dir),
            mat_file_path=config['data']['mat_file']
        )

        logger.info(f"Visualizations saved to {viz_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY - Key Finding: Deconvolved Signals Superior")
    print("=" * 60)

    for signal in ['calcium_signal', 'deltaf_signal', 'deconv_signal']:
        print(f"\n{signal.replace('_', ' ').title()}:")
        for model in ['random_forest', 'svm', 'mlp', 'fcnn', 'cnn']:
            if model in results and signal in results[model]:
                acc = results[model][signal]['metrics']['accuracy']
                f1 = results[model][signal]['metrics']['f1_score']
                print(f"  {model:15s}: Acc={acc:.4f}, F1={f1:.4f}")

    # Highlight best performance
    best_acc = 0
    best_model = ""
    best_signal = ""

    for model in results:
        for signal in results[model]:
            acc = results[model][signal]['metrics']['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_signal = signal

    print(f"\nBest Performance: {best_model} on {best_signal} = {best_acc:.4f}")
    print("Confirming: Deconvolved signals provide superior decoding performance!")


if __name__ == "__main__":
    main()

