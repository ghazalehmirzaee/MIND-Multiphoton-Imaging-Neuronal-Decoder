# MIND: Multiphoton Imaging Neuronal Decoder

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)

## Overview

MIND (Multiphoton Imaging Neuronal Decoder) is a framework for decoding behavior from calcium imaging data using various machine learning and deep learning approaches. The framework allows comprehensive comparison between different signal types (raw calcium signals, ΔF/F, and deconvolved signals) and model architectures.


## Project Structure

```bash
MIND-Multiphoton-Imaging-Neuronal-Decoder/
├── data/                     # Data storage directory
│   └── raw/                  # Raw MATLAB and behavioral Excel data files
├── experiments/              # Experiment runner scripts
├── mind/                     # Main package code
│   ├── config/               # Configuration files (Hydra)
│   ├── data/                 # Data handling
│   ├── evaluation/           # Metrics and analysis
│   ├── models/               # Model implementations
│   │   ├── classical/        # Traditional ML models
│   │   └── deep/             # Neural network models
│   ├── training/             # Training utilities
│   ├── utils/                # Helper functions
│   └── visualization/        # Plotting and visualization
├── notebooks/                # Analysis notebooks
├── results/                  # Output directory for results
│   ├── models/               # Saved models
│   └── visualizations/       # Generated figures
└── tests/                    # Unit and integration tests

```

## Installation

```bash
# Clone the repository
git clone https://github.com/ghazalehmirzaee/MIND-Multiphoton-Imaging-Neuronal-Decoder.git
cd MIND-Multiphoton-Imaging-Neuronal-Decoder

# Create and activate conda environment
conda create -n mind python=3.9
conda activate mind

# Install dependencies
pip install -e 

```

## Usage
## Running the Model Comparison Experiment

```bash
# Run with default configuration
python experiments/run_experiment.py --dataset all --visualize --export-excel --fixed-axes

# Run with custom configuration
python experiments/run_experiment.py --dataset all --visualize --export-excel --fixed-axes
```

## Data
The dataset we used in our research comprises behavioral tracking and simultaneous neuronal recordings via two-photon calcium imaging, collected during mouse reaching tasks. Specifically, the dataset includes:

Behavioral data (footstep events) from Excel files
MATLAB files containing:

- Raw calcium signals (2999 time points × 764 neurons)
- Deconvolved signals (2999 × 581 neurons)
- ∆F/F signals (2999 × 581 neurons)

```bash
mind/
  ├── config/            # Hydra configurations
  ├── data/              # Data loading and processing
  ├── models/            # Model implementations
  │   ├── classical/     # Random Forest, SVM, MLP
  │   └── deep/          # FCNN, CNN
  ├── training/          # Training pipelines
  ├── evaluation/        # Metrics and analysis
  ├── visualization/     # Visualization utilities
  └── utils/             # Helper functions
experiments/            # Main experiment scripts
tests/                  # Unit tests
requirements.txt        # Dependencies
setup.py                # Package setup
README.md              # Project documentation
```

