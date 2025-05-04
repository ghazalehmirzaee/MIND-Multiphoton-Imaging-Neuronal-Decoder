# MIND: Multiphoton Imaging Neural Decoder

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

MIND (Multiphoton Imaging Neural Decoder) is a framework for decoding behavior from calcium imaging data using various machine learning and deep learning approaches. The framework allows comprehensive comparison between different signal types (raw calcium signals, ΔF/F, and deconvolved signals) and model architectures.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MIND-Multiphoton-Imaging-Neural-Decoder.git
cd MIND-Multiphoton-Imaging-Neural-Decoder

# Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install the package and dependencies
pip install -e .
