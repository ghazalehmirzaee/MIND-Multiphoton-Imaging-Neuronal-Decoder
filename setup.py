# setup.py
from setuptools import setup, find_packages

setup(
    name="mind",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "h5py>=3.3.0",
        "hdf5storage>=0.1.18",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",
        "wandb>=0.12.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.7",
    ],
)

