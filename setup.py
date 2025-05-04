from setuptools import setup, find_packages

setup(
    name="mind",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "wandb>=0.12.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.4.0",
        "hydra-core>=1.1.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Multiphoton Imaging Neural Decoder",
    keywords="calcium-imaging, neural-decoding, machine-learning, deep-learning",
    python_requires=">=3.7",
)

