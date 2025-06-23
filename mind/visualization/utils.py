"""
Utility functions for the visualization system.

This module provides helper functions used across multiple visualization components,
including dependency checking, file operations, and data processing.
"""
import logging
import importlib
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required visualization dependencies are installed.
    """
    dependencies = {
        'matplotlib': True,  # Always available as it's a core dependency
        'seaborn': True,  # Always available as it's a core dependency
        'matplotlib_venn': False,
        'scipy': True,  # Always available as it's a core dependency
        'numpy': True  # Always available as it's a core dependency
    }

    # Check for matplotlib_venn
    try:
        importlib.import_module('matplotlib_venn')
        dependencies['matplotlib_venn'] = True
    except ImportError:
        logger.warning("matplotlib_venn is not installed. Venn diagram visualization will not be available. "
                       "Install with: pip install matplotlib-venn")

    return dependencies


def create_output_directory(base_dir: str, subdirs: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Create output directory structure.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    result = {'base': base_path}

    if subdirs:
        for subdir in subdirs:
            path = base_path / subdir
            path.mkdir(parents=True, exist_ok=True)
            result[subdir] = path

    return result


def save_figure(fig, output_path: str, dpi: int = 300, bbox_inches: str = 'tight', pad_inches: float = 0.1):
    """
    Save a figure with proper configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches
    )

    logger.info(f"Saved figure to {output_path}")


def create_visualization_summary(output_dir: Path, subdirs: Dict[str, Path]):
    """
    Create a summary report of all generated visualizations.
    """
    summary_path = output_dir / 'visualization_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("Calcium Imaging Neural Decoding - Visualization Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Generated on: " + plt.matplotlib.dates.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        f.write("All visualizations use consistent color coding:\n")
        f.write("- Calcium Signal: Blue (#356d9e)\n")
        f.write("- ΔF/F Signal: Green (#4c8b64)\n")
        f.write("- Deconvolved Signal: Red (#a85858)\n\n")

        f.write("Generated Visualizations:\n")
        f.write("-" * 30 + "\n\n")

        for name, path in subdirs.items():
            if name == 'base':
                continue

            f.write(f"{name.replace('_', ' ').title()}:\n")
            f.write("-" * len(name) + "\n")

            # List all PNG files in the subdirectory
            png_files = sorted(path.glob('*.png'))

            if not png_files:
                f.write("  No visualizations generated\n\n")
                continue

            for png_file in png_files:
                f.write(f"  - {png_file.name}\n")

            f.write("\n")

        # Add dependency information
        dependencies = check_dependencies()
        f.write("Visualization Dependencies:\n")
        f.write("-" * 30 + "\n")

        for package, available in dependencies.items():
            status = "Available" if available else "Not Available"
            f.write(f"  - {package}: {status}\n")

        if not dependencies.get('matplotlib_venn', False):
            f.write("\nNote: matplotlib_venn is not installed. Venn diagram visualization is not available.\n")
            f.write("Install with: pip install matplotlib-venn\n")

    logger.info(f"Created visualization summary at {summary_path}")

