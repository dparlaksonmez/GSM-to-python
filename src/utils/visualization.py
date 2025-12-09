"""
File's primary purpose and role in the pipeline:
This module provides utility functions for visualizing data distributions and relationships in the context of bioinformatics analysis.

Key functions:
- plot_histogram: Plots a histogram of the given data.
- plot_scatter: Creates a scatter plot for two variables.
- plot_box: Generates a box plot for visualizing data distributions.

Usage examples:
    plot_histogram(data, bins=30)
    plot_scatter(x, y)
    plot_box(data)

Important notes:
- Ensure that matplotlib is installed in your environment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging

def plot_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """Plots a histogram of the given data."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_scatter(x, y, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    """Creates a scatter plot for two variables."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='green', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_box(data, title='Box Plot', xlabel='Categories', ylabel='Values'):
    """Generates a box plot for visualizing data distributions."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

from typing import Optional

def visualize_f1_scores(results_df: pd.DataFrame, output_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    """
    Visualize F1 scores from GSM pipeline iterations.
    
    Generates two plots:
    1. F1 Scores across Iterations (Line plot)
    2. Average F1 Scores by Number of Groups (Bar plot with error bars)
    
    Args:
        results_df: DataFrame containing 'Iteration', 'NumGroups', and 'F1Score' columns.
        output_dir: Directory to save the plots.
        logger: Optional logger.
    """
    if logger:
        logger.info("📊 Generating F1 score visualizations...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: F1 Scores across Iterations
    plt.figure(figsize=(12, 6))
    # Convert NumGroups to categorical for better plotting if needed, but integer is fine for hue
    sns.lineplot(data=results_df, x="Iteration", y="F1Score", hue="NumGroups", marker="o", palette="viridis")
    plt.title("F1 Scores across Iterations for Different Number of Groups")
    plt.xlabel("Iteration")
    plt.ylabel("F1 Score")
    plt.legend(title="Number of Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "f1_scores_across_iterations.png", dpi=300)
    plt.close()
    
    # Plot 2: Average F1 Scores by Number of Groups (Bar plot as requested in TODO)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="NumGroups", y="F1Score", errorbar="sd", palette="viridis", capsize=.1)
    plt.title("Average F1 Score by Number of Groups (across all iterations)")
    plt.xlabel("Number of Groups Used")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(output_dir / "average_f1_scores_by_groups.png", dpi=300)
    plt.close()
    
    if logger:
        logger.info(f"✅ Plots saved to {output_dir}")
