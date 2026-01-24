"""
📊 Figure Generation Module for GSM Pipeline

Purpose:
    Generate publication-quality figures for manuscript preparation.
    Creates ROC curves, confusion matrices, feature importance plots,
    and performance comparison charts.

Key Functions:
    - generate_all_figures: Main entry point to create all figures
    - plot_roc_curves: ROC curves with AUC values
    - plot_confusion_matrix: Heatmap of classification results
    - plot_feature_importance: Bar chart of top features
    - plot_performance_boxplot: F1 stability across iterations

Example Usage:
    >>> from src.utils.generate_figures import generate_all_figures
    >>> generate_all_figures(output_dir, results_json_path, logger)
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


##### MATPLOTLIB CONFIGURATION #####
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    output_dir: Path
    format: str = 'png'
    dpi: int = 300
    figsize_single: tuple = (8, 6)
    figsize_wide: tuple = (12, 6)
    color_palette: str = 'viridis'


##### MAIN ENTRY POINT #####
def generate_all_figures(
    output_dir: Path,
    results_json_path: Path,
    logger: logging.Logger
) -> Dict[str, Path]:
    """Generate all publication figures from modeling results.
    
    Args:
        output_dir: Directory to save figures
        results_json_path: Path to modeling_results_all_iterations.json
        logger: Logger instance
    
    Returns:
        Dictionary mapping figure names to their file paths
    """
    logger.info("📊 Starting figure generation for publication...")
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    config = FigureConfig(output_dir=figures_dir)
    
    # Load results
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)
    
    generated_figures = {}
    
    # Generate each figure type
    try:
        path = plot_feature_importance_aggregated(all_results, config, logger)
        generated_figures['feature_importance'] = path
    except Exception as e:
        logger.warning(f"⚠️ Could not generate feature importance plot: {e}")
    
    try:
        path = plot_performance_boxplot(all_results, config, logger)
        generated_figures['performance_boxplot'] = path
    except Exception as e:
        logger.warning(f"⚠️ Could not generate performance boxplot: {e}")
    
    try:
        path = plot_auc_roc_comparison(all_results, config, logger)
        generated_figures['auc_roc_comparison'] = path
    except Exception as e:
        logger.warning(f"⚠️ Could not generate AUC-ROC comparison: {e}")
    
    try:
        path = plot_confidence_interval_forest(all_results, config, logger)
        generated_figures['ci_forest_plot'] = path
    except Exception as e:
        logger.warning(f"⚠️ Could not generate CI forest plot: {e}")
    
    try:
        path = plot_cv_stability(all_results, config, logger)
        generated_figures['cv_stability'] = path
    except Exception as e:
        logger.warning(f"⚠️ Could not generate CV stability plot: {e}")
    
    try:
        path = plot_group_performance_heatmap(all_results, config, logger)
        generated_figures['group_heatmap'] = path
    except Exception as e:
        logger.warning(f"⚠️ Could not generate group heatmap: {e}")
    
    logger.info(f"✅ Generated {len(generated_figures)} figures in {figures_dir}")
    return generated_figures


##### FEATURE IMPORTANCE PLOT #####
def plot_feature_importance_aggregated(
    all_results: List[Dict],
    config: FigureConfig,
    logger: logging.Logger,
    top_n: int = 15
) -> Path:
    """Create aggregated feature importance bar chart across all iterations.
    
    Combines feature importances from all iterations and shows mean ± std.
    """
    logger.info("📈 Generating aggregated feature importance plot...")
    
    # Collect feature importances across all iterations
    feature_scores = {}
    
    for iteration_data in all_results:
        for result in iteration_data.get('results', []):
            importance = result.get('feature_importance', {})
            for gene, score in importance.items():
                if gene not in feature_scores:
                    feature_scores[gene] = []
                feature_scores[gene].append(score)
    
    # Calculate mean and std for each feature
    feature_stats = []
    for gene, scores in feature_scores.items():
        feature_stats.append({
            'Gene': gene,
            'Mean Importance': np.mean(scores),
            'Std': np.std(scores),
            'Count': len(scores)
        })
    
    df = pd.DataFrame(feature_stats)
    df = df.sort_values('Mean Importance', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize_single)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    
    bars = ax.barh(
        df['Gene'], 
        df['Mean Importance'], 
        xerr=df['Std'],
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        capsize=3
    )
    
    ax.set_xlabel('Mean Feature Importance Score')
    ax.set_ylabel('Gene Symbol')
    ax.set_title(f'Top {top_n} Features by Aggregated Importance\n(Mean ± SD across all iterations)')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, df['Mean Importance']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    output_path = config.output_dir / f'feature_importance_aggregated.{config.format}'
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"   ✓ Saved: {output_path.name}")
    return output_path


##### PERFORMANCE BOXPLOT #####
def plot_performance_boxplot(
    all_results: List[Dict],
    config: FigureConfig,
    logger: logging.Logger
) -> Path:
    """Create boxplot showing metric distributions across iterations."""
    logger.info("📦 Generating performance boxplot...")
    
    # Collect metrics for best configuration (2 groups based on earlier analysis)
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'AUC-ROC': []
    }
    
    for iteration_data in all_results:
        for result in iteration_data.get('results', []):
            # Focus on 2-group configuration for clarity
            if result.get('num_groups_used') == 2:
                metrics_data['Accuracy'].append(result.get('accuracy', 0))
                metrics_data['Precision'].append(result.get('precision', 0))
                metrics_data['Recall'].append(result.get('recall', 0))
                metrics_data['F1 Score'].append(result.get('f1_score', 0))
                metrics_data['AUC-ROC'].append(result.get('auc_roc', 0))
    
    df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=config.figsize_single)
    
    # Create boxplot with custom styling
    bp = ax.boxplot(
        [df[col].dropna() for col in df.columns],
        labels=df.columns,
        patch_artist=True,
        showmeans=True,
        meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 6}
    )
    
    # Color the boxes
    colors = plt.cm.Set2(np.linspace(0, 1, len(df.columns)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance Distribution\n(Best Configuration: 2 Groups, 8 Features)')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    
    # Add mean values as text
    for i, col in enumerate(df.columns):
        mean_val = df[col].mean()
        ax.text(i + 1, mean_val + 0.03, f'{mean_val:.3f}', 
                ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = config.output_dir / f'performance_boxplot.{config.format}'
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"   ✓ Saved: {output_path.name}")
    return output_path


##### AUC-ROC COMPARISON #####
def plot_auc_roc_comparison(
    all_results: List[Dict],
    config: FigureConfig,
    logger: logging.Logger
) -> Path:
    """Create bar chart comparing AUC-ROC across different group counts."""
    logger.info("📊 Generating AUC-ROC comparison plot...")
    
    # Collect AUC by group count
    auc_by_groups = {}
    
    for iteration_data in all_results:
        for result in iteration_data.get('results', []):
            n_groups = result.get('num_groups_used', 0)
            auc_val = result.get('auc_roc', 0)
            
            if n_groups not in auc_by_groups:
                auc_by_groups[n_groups] = []
            auc_by_groups[n_groups].append(auc_val)
    
    # Calculate statistics
    group_counts = sorted(auc_by_groups.keys())
    means = [np.mean(auc_by_groups[g]) for g in group_counts]
    stds = [np.std(auc_by_groups[g]) for g in group_counts]
    ci_lowers = [np.percentile(auc_by_groups[g], 2.5) for g in group_counts]
    ci_uppers = [np.percentile(auc_by_groups[g], 97.5) for g in group_counts]
    
    fig, ax = plt.subplots(figsize=config.figsize_single)
    
    x = np.arange(len(group_counts))
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(group_counts)))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Groups Used')
    ax.set_ylabel('AUC-ROC Score')
    ax.set_title('Classification Performance by Number of Groups\n(Mean ± SD)')
    ax.set_xticks(x)
    ax.set_xticklabels(group_counts)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='AUC = 0.8')
    
    # Add value labels
    for bar, mean_val, ci_l, ci_u in zip(bars, means, ci_lowers, ci_uppers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.3f}', ha='center', fontsize=9)
    
    # Add legend for CI
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    output_path = config.output_dir / f'auc_roc_by_groups.{config.format}'
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"   ✓ Saved: {output_path.name}")
    return output_path


##### CONFIDENCE INTERVAL FOREST PLOT #####
def plot_confidence_interval_forest(
    all_results: List[Dict],
    config: FigureConfig,
    logger: logging.Logger
) -> Path:
    """Create forest plot showing F1 scores with 95% CI for each iteration."""
    logger.info("🌲 Generating confidence interval forest plot...")
    
    # Collect F1 scores and CIs for 2-group configuration
    iterations = []
    f1_scores = []
    ci_lowers = []
    ci_uppers = []
    
    for i, iteration_data in enumerate(all_results):
        for result in iteration_data.get('results', []):
            if result.get('num_groups_used') == 2:
                iterations.append(f"Iter {i+1}")
                f1_scores.append(result.get('f1_score', 0))
                ci_lowers.append(result.get('f1_ci_lower', 0))
                ci_uppers.append(result.get('f1_ci_upper', 1))
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(iterations) * 0.4)))
    
    y_pos = np.arange(len(iterations))
    
    # Plot points with error bars
    ax.errorbar(
        f1_scores, y_pos,
        xerr=[np.array(f1_scores) - np.array(ci_lowers), 
              np.array(ci_uppers) - np.array(f1_scores)],
        fmt='o', color='steelblue', markersize=8,
        capsize=4, capthick=1.5, elinewidth=1.5
    )
    
    # Add vertical line for pooled mean
    pooled_mean = np.mean(f1_scores)
    ax.axvline(pooled_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Pooled Mean: {pooled_mean:.3f}')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(iterations)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score with 95% Confidence Intervals\n(Best Configuration: 2 Groups)')
    ax.set_xlim(0, 1.1)
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    output_path = config.output_dir / f'f1_confidence_intervals.{config.format}'
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"   ✓ Saved: {output_path.name}")
    return output_path


##### CROSS-VALIDATION STABILITY #####
def plot_cv_stability(
    all_results: List[Dict],
    config: FigureConfig,
    logger: logging.Logger
) -> Path:
    """Create plot showing CV mean vs test performance across iterations."""
    logger.info("📈 Generating cross-validation stability plot...")
    
    cv_means = []
    test_scores = []
    iterations = []
    
    for i, iteration_data in enumerate(all_results):
        for result in iteration_data.get('results', []):
            if result.get('num_groups_used') == 2:
                cv_means.append(result.get('cv_f1_mean', 0))
                test_scores.append(result.get('f1_score', 0))
                iterations.append(i + 1)
    
    fig, ax = plt.subplots(figsize=config.figsize_single)
    
    x = np.arange(len(iterations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv_means, width, label='CV F1 Mean', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test F1', color='coral', alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('F1 Score')
    ax.set_title('Cross-Validation vs Test Performance Stability')
    ax.set_xticks(x)
    ax.set_xticklabels(iterations)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add correlation coefficient
    correlation = np.corrcoef(cv_means, test_scores)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = config.output_dir / f'cv_test_stability.{config.format}'
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"   ✓ Saved: {output_path.name}")
    return output_path


##### GROUP PERFORMANCE HEATMAP #####
def plot_group_performance_heatmap(
    all_results: List[Dict],
    config: FigureConfig,
    logger: logging.Logger
) -> Path:
    """Create heatmap showing metrics across iterations and group counts."""
    logger.info("🔥 Generating group performance heatmap...")
    
    # Build matrix: rows = iterations, columns = group counts
    data = {}
    
    for i, iteration_data in enumerate(all_results):
        for result in iteration_data.get('results', []):
            n_groups = result.get('num_groups_used', 0)
            f1 = result.get('f1_score', 0)
            
            if n_groups not in data:
                data[n_groups] = {}
            data[n_groups][i + 1] = f1
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df = df.sort_index(axis=1)  # Sort by group count
    df.index.name = 'Iteration'
    df.columns.name = 'Groups'
    
    fig, ax = plt.subplots(figsize=config.figsize_wide)
    
    sns.heatmap(
        df, 
        annot=True, 
        fmt='.2f',
        cmap='RdYlGn',
        center=0.7,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'F1 Score'}
    )
    
    ax.set_title('F1 Score Heatmap: Iterations × Group Counts')
    ax.set_xlabel('Number of Groups')
    ax.set_ylabel('Iteration')
    
    plt.tight_layout()
    
    output_path = config.output_dir / f'performance_heatmap.{config.format}'
    fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"   ✓ Saved: {output_path.name}")
    return output_path


##### STANDALONE EXECUTION #####
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) < 2:
        print("Usage: python generate_figures.py <output_directory>")
        print("Example: python generate_figures.py output/gsm_2026_01_24-08_46_49")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    results_path = output_dir / "modeling_results_all_iterations.json"
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    figures = generate_all_figures(output_dir, results_path, logger)
    print(f"\n✅ Generated {len(figures)} figures")
    for name, path in figures.items():
        print(f"   - {name}: {path}")
