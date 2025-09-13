"""
🗂️ Result Saving Utility Module

Purpose:
    Save GSM pipeline results in both detailed and summary formats.
    Provides simple statistics in Excel for easy review.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass
import json
import shutil  # Added for file copying

from src.modeling.run_modeling import ModelingResult

@dataclass
class IterationMetadata:
    """Metadata for a single iteration of the modeling process."""
    iteration: int
    random_seed: int

def setup_save_directory(base_dir: Path, logger: logging.Logger) -> Path:
    """Create and verify the save directory."""
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created results directory: {base_dir}")
        return base_dir
    except Exception as e:
        logger.error(f"❌ Failed to create directory: {str(e)}")
        raise

def create_modeling_statistics_all_iterations_df(results: List[List[ModelingResult]]) -> pd.DataFrame:
    """Convert modeling results into a simple statistics DataFrame."""
    stats_data = []
    
    for iteration_idx, result_group in enumerate(results):
        for group_idx, result in enumerate(result_group):
            stats_data.append({
                'Iteration': iteration_idx + 1,
                'Group Count': len(result_group) - group_idx,
                'Feature Count': result.num_features_used,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
            })
    
    return pd.DataFrame(stats_data)

def create_modeling_statistics_averaged_df(results: List[List[ModelingResult]]) -> pd.DataFrame:
    """Calculate average statistics across all iterations for each group count.
    
    Args:
        results: Nested list of modeling results per iteration
    
    Returns:
        DataFrame with averaged metrics per group count
    """
    # First, collect all results by group count
    stats_by_group_count = {}
    
    for iteration_results in results:
        for group_idx, result in enumerate(iteration_results):
            group_count = len(iteration_results) - group_idx
            if group_count not in stats_by_group_count:
                stats_by_group_count[group_count] = []
            
            stats_by_group_count[group_count].append({
                'Feature Count': result.num_features_used,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
                'Training Time': result.training_time,
                # TODO: add auc support for results
                # 'AUC' : result.auc
            })
    
    # Calculate averages for each group count
    averaged_stats = []
    for group_count, stats_list in stats_by_group_count.items():
        avg_stats = {
            'Group Count': group_count,
            'Feature Count': sum(s['Feature Count'] for s in stats_list) / len(stats_list),
            'Accuracy': sum(s['Accuracy'] for s in stats_list) / len(stats_list),
            'Precision': sum(s['Precision'] for s in stats_list) / len(stats_list),
            'Recall': sum(s['Recall'] for s in stats_list) / len(stats_list),
            'F1 Score': sum(s['F1 Score'] for s in stats_list) / len(stats_list),
            # 'AUC': sum(s['AUC'] for s in stats_list) / len(stats_list),
            'Std Accuracy': np.std([s['Accuracy'] for s in stats_list]),
            'Std Precision': np.std([s['Precision'] for s in stats_list]),
            'Std Recall': np.std([s['Recall'] for s in stats_list]),
            'Std F1 Score': np.std([s['F1 Score'] for s in stats_list]),
            # 'Std AUC': np.std([s['AUC'] for s in stats_list])
        }
        averaged_stats.append(avg_stats)
    
    # Sort by group count descending
    averaged_stats.sort(key=lambda x: x['Group Count'], reverse=True)
    return pd.DataFrame(averaged_stats)

def adjust_column_width(worksheet):
    """Adjust column widths to fit content."""
    for column in worksheet.columns:
        max_length = 0
        column_letter = None
        
        # Get column letter safely, handling MergedCell objects
        for cell in column:
            if hasattr(cell, 'column_letter'):
                column_letter = cell.column_letter
                break
        
        if column_letter is None:
            continue  # Skip if we couldn't find a column letter
            
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column_letter].width = adjusted_width

def save_config_file(output_dir: Path, logger: logging.Logger) -> None:
    """
    Copy the config.py file to the results directory for reproducibility.
    
    Args:
        output_dir: Directory where results are saved
        logger: Logger instance
    """
    try:
        # Get the path to config.py (assumed to be two levels up from this file)
        config_path = Path(__file__).resolve().parents[1] / "config.py"
        
        if not config_path.exists():
            logger.warning(f"⚠️ Config file not found at {config_path}")
            return
            
        # Copy the file to the output directory
        shutil.copy(config_path, output_dir / "config.txt")
        logger.info(f"📄 Copied config.py to results directory: {output_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to copy config file: {str(e)}")

def save_modeling_results(
    results: List[List[ModelingResult]],
    iteration_metadata: List[IterationMetadata],
    output_dir: str,
    experiment_name: str,
    logger: logging.Logger
) -> None:
    """
    Save modeling results in both detailed and summary formats.
    
    Args:
        results: Nested list of modeling results
        iteration_metadata: List of metadata for each iteration
        output_dir: Directory to save results
        experiment_name: Name prefix for output files
        logger: Logger instance
    """
    output_path = Path(output_dir)
    setup_save_directory(output_path, logger)
    
    # Save the config file to the results directory for reproducibility
    save_config_file(output_path, logger)
    
    try:
        # Save detailed results
        for iteration_idx, result_group in enumerate(results):
            metadata = asdict(iteration_metadata[iteration_idx])
            
            # Create results dictionary with just the basic metadata
            results_dict = {
                'metadata': {
                    'iteration': metadata['iteration'],
                    'random_seed': metadata['random_seed']
                },
                'results': {}
            }
            
            # Add each group's results
            for idx, result in enumerate(result_group):
                group_key = f"group_count_{len(result_group)-idx}"
                results_dict['results'][group_key] = {
                    'ModelingResult': asdict(result),
                    # 'groups': result.groups,
                    # 'features': result.features
                }
            
            # Save to JSON file using json module instead of pandas
            output_file = output_path / f"{experiment_name}_iteration_{iteration_idx+1}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)

            logger.info(f"📊 Saved results for iteration {iteration_idx+1}")

        # Create and save statistics summary
        stats_df = create_modeling_statistics_all_iterations_df(results)
        avg_stats_df = create_modeling_statistics_averaged_df(results)
        excel_path = output_path / f"{experiment_name}_statistics.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary statistics for all iterations
            stats_df.to_excel(writer, sheet_name='All Iterations', index=False)
            adjust_column_width(writer.sheets['All Iterations'])
            
            # Averaged statistics across iterations
            avg_stats_df.to_excel(writer, sheet_name='Averaged Results', index=False)
            adjust_column_width(writer.sheets['Averaged Results'])
            
            # Aggregated statistics by iteration
            agg_by_iteration = stats_df.groupby('Iteration').agg({
                'Accuracy': ['mean', 'std'],
                'Precision': ['mean', 'std'],
                'Recall': ['mean', 'std']
            }).round(4)
            agg_by_iteration.to_excel(writer, sheet_name='By Iteration')
            adjust_column_width(writer.sheets['By Iteration'])
        
        logger.info(f"📊 Saved statistics summary to {excel_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {str(e)}")
        raise
