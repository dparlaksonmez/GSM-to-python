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


@dataclass
class IterationResultsPayload:
    """Serializable payload for a single iteration of results."""
    metadata: IterationMetadata
    results: List[ModelingResult]

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
        for result in result_group:
            stats_data.append({
                'Iteration': iteration_idx + 1,
                'Group Count': result.num_groups_used,
                'Feature Count': result.num_features_used,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
                'AUC-ROC': getattr(result, 'auc_roc', 0.0),
                'F1 95% CI Lower': getattr(result, 'f1_ci_lower', 0.0),
                'F1 95% CI Upper': getattr(result, 'f1_ci_upper', 0.0),
                'CV F1 Mean': getattr(result, 'cv_f1_mean', 0.0),
                'CV F1 Std': getattr(result, 'cv_f1_std', 0.0),
                'Mean Positive Probability': getattr(result, 'mean_positive_probability', 0.0),
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
        for result in iteration_results:
            group_count = result.num_groups_used
            if group_count not in stats_by_group_count:
                stats_by_group_count[group_count] = []

            stats_by_group_count[group_count].append({
                'Feature Count': result.num_features_used,
                'Accuracy': result.accuracy,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1 Score': result.f1_score,
                'Training Time': result.training_time,
                'AUC-ROC': getattr(result, 'auc_roc', 0.0),
                'CV F1 Mean': getattr(result, 'cv_f1_mean', 0.0),
                'CV F1 Std': getattr(result, 'cv_f1_std', 0.0),
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
            'AUC-ROC': sum(s['AUC-ROC'] for s in stats_list) / len(stats_list),
            'CV F1 Mean': sum(s['CV F1 Mean'] for s in stats_list) / len(stats_list),
            'Std Accuracy': np.std([s['Accuracy'] for s in stats_list]),
            'Std Precision': np.std([s['Precision'] for s in stats_list]),
            'Std Recall': np.std([s['Recall'] for s in stats_list]),
            'Std F1 Score': np.std([s['F1 Score'] for s in stats_list]),
            'Std AUC-ROC': np.std([s['AUC-ROC'] for s in stats_list])
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
    logger: logging.Logger,
    save_visualizations: bool = True
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
    
    # Note: Configuration is defined in GSM_workflow_config.py
    # No need to copy a separate config.py file
    
    try:
        # Save detailed results in a single combined file
        payloads = []
        for iteration_idx, result_group in enumerate(results):
            payloads.append(
                IterationResultsPayload(
                    metadata=iteration_metadata[iteration_idx],
                    results=result_group
                )
            )

        output_file = output_path / f"{experiment_name}_all_iterations.json"
        with open(output_file, 'w') as f:
            json.dump([asdict(payload) for payload in payloads], f, indent=2)

        logger.info(f"📊 Saved combined results to {output_file}")

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

def save_summary_report(
    results: List[List[ModelingResult]], 
    iteration_metadata: List[IterationMetadata], 
    output_dir: Path, 
    logger: logging.Logger
) -> None:
    """
    Identify the best performing model configuration and save a comprehensive summary report.
    
    This report addresses reviewer concerns by including:
    - Detailed methodology description
    - Statistical validation metrics (AUC-ROC, confidence intervals)
    - Cross-validation results with standard deviations
    - Probability prediction thresholds
    
    Args:
        results: List of lists of ModelingResult objects
        iteration_metadata: Metadata for each iteration
        output_dir: Directory to save the report
        logger: Logger instance
    """
    best_f1 = -1.0
    best_result = None
    best_iteration_meta = None
    
    # Find best result
    for i, iteration_results in enumerate(results):
        meta = iteration_metadata[i]
        for res in iteration_results:
            if res.f1_score > best_f1:
                best_f1 = res.f1_score
                best_result = res
                best_iteration_meta = meta
                
    if best_result is None or best_iteration_meta is None:
        logger.warning("⚠️ No results found to generate summary report.")
        return

    report_path = output_dir / "summary_report.txt"
    
    try:
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("🏆 GSM Pipeline Summary Report 🏆\n")
            f.write("Grouping-Scoring-Modeling (G-S-M) Gene Analysis Pipeline\n")
            f.write("=" * 70 + "\n\n")
            
            # Methodology Section (addressing reviewer concerns)
            f.write("METHODOLOGY\n")
            f.write("-" * 70 + "\n")
            f.write("This pipeline implements a rigorous gene group-based classification approach:\n\n")
            f.write("1. FEATURE FILTERING:\n")
            f.write("   - Welch's t-test for differential expression analysis\n")
            f.write("   - Multiple comparison correction: Benjamini-Hochberg FDR\n")
            f.write("   - Significance threshold: α = 0.05 (adjusted p-values)\n\n")
            f.write("2. GENE GROUPING:\n")
            f.write("   - Pre-existing knowledge-based grouping (e.g., DisGeNET pathways)\n")
            f.write("   - Groups ranked by embedded feature selection via ML models\n\n")
            f.write("3. MODEL TRAINING:\n")
            f.write("   - RandomForest: Ensemble method with embedded feature importance\n")
            f.write("   - SVM: Support vector classification with probability estimates\n")
            f.write("   - Rationale: These methods handle high-dimensional data well and\n")
            f.write("     provide feature importance for biological interpretation.\n\n")
            f.write("4. VALIDATION:\n")
            f.write("   - Stratified K-fold cross-validation for robust estimates\n")
            f.write("   - Bootstrap confidence intervals (1000 samples, 95% CI)\n")
            f.write("   - AUC-ROC for threshold-independent classification quality\n")
            f.write("   - Probability predictions for risk stratification\n\n")
            
            f.write("BEST PERFORMANCE ACHIEVED\n")
            f.write("-" * 70 + "\n")
            f.write(f"F1 Score:       {best_result.f1_score:.4f}")
            f1_ci_lower = getattr(best_result, 'f1_ci_lower', 0.0)
            f1_ci_upper = getattr(best_result, 'f1_ci_upper', 0.0)
            if f1_ci_lower > 0:
                f.write(f" (95% CI: {f1_ci_lower:.4f} - {f1_ci_upper:.4f})")
            f.write("\n")
            f.write(f"Accuracy:       {best_result.accuracy:.4f}\n")
            f.write(f"Precision:      {best_result.precision:.4f}\n")
            f.write(f"Recall:         {best_result.recall:.4f}\n")
            auc_roc = getattr(best_result, 'auc_roc', 0.0)
            if auc_roc > 0:
                auc_ci_lower = getattr(best_result, 'auc_ci_lower', 0.0)
                auc_ci_upper = getattr(best_result, 'auc_ci_upper', 0.0)
                f.write(f"AUC-ROC:        {auc_roc:.4f}")
                if auc_ci_lower > 0:
                    f.write(f" (95% CI: {auc_ci_lower:.4f} - {auc_ci_upper:.4f})")
                f.write("\n")
            
            # Cross-validation results
            cv_f1_mean = getattr(best_result, 'cv_f1_mean', 0.0)
            cv_f1_std = getattr(best_result, 'cv_f1_std', 0.0)
            if cv_f1_mean > 0:
                f.write(f"\nCross-Validation (5-fold):\n")
                f.write(f"  CV F1 Mean:   {cv_f1_mean:.4f} ± {cv_f1_std:.4f}\n")
            
            # Probability predictions
            mean_prob = getattr(best_result, 'mean_positive_probability', 0.0)
            if mean_prob > 0:
                f.write(f"\nProbability Predictions:\n")
                f.write(f"  Mean P(positive): {mean_prob:.4f}\n")
                f.write(f"  Note: Threshold can be adjusted based on clinical requirements\n")
                f.write(f"        (e.g., 0.3 for high sensitivity, 0.7 for high specificity)\n")
            f.write("\n")
            
            f.write("CONFIGURATION DETAILS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Iteration:      {best_iteration_meta.iteration}\n")
            f.write(f"Random Seed:    {best_iteration_meta.random_seed}\n")
            f.write(f"Groups Used:    {best_result.num_groups_used}\n")
            f.write(f"Features Used:  {best_result.num_features_used}\n")
            f.write(f"Model Name:     {best_result.model_name}\n\n")

            f.write("TOP CONFIGURATIONS (Top 5 by F1)\n")
            f.write("-" * 70 + "\n")
            ranked_results = []
            for i, iteration_results in enumerate(results):
                meta = iteration_metadata[i]
                for res in iteration_results:
                    ranked_results.append({
                        "Iteration": meta.iteration,
                        "Random Seed": meta.random_seed,
                        "Groups Used": res.num_groups_used,
                        "Features Used": res.num_features_used,
                        "Model Name": res.model_name,
                        "F1 Score": res.f1_score,
                        "Accuracy": res.accuracy,
                        "Precision": res.precision,
                        "Recall": res.recall,
                        "AUC-ROC": getattr(res, 'auc_roc', 0.0)
                    })
            ranked_results.sort(key=lambda x: x["F1 Score"], reverse=True)
            for rank, entry in enumerate(ranked_results[:5], 1):
                f.write(
                    f"{rank}. Iter {entry['Iteration']} | Groups {entry['Groups Used']} | "
                    f"F1 {entry['F1 Score']:.4f} | AUC {entry['AUC-ROC']:.4f} | "
                    f"Acc {entry['Accuracy']:.4f}\n"
                )
            f.write("\n")

            f.write("PER-ITERATION BEST CONFIGURATIONS\n")
            f.write("-" * 70 + "\n")
            for i, iteration_results in enumerate(results):
                meta = iteration_metadata[i]
                if not iteration_results:
                    continue
                best_iter = max(iteration_results, key=lambda r: r.f1_score)
                auc = getattr(best_iter, 'auc_roc', 0.0)
                f.write(
                    f"Iter {meta.iteration:2d} | Groups {best_iter.num_groups_used:2d} | "
                    f"Features {best_iter.num_features_used:3d} | F1 {best_iter.f1_score:.4f} | "
                    f"AUC {auc:.4f} | Acc {best_iter.accuracy:.4f}\n"
                )
            f.write("\n")
            
            f.write("TOP FEATURES (by importance)\n")
            f.write("-" * 70 + "\n")
            # Sort features by importance if available
            sorted_features = sorted(
                best_result.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            if sorted_features:
                for feature, importance in sorted_features[:10]:  # Top 10 features
                    f.write(f"  {feature}: {importance:.4f}\n")
            else:
                f.write("  Feature importance not available for this model.\n")
            f.write("\n")
            
            # Used groups
            if best_result.used_groups:
                f.write("TOP GROUPS USED\n")
                f.write("-" * 70 + "\n")
                for idx, group in enumerate(best_result.used_groups[:10], 1):
                    f.write(f"  {idx}. {group}\n")
                f.write("\n")
                
        logger.info(f"✅ Summary report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save summary report: {str(e)}")
