"""
🗂️ Result Saving Utility Module

Purpose:
    Save GSM pipeline results in both detailed and summary formats.
    Provides simple statistics in Excel for easy review.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
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
    logger: logging.Logger,
    aggregated_groups: Optional[List[Dict]] = None,
    aggregated_features: Optional[List[Dict]] = None,
    robust_rank_groups: Optional[List[Dict]] = None,
    robust_rank_features: Optional[List[Dict]] = None
) -> None:
    """
    Identify the best performing model configuration and save a comprehensive summary report.
    
    This report addresses reviewer concerns by including:
    - Detailed methodology description
    - Statistical validation metrics (AUC-ROC, confidence intervals)
    - Cross-validation results with standard deviations
    - Probability prediction thresholds
    - Best averaged groups and features across all iterations
    - Robust rank aggregation results
    
    Args:
        results: List of lists of ModelingResult objects
        iteration_metadata: Metadata for each iteration
        output_dir: Directory to save the report
        logger: Logger instance
        aggregated_groups: Optional list of best averaged groups
        aggregated_features: Optional list of best averaged features
                                f.write("ranked_groups/iter_XXX_groups.csv\n")
                                f.write("    Per-iteration group rankings (CSV files, one per iteration).\n")
                                f.write("    Columns: Rank, Group Name, Accuracy, F1 Score, Iteration\n\n")
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
            
            # Best Averaged Groups (across all iterations)
            if aggregated_groups:
                f.write("=" * 70 + "\n")
                f.write("BEST AVERAGED GROUPS (across all iterations)\n")
                f.write("-" * 70 + "\n")
                f.write("Groups ranked by average F1 score when used in modeling:\n\n")
                for idx, group in enumerate(aggregated_groups[:15], 1):
                    avg_f1 = group.get('Average F1 Score', 0.0)
                    std_f1 = group.get('Std F1 Score', 0.0)
                    occurrences = group.get('Occurrences', 0)
                    f.write(f"  {idx:2d}. {group['Group Name'][:50]:<50} "
                            f"Avg F1: {avg_f1:.4f} ± {std_f1:.4f} (n={occurrences})\n")
                f.write("\n")
            
            # Best Averaged Features (across all iterations)
            if aggregated_features:
                f.write("BEST AVERAGED FEATURES (across all iterations)\n")
                f.write("-" * 70 + "\n")
                f.write("Features ranked by average importance score:\n\n")
                for idx, feature in enumerate(aggregated_features[:20], 1):
                    avg_imp = feature.get('Average Importance', 0.0)
                    std_imp = feature.get('Std Importance', 0.0)
                    occurrences = feature.get('Occurrences', 0)
                    f.write(f"  {idx:2d}. {feature['Feature Name']:<20} "
                            f"Avg Importance: {avg_imp:.4f} ± {std_imp:.4f} (n={occurrences})\n")
                f.write("\n")
            
            # Robust Rank Aggregation Results - Groups
            if robust_rank_groups:
                f.write("=" * 70 + "\n")
                f.write("ROBUST RANK AGGREGATION - GROUPS\n")
                f.write("-" * 70 + "\n")
                f.write("Groups ranked using Robust Rank Aggregation (RRA) method.\n")
                f.write("Lower p-value = more consistently top-ranked across iterations.\n\n")
                f.write(f"{'Rank':<6}{'Group Name':<45}{'RRA Score':>12}{'Avg Rank':>10}{'N':>6}\n")
                f.write("-" * 70 + "\n")
                for idx, group in enumerate(robust_rank_groups[:15], 1):
                    name = group.get('Group Name', group.get('group_name', ''))[:42]
                    score = group.get('Aggregated Score', group.get('aggregated_score', 0.0))
                    avg_rank = group.get('Average Rank', group.get('average_rank', 0.0))
                    occ = group.get('Occurrences', group.get('occurrences', 0))
                    f.write(f"  {idx:<4}{name:<45}{score:>10.4f}{avg_rank:>10.2f}{occ:>6}\n")
                f.write("\n")
            
            # Robust Rank Aggregation Results - Features
            if robust_rank_features:
                f.write("ROBUST RANK AGGREGATION - FEATURES\n")
                f.write("-" * 70 + "\n")
                f.write("Features ranked using Robust Rank Aggregation (RRA) method.\n")
                f.write("Higher RRA score = more consistently top-ranked.\n\n")
                f.write(f"{'Rank':<6}{'Feature Name':<25}{'RRA Score':>12}{'Avg Rank':>10}{'Avg Imp':>10}{'N':>6}\n")
                f.write("-" * 70 + "\n")
                for idx, feature in enumerate(robust_rank_features[:25], 1):
                    name = feature.get('Feature Name', feature.get('feature_name', ''))[:22]
                    score = feature.get('Aggregated Score', feature.get('aggregated_score', 0.0))
                    avg_rank = feature.get('Average Rank', feature.get('average_rank', 0.0))
                    avg_imp = feature.get('Average Importance', feature.get('average_importance', 0.0))
                    occ = feature.get('Occurrences', feature.get('occurrences', 0))
                    f.write(f"  {idx:<4}{name:<25}{score:>10.4f}{avg_rank:>10.2f}{avg_imp:>10.4f}{occ:>6}\n")
                f.write("\n")
            
            # Summary statistics section
            f.write("=" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 70 + "\n")
            
            # Collect all F1 scores
            all_f1_scores = []
            all_auc_scores = []
            for iteration_results in results:
                for res in iteration_results:
                    all_f1_scores.append(res.f1_score)
                    all_auc_scores.append(getattr(res, 'auc_roc', 0.0))
            
            if all_f1_scores:
                f.write(f"Total Configurations Tested: {len(all_f1_scores)}\n")
                f.write(f"F1 Score Range: {min(all_f1_scores):.4f} - {max(all_f1_scores):.4f}\n")
                f.write(f"F1 Score Mean ± Std: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}\n")
                if all_auc_scores:
                    f.write(f"AUC-ROC Mean ± Std: {np.mean(all_auc_scores):.4f} ± {np.std(all_auc_scores):.4f}\n")
            
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("📊 Full results saved in accompanying Excel files.\n")
            f.write("📈 Visualizations saved in figures/ directory.\n")
            f.write("=" * 70 + "\n")
                
        logger.info(f"✅ Summary report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save summary report: {str(e)}")


def save_output_readme(output_dir: Path, logger: logging.Logger) -> None:
    """
    Save a README file explaining all output files in the results folder.
    
    Args:
        output_dir: Path to the output directory
        logger: Logger instance
    """
    readme_path = output_dir / "README_OUTPUT_FILES.txt"
    
    try:
        with open(readme_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GSM PIPELINE OUTPUT FILES GUIDE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("This folder contains the results of a GSM (Grouping-Scoring-Modeling) pipeline run.\n")
            f.write("Below is an explanation of each output file.\n\n")
            
            # Core Results Section
            f.write("-" * 80 + "\n")
            f.write("📊 CORE RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("summary_report.txt\n")
            f.write("    Human-readable summary of the best results.\n")
            f.write("    START HERE to see the best performing configuration.\n\n")
            
            f.write("modeling_results_all_iterations.json\n")
            f.write("    Complete JSON file with all modeling results across all iterations.\n")
            f.write("    Contains: F1, AUC, accuracy, precision, recall, feature importance,\n")
            f.write("    used groups, used features, confidence intervals, and more.\n\n")
            
            f.write("modeling_results_statistics.xlsx\n")
            f.write("    Excel file with performance metrics for each group count configuration.\n")
            f.write("    Columns: Iteration, Groups Used, Features Used, F1, AUC, Accuracy, etc.\n\n")
            
            # Group Rankings Section
            f.write("-" * 80 + "\n")
            f.write("📁 GROUP RANKINGS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("ranked_groups_all_iterations.xlsx\n")
            f.write("    Group rankings for each iteration, sorted by F1 score (descending).\n")
            f.write("    Groups that classify samples better are ranked higher.\n\n")
            
            f.write("aggregated_group_ranking_rra.xlsx  ⭐ RECOMMENDED\n")
            f.write("    Robust Rank Aggregation (RRA) of group rankings across all iterations.\n")
            f.write("    Groups consistently ranked high get low p-values.\n")
            f.write("    Columns:\n")
            f.write("      - Rank: Final aggregated rank (1 = best)\n")
            f.write("      - Aggregated P-Value: Lower = more consistently high-ranked\n")
            f.write("      - Aggregated Score: -log10(p-value), higher = better\n")
            f.write("      - Average Rank: Mean rank across iterations\n")
            f.write("      - Occurrences: How many iterations the group appeared in\n\n")
            
            f.write("best_averaged_groups.xlsx\n")
            f.write("    Simple average ranking of groups across iterations.\n\n")
            
            # Feature Rankings Section
            f.write("-" * 80 + "\n")
            f.write("🧬 FEATURE RANKINGS (Two Methods)\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("The pipeline provides TWO different ways to rank features:\n\n")
            
            f.write("METHOD 1: Individual Feature Scoring (ML-based)\n")
            f.write("    Features ranked by their own machine learning importance.\n\n")
            
            f.write("    ranked_features_individual/iteration_XXX.xlsx\n")
            f.write("        Per-iteration individual feature scores. One file per iteration.\n")
            f.write("        Columns: feature_name, f1_score, importance_score, mutual_info\n\n")
            
            f.write("    aggregated_feature_ranking_individual_rra.xlsx\n")
            f.write("        RRA aggregation of individual feature rankings.\n\n")
            
            f.write("METHOD 2: Group-Derived Feature Scoring ⭐ RECOMMENDED\n")
            f.write("    Features ranked by their BEST GROUP's F1 score.\n")
            f.write("    This aligns with GSM methodology where groups are the key unit.\n\n")
            
            f.write("    ranked_features_group_derived/iteration_XXX.xlsx\n")
            f.write("        Per-iteration group-derived feature scores. One file per iteration.\n")
            f.write("        Columns: feature_name, best_group_name, group_f1_score, group_rank\n\n")
            
            f.write("    aggregated_feature_ranking_group_derived_rra.xlsx  ⭐ USE THIS\n")
            f.write("        RRA aggregation of group-derived feature rankings.\n")
            f.write("        Features from consistently top-performing groups rank highest.\n")
            f.write("        Columns:\n")
            f.write("          - Rank: Final aggregated rank (1 = best)\n")
            f.write("          - Most Common Group: Group this feature was most often in\n")
            f.write("          - Average Group F1: Mean F1 of the feature's best groups\n")
            f.write("          - Aggregated P-Value: RRA p-value (lower = better)\n\n")
            
            f.write("best_averaged_features.xlsx\n")
            f.write("    Simple average feature ranking.\n\n")
            
            # Figures Section
            f.write("-" * 80 + "\n")
            f.write("📈 FIGURES (in figures/ subfolder)\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Performance Visualizations:\n")
            f.write("    performance_by_groups_boxplot.png - F1/AUC distribution by group count\n")
            f.write("    performance_comparison_heatmap.png - Metrics heatmap across groups\n")
            f.write("    confidence_intervals_comparison.png - CI comparison plot\n")
            f.write("    group_count_optimization.png - Optimal group count analysis\n\n")
            
            f.write("Feature Analysis:\n")
            f.write("    feature_importance_*.png - Top feature importance plots\n")
            f.write("    feature_frequency_*.png - Feature selection frequency\n\n")
            
            f.write("Iteration Analysis:\n")
            f.write("    iteration_performance_trend.png - Performance across iterations\n")
            f.write("    iteration_variability.png - Stability analysis\n\n")
            
            f.write("Statistics Excel Files (in figures/):\n")
            f.write("    statistics_performance_by_groups.xlsx - Metrics by group count\n")
            f.write("    statistics_feature_importance.xlsx - Feature importance data\n")
            f.write("    statistics_iteration_summary.xlsx - Per-iteration stats\n\n")
            
            # Configuration Section
            f.write("-" * 80 + "\n")
            f.write("⚙️ CONFIGURATION & LOGS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("run_parameters.txt\n")
            f.write("    The ACTUAL parameters used for this specific run.\n")
            f.write("    Shows input files, iterations, model, normalization, etc.\n\n")
            
            # config_used.py removed; run_parameters.txt is the authoritative record
            
            f.write("gsm_workflow.log\n")
            f.write("    Detailed log of the entire pipeline execution.\n")
            f.write("    Useful for debugging or understanding the process.\n\n")
            
            # Biological Validation Section
            f.write("-" * 80 + "\n")
            f.write("🧬 BIOLOGICAL VALIDATION (if enabled)\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("biological_validation/ subfolder contains:\n\n")
            
            f.write("    README_BIOLOGICAL_VALIDATION.txt\n")
            f.write("        Detailed explanation of biological validation results.\n\n")
            
            f.write("    enrichr_results.xlsx\n")
            f.write("        Pathway enrichment analysis (KEGG, GO, Reactome, WikiPathways).\n")
            f.write("        Shows which biological pathways your top genes are involved in.\n\n")
            
            f.write("    string_interactions.xlsx\n")
            f.write("        Protein-protein interaction network from STRING-db.\n\n")
            
            f.write("    disgenet_results.xlsx (if API key provided)\n")
            f.write("        Gene-disease associations from DisGeNET.\n\n")
            
            f.write("    validation_summary.txt\n")
            f.write("        Summary of all validation results.\n\n")
            
            f.write("    group_validation_summary.xlsx\n")
            f.write("        Validation results for top-ranked groups.\n\n")
            
            # Quick Start Section
            f.write("-" * 80 + "\n")
            f.write("🚀 QUICK START: WHERE TO LOOK FIRST\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("1. summary_report.txt\n")
            f.write("   → See best F1 score, best configuration, top groups & features\n\n")
            
            f.write("2. aggregated_group_ranking_rra.xlsx\n")
            f.write("   → Your most important biological groups/pathways\n\n")
            
            f.write("3. aggregated_feature_ranking_group_derived_rra.xlsx\n")
            f.write("   → Your most important genes (ranked by group performance)\n\n")
            
            f.write("4. figures/group_count_optimization.png\n")
            f.write("   → How many groups to use for best performance\n\n")
            
            f.write("5. biological_validation/ (if enabled)\n")
            f.write("   → External database validation of your findings\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("For more details, see DOCS/feature_ranking_methods_explanation.txt\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"📄 Saved output README to: {readme_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save output README: {str(e)}")
