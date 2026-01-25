# TODO: Implement unit tests for gsm_run and gsm_main_loop functions
# TODO: Add support for additional ML models in the modeling stage

"""
🧬 GSM_pipeline.py - Main Pipeline Implementation for Gene Analysis

Purpose:
    Core implementation of the Grouping-Scoring-Modeling (GSM) pipeline for gene analysis.
    This pipeline processes gene expression data through multiple stages to identify
    significant gene groups and build predictive models.

Key Components:
    1. Data Preprocessing: Normalizes and validates input data
    2. Grouping: Groups genes based on biological relationships
    3. Scoring: Evaluates gene groups using ML metrics
    4. Modeling: Trains and validates ML models on selected groups

Key Functions:
    📊 gsm_run: Main entry point - orchestrates the complete pipeline execution
    🔄 gsm_main_loop: Core processing loop implementing the GSM workflow stages

Usage Example:
    >>> # Load input data
    >>> expression_data = pd.read_csv("expression_data.csv")
    >>> group_data = pd.read_csv("group_data.csv")
    >>> logger = setup_logger()
    >>> 
    >>> # Configure and run pipeline
    >>> config = GSMConfig(sample_ratio=0.8, n_iteration_workflow=5)
    >>> gsm_run(expression_data, group_data, logger, config)

Notes:
    - Ensures reproducibility through fixed random seeds
    - Implements comprehensive error handling and logging
    - Supports both notebook and script execution modes
"""

##### Imports #####
import sys
from pathlib import Path
import logging
import shutil

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.workflows.GSM_workflow_config import (INPUT_EXPRESSION_DATA, INPUT_GROUP_DATA, OUTPUT_DIR, 
                        RANDOM_SEED, CROSS_VALIDATION_FOLDS, NUMBER_OF_ITERATIONS,
                        GENE_COLUMN_NAME, GROUP_COLUMN_NAME, INITIAL_FEATURE_FILTER_SIZE,
                        TTEST_THRESHOLD,
                        SAVE_INTERMEDIATE_RESULTS, TRAIN_TEST_SPLIT_RATIO, MODEL_NAME, LABEL_COLUMN_NAME, NORMALIZATION_METHOD,
                        CLASS_LABELS_NEGATIVE, CLASS_LABELS_POSITIVE, BEST_GROUPS_TO_KEEP,
                        MAIN_DATA_FILE_SEPARATOR,
                        GROUPING_FILE_SEPARATOR)

# Import the config module itself (not only constants) so we can log where it
# was loaded from at runtime. This is critical for debugging “wrong config file
# / stale notebook kernel / wrong checkout” issues.
import src.workflows.GSM_workflow_config as gsm_workflow_config

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import random
import time
from datetime import datetime, timedelta

# Updated imports to use correct module paths
from src.grouping.run_grouping import run_grouping
from src.scoring.run_scoring import run_scoring, score_all_features
from src.scoring.feature_scorer import FeatureScore
from src.modeling.run_modeling import ModelingResult, run_modeling, select_features_from_top_groups
from src.data_processing.data_loader import load_input_file, load_group_file
from src.data_processing.data_preprocess import preprocess_data, preprocess_grouping_data
from src.data_processing.train_test_splitter import split_data
from src.data_processing.preliminary_filtering import preliminary_ttest_filter
from src.utils import save_results
from src.utils.visualization import visualize_f1_scores
from src.utils.logger import setup_logger  # Add this import at the top with other imports
from src.utils.generate_figures import generate_all_figures
from src.utils.rank_aggregation import (
    load_ranked_groups_from_excel,
    load_ranked_features_from_excel,
    aggregate_group_ranks_rra,
    aggregate_feature_ranks_rra,
    save_aggregated_group_ranking,
    save_aggregated_feature_ranking,
    compute_best_averaged_groups,
    compute_best_averaged_features,
    save_best_averaged_rankings
)
import time


##### Data Structures #####
@dataclass
class IterationResult:
    """Track results and metadata for each GSM iteration."""
    iteration: int
    random_seed: int
    modeling_results: List[ModelingResult]


@dataclass
class ConfigLogItem:
    """Single config item for log output."""
    name: str
    value: str


@dataclass
class FeatureCountResult:
    """Feature count result for a specific top-N group selection."""
    top_group_count: int
    unique_feature_count: int


@dataclass
class AdjustedGroupSelection:
    """Final group selection after enforcing feature diversity."""
    requested_top_groups: int
    used_top_groups: int
    requested_feature_count: int
    used_feature_count: int
    baseline_feature_count: int


def copy_used_config_file(*, output_dir: Path, logger) -> None:
    """Copy the exact config module file used at runtime into the output folder."""
    try:
        config_source_path = Path(gsm_workflow_config.__file__).resolve()
        config_dest_path = output_dir / "config_used.py"
        shutil.copy2(config_source_path, config_dest_path)
        logger.info(f"Copied used config file to: {config_dest_path}")
    except Exception as exc:
        logger.warning(f"Could not copy used config file: {exc}")


def build_config_log_items() -> List[ConfigLogItem]:
    """Build the list of important config variables for logging."""
    return [
        ConfigLogItem("INPUT_EXPRESSION_DATA", str(gsm_workflow_config.INPUT_EXPRESSION_DATA)),
        ConfigLogItem("INPUT_GROUP_DATA", str(gsm_workflow_config.INPUT_GROUP_DATA)),
        ConfigLogItem("MAIN_DATA_FILE_SEPARATOR", str(gsm_workflow_config.MAIN_DATA_FILE_SEPARATOR)),
        ConfigLogItem("GROUPING_FILE_SEPARATOR", str(gsm_workflow_config.GROUPING_FILE_SEPARATOR)),
        ConfigLogItem("OUTPUT_DIR", str(gsm_workflow_config.OUTPUT_DIR)),
        ConfigLogItem("NUMBER_OF_ITERATIONS", str(gsm_workflow_config.NUMBER_OF_ITERATIONS)),
        ConfigLogItem("TRAIN_TEST_SPLIT_RATIO", str(gsm_workflow_config.TRAIN_TEST_SPLIT_RATIO)),
        ConfigLogItem("MODEL_NAME", str(gsm_workflow_config.MODEL_NAME)),
        ConfigLogItem("LABEL_COLUMN_NAME", str(gsm_workflow_config.LABEL_COLUMN_NAME)),
        ConfigLogItem("NORMALIZATION_METHOD", str(gsm_workflow_config.NORMALIZATION_METHOD)),
        ConfigLogItem("CLASS_LABELS_POSITIVE", str(gsm_workflow_config.CLASS_LABELS_POSITIVE)),
        ConfigLogItem("CLASS_LABELS_NEGATIVE", str(gsm_workflow_config.CLASS_LABELS_NEGATIVE)),
        ConfigLogItem("RANDOM_SEED", str(gsm_workflow_config.RANDOM_SEED)),
        ConfigLogItem("CROSS_VALIDATION_FOLDS", str(gsm_workflow_config.CROSS_VALIDATION_FOLDS)),
        ConfigLogItem("INITIAL_FEATURE_FILTER_SIZE", str(gsm_workflow_config.INITIAL_FEATURE_FILTER_SIZE)),
        ConfigLogItem("TTEST_THRESHOLD", str(gsm_workflow_config.TTEST_THRESHOLD)),
        ConfigLogItem("BEST_GROUPS_TO_KEEP", str(gsm_workflow_config.BEST_GROUPS_TO_KEEP)),
        ConfigLogItem("SAVE_INTERMEDIATE_RESULTS", str(gsm_workflow_config.SAVE_INTERMEDIATE_RESULTS)),
    ]


def log_config_values(*, logger) -> None:
    """Log important configuration values with uppercase keys."""
    logger.info("##### CONFIGURATION (IMPORTANT VARIABLES) #####")
    for item in build_config_log_items():
        logger.info(f"{item.name}={item.value}")


def count_unique_features_for_top_groups(
    *,
    group_ranks,
    group_feature_mapping,
    top_n_groups: int,
    data_columns,
    logger
) -> FeatureCountResult:
    """Count valid unique features for the top-N groups."""
    feature_result = select_features_from_top_groups(
        group_ranks,
        group_feature_mapping,
        top_n_groups,
        logger
    )
    available = [f for f in feature_result.selected_features if f in data_columns]
    return FeatureCountResult(top_group_count=top_n_groups, unique_feature_count=len(available))


def expand_groups_until_feature_increase(
    *,
    requested_top_groups: int,
    group_ranks,
    group_feature_mapping,
    data_columns,
    baseline_feature_count: int,
    logger
) -> AdjustedGroupSelection:
    """Increase top-N groups until features increase vs. the baseline or groups are exhausted."""
    max_groups = len(group_ranks)
    base = count_unique_features_for_top_groups(
        group_ranks=group_ranks,
        group_feature_mapping=group_feature_mapping,
        top_n_groups=min(requested_top_groups, max_groups),
        data_columns=data_columns,
        logger=logger
    )
    if base.unique_feature_count > baseline_feature_count:
        return AdjustedGroupSelection(
            requested_top_groups=base.top_group_count,
            used_top_groups=base.top_group_count,
            requested_feature_count=base.unique_feature_count,
            used_feature_count=base.unique_feature_count,
            baseline_feature_count=baseline_feature_count
        )
    current = base
    while current.top_group_count < max_groups:
        candidate = count_unique_features_for_top_groups(
            group_ranks=group_ranks,
            group_feature_mapping=group_feature_mapping,
            top_n_groups=current.top_group_count + 1,
            data_columns=data_columns,
            logger=logger
        )
        if candidate.unique_feature_count > baseline_feature_count:
            current = candidate
            break
        current = candidate
    return AdjustedGroupSelection(
        requested_top_groups=base.top_group_count,
        used_top_groups=current.top_group_count,
        requested_feature_count=base.unique_feature_count,
        used_feature_count=current.unique_feature_count,
        baseline_feature_count=baseline_feature_count
    )

def gsm_run(
    input_data: pd.DataFrame,
    group_data: pd.DataFrame,
    *,  # Force named parameters
    sample_ratio: float = TRAIN_TEST_SPLIT_RATIO,
    n_iterations: int = NUMBER_OF_ITERATIONS,
    model_name: str = MODEL_NAME,
    label_column: str = LABEL_COLUMN_NAME,
    positive_class_label: str = CLASS_LABELS_POSITIVE,
    negative_class_label: str = CLASS_LABELS_NEGATIVE,
    gene_column: str = GENE_COLUMN_NAME,
    group_column: str = GROUP_COLUMN_NAME,
    normalization_method: str = NORMALIZATION_METHOD,
    initial_feature_filter_size: int = 0,
    initial_seed: int = 42,
    logger_path: Optional[Path] = None,
    notebook_mode: bool = False,
    extra_handlers: Optional[List[logging.Handler]] = None,
    input_data_name: Optional[str] = None,
    group_data_name: Optional[str] = None,
) -> Path:
    """
    Main entry point for the GSM pipeline execution.
    
    Args:
        input_data: Gene expression matrix (samples × genes)
        group_data: Gene grouping information
        sample_ratio: Train/test split ratio
        n_iterations: Number of GSM workflow iterations
        model_name: Selected ML model identifier
        label_column: Column name containing class labels
        normalization_method: Data normalization strategy
        initial_seed: Starting seed for reproducibility
        logger_path: Path where log files will be stored
        notebook_mode: Enable notebook-specific optimizations
        input_data_name: Name of input data source (for logging/output folder)
        group_data_name: Name of grouping data source (for logging/output folder)
        
    Returns:
        Path: The directory where results were saved.
    """
    # Use provided names or fall back to config file values
    main_data_stem = input_data_name if input_data_name else Path(INPUT_EXPRESSION_DATA).stem
    group_data_stem = group_data_name if group_data_name else Path(INPUT_GROUP_DATA).stem
    
    output_folder_path = Path(OUTPUT_DIR) / f"gsm_{time.strftime('%Y_%m_%d-%H_%M_%S')}_{main_data_stem}_{group_data_stem}"
    output_folder_path.mkdir(parents=True, exist_ok=True)
    if logger_path is None:
        logger_path = output_folder_path / "gsm_workflow.log"

    logger = setup_logger(str(logger_path),
                          logger_name='GSM_workflow_logger')

    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

    logger.info(
        "GSM config resolved: "
        f"n_iterations={n_iterations} (default NUMBER_OF_ITERATIONS={NUMBER_OF_ITERATIONS}); "
        f"config_module={gsm_workflow_config.__file__}; "
        f"output_dir={OUTPUT_DIR}"
    )

    # Log actual data being used (not just config file values)
    logger.info("##### ACTUAL DATA BEING PROCESSED #####")
    logger.info(f"INPUT_DATA={main_data_stem} (shape: {input_data.shape})")
    logger.info(f"GROUP_DATA={group_data_stem} (shape: {group_data.shape})")
    
    log_config_values(logger=logger)

    copy_used_config_file(output_dir=output_folder_path, logger=logger)
    
    if extra_handlers:
        for handler in extra_handlers:
            logger.addHandler(handler)

    logger.info("🚀 Starting GSM pipeline...")
    
    # Log methodology overview for transparency
    logger.info("=" * 70)
    logger.info("📋 GSM METHODOLOGY OVERVIEW")
    logger.info("-" * 70)
    logger.info("1. FEATURE FILTERING: Welch's t-test with Benjamini-Hochberg FDR correction")
    logger.info("2. GENE GROUPING: Pre-existing knowledge-based grouping (e.g., DisGeNET)")
    logger.info("3. GROUP SCORING: Embedded feature selection via ML models within each group")
    logger.info("4. MODEL TRAINING: Classification with probability outputs and AUC-ROC")
    logger.info("5. VALIDATION: Bootstrap 95% CI, stratified K-fold cross-validation")
    logger.info("=" * 70)

    # Run the GSM pipeline
    logger.info("Start data preprocessing.")
    data_preprocessed = preprocess_data(
        input_data,
        label_column,
        logger=logger,
        label_of_negative_class=negative_class_label,
        label_of_positive_class=positive_class_label,
        normalization_method=normalization_method
    )
    logger.info("Data preprocessing completed.")
    group_data_processed = preprocess_grouping_data(group_data, 
                                                    gene_column_name=gene_column,
                                                    group_column_name=group_column,
                                                    logger=logger)
    logger.info("Grouping data preprocessing completed.")

    logger.info("🎯 Scoring features once (shared across iterations)...")
    feature_data_x = data_preprocessed.drop(columns=[label_column])
    feature_labels = data_preprocessed[label_column]
    precomputed_feature_scores: List[FeatureScore] = score_all_features(
        feature_data_x,
        feature_labels,
        logger
    )
    logger.info("✅ Feature scoring completed once.")
    
    iteration_results: List[IterationResult] = []
    iteration_times: List[float] = []  # Track iteration durations for estimation
    pipeline_start_time = time.time()
    
    # Changed from range(n_iterations) to range(1, n_iterations + 1)
    # This ensures that for n_iterations=1, it only runs once
    for i in range(1, n_iterations + 1):
        iteration_start_time = time.time()
        
        logger.info("")
        logger.info("#" * 70)
        logger.info(f"#{'':^68}#")
        logger.info(f"#{'ITERATION ' + str(i) + ' / ' + str(n_iterations):^68}#")
        logger.info(f"#{'':^68}#")
        logger.info("#" * 70)
        
        # Set random seed for reproducibility
        iteration_seed = generate_iteration_seed(initial_seed, i)
        set_random_seed(iteration_seed, logger)
        
        modeling_result = gsm_main_loop(
            data=data_preprocessed, 
            grouping_data=group_data_processed, 
            model_name=model_name,
            output_dir=output_folder_path,
            iteration=i,
            logger=logger,
            gene_column=gene_column,
            group_column=group_column,
            precomputed_feature_scores=precomputed_feature_scores,
            save_feature_scores=(i == 1)
        )
        
        iteration_results.append(IterationResult(
            iteration=i,
            random_seed=iteration_seed,
            modeling_results=modeling_result
        ))
        
        # Calculate timing and estimate remaining time
        iteration_duration = time.time() - iteration_start_time
        iteration_times.append(iteration_duration)
        
        avg_time_per_iteration = sum(iteration_times) / len(iteration_times)
        remaining_iterations = n_iterations - i
        estimated_remaining_seconds = avg_time_per_iteration * remaining_iterations
        
        elapsed_total = time.time() - pipeline_start_time
        estimated_finish_time = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
        
        # Format times for display
        def format_duration(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}min"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                return f"{hours}h {mins}min"
        
        logger.info(f"⏱️  Iteration {i} completed in {format_duration(iteration_duration)}")
        if remaining_iterations > 0:
            logger.info(
                f"📊 Progress: {i}/{n_iterations} ({100*i/n_iterations:.0f}%) | "
                f"Elapsed: {format_duration(elapsed_total)} | "
                f"Remaining: ~{format_duration(estimated_remaining_seconds)} | "
                f"ETA: {estimated_finish_time.strftime('%H:%M:%S')}"
            )

    # Save results
    if SAVE_INTERMEDIATE_RESULTS:
        logger.info("Saving intermediate results...")
        save_results.save_modeling_results(
            results=[r.modeling_results for r in iteration_results],
            iteration_metadata=[save_results.IterationMetadata(
                iteration=r.iteration,
                random_seed=r.random_seed
            ) for r in iteration_results],
            output_dir=str(output_folder_path),
            experiment_name="modeling_results",
            logger=logger
        )

        logger.info("Intermediate results saved.")

    # Visualize the F1 scores across different numbers of groups for all iterations and save the plot.
    logger.info("📊 Generating visualizations...")
    viz_data = []
    for res in iteration_results:
        for model_res in res.modeling_results:
            viz_data.append({
                "Iteration": res.iteration,
                "NumGroups": model_res.num_groups_used,
                "F1Score": model_res.f1_score
            })
    
    if viz_data:
        viz_df = pd.DataFrame(viz_data)
        visualize_f1_scores(viz_df, output_folder_path, logger)
    else:
        logger.warning("⚠️ No data available for visualization.")

    # Load results JSON for aggregation
    results_json_path = output_folder_path / "modeling_results_all_iterations.json"
    all_results_data = None
    aggregated_groups = None
    aggregated_features = None
    robust_rank_groups = None
    robust_rank_features = None
    
    if results_json_path.exists():
        import json
        with open(results_json_path, 'r') as f:
            all_results_data = json.load(f)
        
        # Compute and save best averaged groups/features
        logger.info("📊 Computing best averaged rankings...")
        try:
            save_best_averaged_rankings(output_folder_path, all_results_data, logger)
            aggregated_groups = compute_best_averaged_groups(all_results_data, logger)
            aggregated_features = compute_best_averaged_features(all_results_data, logger)
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute averaged rankings: {e}")
        
        # Perform robust rank aggregation on groups
        logger.info("📊 Performing robust rank aggregation...")
        try:
            ranked_groups_path = output_folder_path / "ranked_groups_all_iterations.xlsx"
            ranked_groups_lists = load_ranked_groups_from_excel(ranked_groups_path, logger)
            if ranked_groups_lists:
                aggregated_group_ranking = aggregate_group_ranks_rra(ranked_groups_lists)
                save_aggregated_group_ranking(
                    aggregated_group_ranking, 
                    output_folder_path / "aggregated_group_ranking_rra.xlsx", 
                    logger
                )
                robust_rank_groups = [
                    {
                        'Group Name': item.group_name,
                        'Aggregated P-Value': item.aggregated_p_value,
                        'Aggregated Score': item.aggregated_score,
                        'Average Rank': item.average_rank,
                        'Occurrences': item.occurrences
                    }
                    for item in aggregated_group_ranking.items
                ]
        except Exception as e:
            logger.warning(f"⚠️ Failed to aggregate group rankings: {e}")
        
        # Perform robust rank aggregation on features
        try:
            ranked_features_path = output_folder_path / "ranked_features_all_iterations.xlsx"
            ranked_features_lists = load_ranked_features_from_excel(ranked_features_path, logger)
            if ranked_features_lists:
                aggregated_feature_ranking = aggregate_feature_ranks_rra(ranked_features_lists)
                save_aggregated_feature_ranking(
                    aggregated_feature_ranking,
                    output_folder_path / "aggregated_feature_ranking_rra.xlsx",
                    logger
                )
                robust_rank_features = [
                    {
                        'Feature Name': item.feature_name,
                        'Aggregated P-Value': item.aggregated_p_value,
                        'Aggregated Score': item.aggregated_score,
                        'Average Rank': item.average_rank,
                        'Average Importance': item.average_importance,
                        'Occurrences': item.occurrences
                    }
                    for item in aggregated_feature_ranking.items
                ]
        except Exception as e:
            logger.warning(f"⚠️ Failed to aggregate feature rankings: {e}")

    # Identify the best performing iteration and save a summary report to a text file
    logger.info("📝 Generating summary report...")
    save_results.save_summary_report(
        results=[r.modeling_results for r in iteration_results],
        iteration_metadata=[save_results.IterationMetadata(
            iteration=r.iteration,
            random_seed=r.random_seed
        ) for r in iteration_results],
        output_dir=output_folder_path,
        logger=logger,
        aggregated_groups=aggregated_groups,
        aggregated_features=aggregated_features,
        robust_rank_groups=robust_rank_groups,
        robust_rank_features=robust_rank_features
    )

    # Generate publication-quality figures
    logger.info("📊 Generating publication figures...")
    try:
        if results_json_path.exists():
            figures = generate_all_figures(output_folder_path, results_json_path, logger)
            logger.info(f"✅ Generated {len(figures)} publication figures")
        else:
            logger.warning("⚠️ Results JSON not found, skipping figure generation")
    except Exception as e:
        logger.warning(f"⚠️ Figure generation failed: {e}")

    logger.info("GSM pipeline completed successfully.")
    return output_folder_path

    

##### Main Pipeline Functions #####
def gsm_main_loop(data: pd.DataFrame, 
                  grouping_data: pd.DataFrame,
                  model_name: str, 
                  output_dir: Path,
                  iteration: int,
                  logger,
                  gene_column: str = GENE_COLUMN_NAME,
                  group_column: str = GROUP_COLUMN_NAME,
                  precomputed_feature_scores: Optional[List[FeatureScore]] = None,
                  save_feature_scores: bool = False) -> List[ModelingResult]:
    """
    Executes one complete iteration of the GSM workflow.

    Pipeline Stages:
    1. Data splitting (train/test)
    2. Preliminary feature filtering (t-test)
    3. Gene grouping analysis
    4. Group performance scoring
    5. Model training and evaluation

    Args:
        data: Preprocessed expression data
        grouping_data: Processed group definitions
        model_name: Selected ML model identifier
        output_dir: Directory to save results
        iteration: Current iteration number
        logger: Pipeline logging interface
        gene_column: Column name for gene identifiers
        group_column: Column name for group identifiers

    Technical Notes:
        - Uses stratified sampling for data splitting
        - Implements vectorized operations for performance
        - Supports intermediate result caching
    """
    # Data Splitting
    logger.info("📊 Splitting data (train/test)...")
    train_test_split_data = split_data(data, LABEL_COLUMN_NAME, test_size=0.2, stratify=True)
    
    # Feature Filtering
    logger.info("🔍 Applying preliminary t-test filter...")

    # TODO: Use this one !!!
    filtered_train = preliminary_ttest_filter(train_test_split_data.X_train, 
                                        train_test_split_data.y_train,
                                        threshold=TTEST_THRESHOLD,
                                        initial_feature_filter_size=INITIAL_FEATURE_FILTER_SIZE,
                                        logger=logger)

    # Gene Grouping - Fixed to use the proper function from grouping_utils
    logger.info("🔗 Running gene grouping analysis...")

    # Pass the filtered feature names to grouping
    if filtered_train.selected_feature_names is not None:
        filtered_feature_names = list(filtered_train.selected_feature_names)
    else:
        # Fallback: use the column names from the filtered data if available
        filtered_feature_names = list(train_test_split_data.X_train.columns[filtered_train.selected_features])
    
    group_feature_mappings = run_grouping(grouping_data, 
                                          gene_column_name=gene_column,
                                          group_column_name=group_column,
                                          filtered_features=filtered_feature_names,
                                          logger=logger)

    # Group Scoring
    logger.info("📈 Evaluating group performance...")
    scoring_results = run_scoring(data_x=train_test_split_data.X_train, 
                                labels=train_test_split_data.y_train,
                                model_name=model_name, 
                                groups=group_feature_mappings,
                                output_dir=output_dir,
                                iteration=iteration,
                                logger=logger,
                                feature_scores=precomputed_feature_scores,
                                save_feature_scores=save_feature_scores)

    # Get ranked groups from scoring results
    ranked_groups = scoring_results.ranked_groups
    logger.info("Scoring completed.")

    # Model Training with decreasing numbers of top groups
    logger.info("🤖 Training and evaluating models...")
    modeling_result_list = []
    
    # Start with top 1 group and increase to BEST_GROUPS_TO_KEEP
    if not ranked_groups:
        logger.warning("⚠️ No ranked groups available for modeling.")
        return modeling_result_list

    max_group_count = min(BEST_GROUPS_TO_KEEP, len(ranked_groups))
    previous_feature_count = 0
    for requested_groups in range(1, max_group_count + 1):
        selection = expand_groups_until_feature_increase(
            requested_top_groups=requested_groups,
            group_ranks=ranked_groups,
            group_feature_mapping=group_feature_mappings,
            data_columns=train_test_split_data.X_train.columns,
            baseline_feature_count=previous_feature_count,
            logger=logger
        )

        step_label = f"Step {requested_groups}/{max_group_count}"
        if selection.used_top_groups == selection.requested_top_groups:
            logger.debug(
                f"{step_label}: top {selection.used_top_groups} groups → "
                f"{selection.used_feature_count} features"
            )
        elif selection.used_feature_count > selection.baseline_feature_count:
            logger.debug(
                f"{step_label}: expanded to top {selection.used_top_groups} groups → "
                f"{selection.used_feature_count} features"
            )
        else:
            logger.warning(
                f"{step_label}: top {selection.used_top_groups} groups added no new features"
            )
        
        # Run modeling with decreasing number of top groups
        modeling_result = run_modeling(
            data_train_x=train_test_split_data.X_train,
            data_train_y=train_test_split_data.y_train, 
            data_test_x=train_test_split_data.X_test,
            data_test_y=train_test_split_data.y_test,
            group_ranks=ranked_groups,
            group_feature_mapping=group_feature_mappings,
            model_name=model_name,
            top_n_groups=selection.used_top_groups,
            logger=logger
        )
        modeling_result_list.append(modeling_result)

        if modeling_result.num_features_used > previous_feature_count:
            previous_feature_count = modeling_result.num_features_used
        
        # Log only best results at INFO level
        if modeling_result.f1_score >= 0.9:
            logger.info(
                f"  Groups={modeling_result.num_groups_used} → "
                f"F1={modeling_result.f1_score:.4f} "
                f"AUC={modeling_result.auc_roc:.4f}"
            )
        else:
            logger.debug(
                f"  Groups={modeling_result.num_groups_used} → "
                f"F1={modeling_result.f1_score:.4f}"
            )

    logger.info("✅ Modeling completed.")
    return modeling_result_list

def generate_iteration_seed(initial_seed: int, iteration: int) -> int:
    """Generate a deterministic seed for each iteration based on initial seed."""
    return initial_seed + (iteration * 1000)

def set_random_seed(seed: int, logger) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

##### Main Execution Function #####
def main() -> None:
    """
    Main execution function for the GSM pipeline.
    
    This function orchestrates the complete GSM workflow by:
    1. Resolving project folder and file paths
    2. Loading input expression and group data
    3. Executing the GSM pipeline
    
    Example:
        >>> main()  # Runs the complete pipeline with default configuration
    """
    # Resolve project folder path
    project_folder = Path().resolve()
    print(f"🏠 Project Folder: {project_folder}")
    
    # Set up file paths
    input_file = project_folder / INPUT_EXPRESSION_DATA
    group_file = project_folder / INPUT_GROUP_DATA
    
    print(f"📊 Loading expression data from: {input_file}")
    print(f"🔗 Loading group data from: {group_file}")
    
    # Load input data
    input_data = load_input_file(input_file, separator=MAIN_DATA_FILE_SEPARATOR)
    group_data = load_group_file(group_file, separator=GROUPING_FILE_SEPARATOR)
    
    print(f"✅ Expression data loaded: {input_data.shape}")
    print(f"✅ Group data loaded: {group_data.shape}")
    
    # Execute GSM pipeline
    print("🚀 Starting GSM workflow execution...")
    gsm_run(input_data, group_data)
    print("🎉 GSM workflow completed successfully!")

##### Script Execution Entry Point #####
if __name__ == '__main__':
    """Enable direct script execution from command line."""
    main()
