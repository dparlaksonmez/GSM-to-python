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
                        CLASS_LABELS_NEGATIVE, CLASS_LABELS_POSITIVE, BEST_GROUPS_TO_KEEP)

# Import the config module itself (not only constants) so we can log where it
# was loaded from at runtime. This is critical for debugging “wrong config file
# / stale notebook kernel / wrong checkout” issues.
import src.workflows.GSM_workflow_config as gsm_workflow_config

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import random

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
    extra_handlers: Optional[List[logging.Handler]] = None
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
        
    Returns:
        Path: The directory where results were saved.
    """
    output_folder_path = Path(OUTPUT_DIR) / f"gsm_{time.strftime('%Y_%m_%d-%H_%M_%S')}"
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

    log_config_values(logger=logger)

    copy_used_config_file(output_dir=output_folder_path, logger=logger)
    
    if extra_handlers:
        for handler in extra_handlers:
            logger.addHandler(handler)

    logger.info("🚀 Starting GSM pipeline...")

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
    # Changed from range(n_iterations) to range(1, n_iterations + 1)
    # This ensures that for n_iterations=1, it only runs once
    for i in range(1, n_iterations + 1):    
        logger.info(f"=" * 50)
        logger.info(f"Iteration {i} for gsm_main_loop started...")
        
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
        logger.info(f"Iteration {i} for gsm_main_loop completed.")

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

    # Identify the best performing iteration and save a summary report to a text file
    logger.info("📝 Generating summary report...")
    save_results.save_summary_report(
        results=[r.modeling_results for r in iteration_results],
        iteration_metadata=[save_results.IterationMetadata(
            iteration=r.iteration,
            random_seed=r.random_seed
        ) for r in iteration_results],
        output_dir=output_folder_path,
        logger=logger
    )

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
    logger.info("##### Starting GSM Main Loop #####")
    
    # Data Splitting
    logger.info("📊 Splitting data into training and testing sets...")
    train_test_split_data = split_data(data, LABEL_COLUMN_NAME, test_size=0.2, stratify=True)
    logger.info("Data split completed.")
    
    # Feature Filtering
    logger.info("🔍 Applying preliminary t-test filter...")

    # TODO: Use this one !!!
    filtered_train = preliminary_ttest_filter(train_test_split_data.X_train, 
                                        train_test_split_data.y_train,
                                        threshold=TTEST_THRESHOLD,
                                        initial_feature_filter_size=INITIAL_FEATURE_FILTER_SIZE,
                                        logger=logger)
    logger.info("Preliminary filtering completed.")

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
    
    logger.info("Grouping completed.")

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
            logger.info(
                f"{step_label}: using top {selection.used_top_groups} groups. "
                f"Unique features: {selection.baseline_feature_count} → {selection.used_feature_count}."
            )
        elif selection.used_feature_count > selection.baseline_feature_count:
            logger.info(
                f"{step_label}: top {selection.requested_top_groups} groups added no new features "
                f"(still {selection.baseline_feature_count}). Expanded to top "
                f"{selection.used_top_groups} groups to reach {selection.used_feature_count} unique features."
            )
        else:
            logger.warning(
                f"{step_label}: top {selection.requested_top_groups} groups added no new features "
                f"(still {selection.baseline_feature_count}). Even after expanding to "
                f"{selection.used_top_groups} groups, the feature count stayed the same."
            )

        logger.info(f"Training model with top {selection.used_top_groups} groups...")
        
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
        
        logger.info(
            f"✅ Model with top {modeling_result.num_groups_used} groups - "
            f"F1 score: {modeling_result.f1_score:.4f}"
        )

    logger.info("Modeling completed successfully.")
    return modeling_result_list

def generate_iteration_seed(initial_seed: int, iteration: int) -> int:
    """Generate a deterministic seed for each iteration based on initial seed."""
    return initial_seed + (iteration * 1000)

def set_random_seed(seed: int, logger) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")

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
    input_data = load_input_file(input_file)
    group_data = load_group_file(group_file)
    
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
