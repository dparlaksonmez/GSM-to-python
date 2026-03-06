"""
Main scoring module for the GSM pipeline.

This module coordinates feature and group scoring operations.
Uses parallel processing for faster group scoring.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed
import os

# Default cross-validation folds
DEFAULT_CROSS_VALIDATION_FOLDS = 5
# Default number of parallel jobs (-1 = use all CPUs)
DEFAULT_N_JOBS = -1

from src.utils.save_ranked_features import (
    save_ranked_features, 
    FeatureRankingOutput,
    compute_group_derived_feature_scores,
    save_group_derived_features,
    GroupDerivedRankingOutput,
)
from src.utils.save_ranked_groups import save_ranked_groups
from src.scoring.score_data import ScoringParameters, score_data
from src.scoring.metrics import MetricsData, rank_by_score
from src.scoring.feature_scorer import score_features, FeatureScore
from src.grouping.grouping_utils import GroupFeatureMappingData


@dataclass
class GroupScoringTask:
    """Container for a single group scoring task."""
    group_name: str
    available_features: List[str]


@dataclass
class ScoringResults:
    """Container for all scoring results."""
    ranked_groups: List[MetricsData]
    feature_scores: List[FeatureScore]
    group_feature_mapping: Dict[str, List[str]] = field(default_factory=dict)


##### PARALLEL SCORING HELPER #####
def _score_single_group(
    group_data: pd.DataFrame,
    group_name: str,
    labels,  # np.ndarray or pd.Series — pre-converted for speed
    model_name: str,
    cross_validation_folds: int,
    random_state: int = 42
) -> Optional[MetricsData]:
    """
    Score a single group in a worker process.
    
    This function is designed to be called in parallel by joblib.
    It avoids passing the logger to prevent serialization issues.
    
    Args:
        group_data: Feature matrix for this group
        group_name: Name of the group
        labels: Target labels
        model_name: Classifier name
        cross_validation_folds: Number of CV folds
        random_state: Explicit random seed for this group (required because
            joblib worker processes do not inherit the global np.random.seed)
        
    Returns:
        MetricsData with scoring results, or None if scoring fails
    """
    try:
        # Create a minimal logger for the worker (no file handlers)
        import logging
        worker_logger = logging.getLogger(f"worker_{group_name}")
        worker_logger.setLevel(logging.WARNING)  # Suppress info logs in workers
        
        scoring_params = ScoringParameters(
            data_x=group_data,
            group_name=group_name,
            labels=labels,
            classifier_name=model_name,
            cross_validation_folds=cross_validation_folds,
            logger=worker_logger,
            random_state=random_state
        )
        
        result = score_data(scoring_params)
        result.name = group_name
        return result
    except Exception as e:
        import logging
        worker_logger = logging.getLogger(f"worker_{group_name}")
        worker_logger.warning(f"Scoring failed for group '{group_name}': {e}")
        return None


def _prepare_group_tasks(
    groups: List[GroupFeatureMappingData],
    data_x: pd.DataFrame,
    logger
) -> Tuple[List[GroupScoringTask], Dict[str, List[str]]]:
    """
    Prepare group scoring tasks by validating features.
    
    Args:
        groups: List of group-feature mappings
        data_x: Full feature matrix
        logger: Logger instance
        
    Returns:
        Tuple of (valid scoring tasks, group-to-features mapping)
    """
    feature_map = {col.lower(): col for col in data_x.columns}
    data_columns_set = set(data_x.columns)
    
    tasks: List[GroupScoringTask] = []
    group_features: Dict[str, List[str]] = {}
    
    for current_group in groups:
        available_features = []
        
        for feature in current_group.feature_list:
            if feature in data_columns_set:
                available_features.append(feature)
            else:
                lower_feature = feature.lower()
                if lower_feature in feature_map:
                    available_features.append(feature_map[lower_feature])
        
        if not available_features:
            logger.warning(f"⚠️ Skipping group with no valid features: {current_group.group_name}")
            continue
        
        group_features[current_group.group_name] = available_features
        tasks.append(GroupScoringTask(
            group_name=current_group.group_name,
            available_features=available_features
        ))
    
    return tasks, group_features


def run_scoring(
    data_x: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    groups: List[GroupFeatureMappingData],
    output_dir: Path,
    iteration: int,
    logger,
    cross_validation_folds: int = DEFAULT_CROSS_VALIDATION_FOLDS,
    feature_scores: Optional[List[FeatureScore]] = None,
    should_save_group_features: bool = False,
    n_jobs: int = DEFAULT_N_JOBS,
    random_state: int = 42
) -> ScoringResults:
    """
    Run the complete scoring pipeline for both groups and features.
    
    Uses parallel processing to speed up group scoring significantly.
    
    Args:
        data_x: Feature matrix
        labels: Target labels
        model_name: Name of the model to use
        groups: Group assignments
        output_dir: Directory to save results
        iteration: Current iteration number
        logger: Logger instance
        cross_validation_folds: Number of CV folds (default: 5)
        feature_scores: Pre-computed feature scores (optional)
        should_save_group_features: Whether to save group-derived feature scores
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
        random_state: Base random seed for deterministic scoring across parallel
            workers. Each group gets a unique seed derived from this value.
        
    Returns:
        ScoringResults containing ranked groups and feature scores
    """
    try:
        # Feature scoring is done once before the iteration loop (in gsm_run).
        # Here we only need group-level scoring.
        if feature_scores is None:
            feature_scores = []

        logger.info(f"Scoring {len(groups)} groups ({len(data_x)} samples)")

        # Prepare scoring tasks (filter valid groups)
        tasks, group_features = _prepare_group_tasks(groups, data_x, logger)
        
        if not tasks:
            logger.warning("No valid groups to score")
            return ScoringResults(
                ranked_groups=[],
                feature_scores=feature_scores,
                group_feature_mapping={}
            )

        # Determine number of jobs
        actual_n_jobs = n_jobs if n_jobs != -1 else os.cpu_count() or 1
        logger.debug(f"Parallel scoring with {actual_n_jobs} workers")

        # Run scoring in parallel using joblib
        # prefer="threads" would share memory but GIL limits parallelism
        # prefer="processes" (default) gives true parallelism for CPU-bound work
        # NOTE: tqdm wraps a pre-built list (not the generator) to avoid
        # interfering with joblib's lazy dispatch
        # Pre-convert labels to numpy once to avoid serializing a pandas Series per task
        labels_np = labels.values if hasattr(labels, 'values') else labels
        # Generate a deterministic per-group seed so each parallel worker
        # produces the same result regardless of execution order.
        # Using enumerate index (not group_name hash) keeps it simple and stable.
        scoring_tasks = [
            delayed(_score_single_group)(
                group_data=data_x[task.available_features],
                group_name=task.group_name,
                labels=labels_np,
                model_name=model_name,
                cross_validation_folds=cross_validation_folds,
                random_state=random_state + idx
            )
            for idx, task in enumerate(tasks)
        ]
        results = list(Parallel(n_jobs=n_jobs, verbose=0)(tqdm(scoring_tasks, desc="📊 Scoring groups")))
        
        # Filter out failed results (keep only successful MetricsData)
        processed_group_scores: List[MetricsData] = [r for r in results if r is not None]
        
        total_count = len(results)
        failed_count = total_count - len(processed_group_scores)
        if failed_count > 0:
            logger.warning(f"{failed_count}/{total_count} groups failed scoring")
        
        logger.info(f"Scored {len(processed_group_scores)}/{total_count} groups")
        
        # Check if we have any valid group scores before ranking
        if not processed_group_scores:
            logger.warning("All groups failed scoring (missing features)")
            ranked_metrics = []
        else:
            ranked_groups = rank_by_score(
                metrics_list=processed_group_scores,
                score_type="f1",
                logger=logger
            )
            ranked_metrics = ranked_groups.metrics

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ranked groups per-iteration to a subfolder to avoid large single files
        groups_dir = output_dir / "ranked_groups"
        groups_dir.mkdir(parents=True, exist_ok=True)
        groups_output = groups_dir / f"iter_{iteration:03d}_groups.csv"
        save_ranked_groups(
            str(groups_output),
            ranked_metrics,
            iteration=iteration,
            logger=logger
        )

        # Save group-derived feature scores (cheap — only on iteration 1)
        if should_save_group_features and ranked_metrics:
            group_derived_scores = compute_group_derived_feature_scores(
                ranked_groups=ranked_metrics,
                group_feature_mapping=group_features,
                logger=logger
            )
            if group_derived_scores:
                group_derived_output = output_dir / "group_derived_feature_scores.csv"
                group_derived_ranking = GroupDerivedRankingOutput(
                    output_path=group_derived_output,
                    feature_scores=group_derived_scores,
                    timestamp=timestamp,
                    model_name=model_name,
                    iteration=iteration
                )
                save_group_derived_features(group_derived_ranking, logger)
        
        return ScoringResults(
            ranked_groups=ranked_metrics,
            feature_scores=feature_scores,
            group_feature_mapping=group_features
        )

    except Exception as e:
        logger.error(f"Scoring pipeline failed: {str(e)}")
        raise


def score_all_features(
    data_x: pd.DataFrame,
    labels: pd.Series,
    logger,
    random_state: int = 42
) -> List[FeatureScore]:
    """Score all features in the dataset."""
    feature_scores = score_features(
        data_x=data_x,
        labels=labels,
        feature_names=list(data_x.columns),
        logger=logger,
        random_state=random_state
    )
    return feature_scores
