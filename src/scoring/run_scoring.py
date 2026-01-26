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
    labels: pd.Series,
    model_name: str,
    cross_validation_folds: int
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
            logger=worker_logger
        )
        
        result = score_data(scoring_params)
        result.name = group_name
        return result
    except Exception:
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
    save_feature_scores: bool = False,
    n_jobs: int = DEFAULT_N_JOBS
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
        save_feature_scores: Whether to save feature scores to file
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
        
    Returns:
        ScoringResults containing ranked groups and feature scores
    """
    try:
        # Only score features that passed t-test filtering
        filtered_feature_names = list(data_x.columns)
        if feature_scores is None:
            feature_scores = score_all_features(data_x[filtered_feature_names], labels, logger)

        logger.info(f"📊 Scoring {len(groups)} groups ({len(data_x)} samples, {len(feature_scores)} features)")

        # Prepare scoring tasks (filter valid groups)
        tasks, group_features = _prepare_group_tasks(groups, data_x, logger)
        
        if not tasks:
            logger.warning("⚠️ No valid groups to score")
            return ScoringResults(
                ranked_groups=[],
                feature_scores=feature_scores,
                group_feature_mapping={}
            )

        # Determine number of jobs
        actual_n_jobs = n_jobs if n_jobs != -1 else os.cpu_count() or 1
        logger.info(f"🚀 Parallel scoring with {actual_n_jobs} workers...")

        # Run scoring in parallel using joblib
        # prefer="threads" would share memory but GIL limits parallelism
        # prefer="processes" (default) gives true parallelism for CPU-bound work
        results = list(Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_score_single_group)(
                group_data=data_x[task.available_features],
                group_name=task.group_name,
                labels=labels,
                model_name=model_name,
                cross_validation_folds=cross_validation_folds
            )
            for task in tqdm(tasks, desc="📊 Scoring groups")
        ))
        
        # Filter out failed results (keep only successful MetricsData)
        processed_group_scores: List[MetricsData] = [r for r in results if r is not None]
        
        total_count = len(results)
        failed_count = total_count - len(processed_group_scores)
        if failed_count > 0:
            logger.warning(f"⚠️ {failed_count} groups failed scoring")
        
        logger.info(f"✅ Successfully scored {len(processed_group_scores)} groups")
        
        # Check if we have any valid group scores before ranking
        if not processed_group_scores:
            logger.warning("⚠️ No valid groups to rank - all groups were skipped due to missing features")
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

        # Save ranked features (only when requested)
        if save_feature_scores:
            # 1. Save individual feature scores (ranked by ML importance) to a subfolder
            features_dir = output_dir / "ranked_features_individual"
            features_dir.mkdir(parents=True, exist_ok=True)
            features_output = features_dir / f"iter_{iteration:03d}_individual_features.csv"
            ranking_output = FeatureRankingOutput(
                output_path=features_output,
                feature_scores=feature_scores,
                timestamp=timestamp,
                model_name=model_name,
                iteration=iteration
            )
            save_ranked_features(ranking_output, logger)

            # 2. Save group-derived feature scores (ranked by group F1) to a subfolder
            if ranked_metrics:  # Only if we have ranked groups
                group_derived_scores = compute_group_derived_feature_scores(
                    ranked_groups=ranked_metrics,
                    group_feature_mapping=group_features,
                    logger=logger
                )
                if group_derived_scores:
                    group_derived_dir = output_dir / "ranked_features_group_derived"
                    group_derived_dir.mkdir(parents=True, exist_ok=True)
                    group_derived_output = group_derived_dir / f"iter_{iteration:03d}_group_derived_features.csv"
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
        logger.error(f"❌ Scoring pipeline failed: {str(e)}")
        raise


def score_all_features(
    data_x: pd.DataFrame,
    labels: pd.Series,
    logger
) -> List[FeatureScore]:
    """Score all features in the dataset."""
    logger.info(f"🎯 Starting feature scoring on {len(data_x.columns)} filtered features...")
    feature_scores = score_features(
        data_x=data_x,
        labels=labels,
        feature_names=list(data_x.columns),
        logger=logger
    )
    logger.info(f"✅ Completed scoring {len(feature_scores)} filtered features")
    return feature_scores
