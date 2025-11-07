"""
Main scoring module for the GSM pipeline.

This module coordinates feature and group scoring operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Default cross-validation folds
DEFAULT_CROSS_VALIDATION_FOLDS = 5
from src.utils.save_ranked_features import save_ranked_features, FeatureRankingOutput
from src.utils.save_ranked_groups import save_ranked_groups
from src.scoring.score_data import ScoringParameters, score_data
from src.scoring.metrics import MetricsData, rank_by_score
from src.scoring.feature_scorer import score_features, FeatureScore
from src.grouping.grouping_utils import GroupFeatureMappingData


@dataclass
class ScoringResults:
    """Container for all scoring results."""
    ranked_groups: List[MetricsData]
    feature_scores: List[FeatureScore]
    group_feature_mapping: Dict[str, List[str]] = field(default_factory=dict)


def run_scoring(
    data_x: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    groups: List[GroupFeatureMappingData],
    output_dir: Path,
    iteration: int,
    logger,
    cross_validation_folds: int = DEFAULT_CROSS_VALIDATION_FOLDS
) -> ScoringResults:
    """
    # TODO: Parallelize the scoring process for groups
    # TODO: Inspect the function using a profiler to identify bottlenecks
    Run the complete scoring pipeline for both groups and features.
    
    Args:
        data_x: Feature matrix
        labels: Target labels
        model_name: Name of the model to use
        groups: Group assignments
        output_dir: Directory to save results
        iteration: Current iteration number
        logger: Logger instance
        
    Returns:
        ScoringResults containing ranked groups and feature scores
    """
    try:
        # Score individual features
        feature_scores = score_all_features(data_x, labels, logger)
        
        # Process groups
        processed_group_scores = []
        group_features: Dict[str, List[str]] = {}  # Track features per group

        logger.info("#" * 50)
        logger.info("🔍 Found the following groups (first 10 as example):"
                    f"{[group.group_name for group in groups[:10]]}")
        logger.info(f"🔍 Found {len(feature_scores)} features in the dataset")
        logger.info(f"🔍 Found {len(groups)} groups in the dataset")
        logger.info(f"🔍 Found {len(data_x)} samples in the dataset")
        logger.info(f"🔍 Found {len(data_x.columns)} features in the dataset")
        logger.info(f"🔍 Found {len(labels.unique())} unique labels in the dataset")
        logger.info(f"🔍 Found {len(groups)} groups in the dataset")

        logger.info("🔄 Starting group scoring process...")
        for current_group in tqdm(groups, desc="📊 Scoring groups"):
            # Try to match features by normalizing both lists (convert to lowercase)
            feature_map = {col.lower(): col for col in data_x.columns}
            available_features = []
            
            for feature in current_group.feature_list:
                if feature in data_x.columns:
                    available_features.append(feature)
                elif feature.lower() in feature_map:
                    available_features.append(feature_map[feature.lower()])
            
            # logger.info(f"🔍 Scoring group: {current_group.group_name}... Found {len(available_features)} valid features for scoring")
            # this log takes too much space in the log file...
            
            if not available_features:
                logger.warning(f"⚠️ Skipping group with no valid features: {current_group.group_name}")
                continue
                
            group_features[current_group.group_name] = available_features
            group_data = data_x[available_features]

            scoring_params = ScoringParameters(
                data_x=group_data,
                group_name=current_group.group_name,
                labels=labels,
                classifier_name=model_name,
                cross_validation_folds=cross_validation_folds,
                logger=logger
            )
            
            current_score = score_data(scoring_params)
            current_score.name = current_group.group_name
            processed_group_scores.append(current_score)
        
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
        
        # Save ranked groups
        groups_output = output_dir / f"ranked_groups_iteration_{iteration}.xlsx"
        save_ranked_groups(
            str(groups_output),
            ranked_metrics,
            iteration=iteration,
            logger=logger
        )

        # Save ranked features
        features_output = output_dir / f"ranked_features_iteration_{iteration}.xlsx"
        ranking_output = FeatureRankingOutput(
            output_path=features_output,
            feature_scores=feature_scores,
            timestamp=timestamp,
            model_name=model_name
        )
        save_ranked_features(ranking_output, logger)
        
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
    logger.info("🎯 Starting feature scoring...")
    feature_scores = score_features(
        data_x=data_x,
        labels=labels,
        feature_names=list(data_x.columns),
        logger=logger
    )
    logger.info(f"✅ Completed scoring {len(feature_scores)} features")
    return feature_scores
