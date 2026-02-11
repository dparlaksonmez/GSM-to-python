"""
Feature Ranking Output Module 📊

Purpose:
    Save ranked feature results to Excel format with scores and metadata.
    Supports two ranking methods:
    1. Individual scoring: Features ranked by their own ML metrics
    2. Group-derived scoring: Features ranked by their group's F1 score
"""

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from pathlib import Path
import logging

@dataclass
class FeatureRankingOutput:
    """Container for feature ranking output configuration."""
    output_path: Path
    feature_scores: List
    timestamp: str
    model_name: str
    iteration: int


@dataclass
class GroupDerivedFeatureScore:
    """Feature score derived from group performance.
    
    Attributes:
        feature_name: Name of the feature (gene)
        group_name: Name of the best group containing this feature
        group_f1_score: F1 score of the best group
        group_rank: Rank of the best group in this iteration
    """
    feature_name: str
    group_name: str
    group_f1_score: float
    group_rank: int


@dataclass
class GroupDerivedRankingOutput:
    """Container for group-derived feature ranking output."""
    output_path: Path
    feature_scores: List[GroupDerivedFeatureScore]
    timestamp: str
    model_name: str
    iteration: int

def save_ranked_features(
    ranking_data: FeatureRankingOutput,
    logger: logging.Logger
) -> None:
    """
    Save ranked features and their scores to Excel.

    Args:
        ranking_data: FeatureRankingOutput containing ranking results
        logger: Logger instance for tracking

    Raises:
        IOError: If file writing fails
    """
    try:
        output_path = ranking_data.output_path
        logger.info(f"Saving ranked features to {output_path.name}")

        # Create DataFrame from feature scores
        results_df = pd.DataFrame([
            {
                'feature_name': score.feature_name,
                'f1_score': score.f1_score,
                'importance_score': score.importance_score,
                'mutual_info': score.mutual_info
            }
            for score in ranking_data.feature_scores
        ])

        # Sort by importance score
        results_df = results_df.sort_values(
            by='importance_score',
            ascending=False
        ).reset_index(drop=True)

        # Add metadata
        results_df['model'] = ranking_data.model_name
        results_df['timestamp'] = ranking_data.timestamp
        results_df['iteration'] = ranking_data.iteration

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If CSV requested, write CSV (append if exists)
        if output_path.suffix.lower() == '.csv':
            if output_path.exists():
                existing_df = pd.read_csv(output_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(output_path, index=False)
            logger.debug(f"Saved {len(results_df)} features to {output_path.name}")
        else:
            # Excel behavior (append if exists)
            if output_path.exists():
                existing_df = pd.read_excel(output_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_excel(output_path, sheet_name='Ranked Features', index=False)
            logger.debug(f"Saved {len(results_df)} features to {output_path.name}")

    except Exception as e:
        logger.error(f"Failed to save ranked features: {str(e)}")
        raise IOError(f"Feature ranking save failed: {str(e)}")


def compute_group_derived_feature_scores(
    ranked_groups: List,  # List[MetricsData] 
    group_feature_mapping: Dict[str, List[str]],
    logger: logging.Logger
) -> List[GroupDerivedFeatureScore]:
    """
    Compute feature scores derived from their group's performance.
    
    Each feature inherits the F1 score and rank of its BEST performing group.
    If a feature belongs to multiple groups, it gets the score from the 
    highest-ranked group.
    
    Args:
        ranked_groups: List of MetricsData objects (already sorted by F1)
        group_feature_mapping: Dict mapping group_name -> list of features
        logger: Logger instance
    
    Returns:
        List of GroupDerivedFeatureScore, sorted by group F1 score (descending)
    """
    # Build feature -> best group mapping
    # Since ranked_groups is sorted by F1 (descending), first occurrence wins
    feature_best_group: Dict[str, GroupDerivedFeatureScore] = {}
    
    for rank_idx, group_metrics in enumerate(ranked_groups, start=1):
        group_name = group_metrics.name
        group_f1 = group_metrics.f1 or 0.0
        
        # Get features for this group
        features = group_feature_mapping.get(group_name, [])
        
        for feature_name in features:
            # Only assign if not already assigned (first = best group)
            if feature_name not in feature_best_group:
                feature_best_group[feature_name] = GroupDerivedFeatureScore(
                    feature_name=feature_name,
                    group_name=group_name,
                    group_f1_score=group_f1,
                    group_rank=rank_idx
                )
    
    # Sort by group F1 score (descending), then by group rank (ascending)
    sorted_scores = sorted(
        feature_best_group.values(),
        key=lambda x: (-x.group_f1_score, x.group_rank)
    )
    
    logger.debug(f"Group-derived scores for {len(sorted_scores)} features")
    return sorted_scores


def save_group_derived_features(
    ranking_data: GroupDerivedRankingOutput,
    logger: logging.Logger
) -> None:
    """
    Save group-derived feature rankings to Excel.
    
    Features are ranked by their best group's F1 score.

    Args:
        ranking_data: GroupDerivedRankingOutput containing ranking results
        logger: Logger instance for tracking

    Raises:
        IOError: If file writing fails
    """
    try:
        output_path = ranking_data.output_path
        logger.debug(f"Saving group-derived rankings to {output_path.name}")

        # Create DataFrame from feature scores
        results_df = pd.DataFrame([
            {
                'feature_name': score.feature_name,
                'best_group_name': score.group_name,
                'group_f1_score': score.group_f1_score,
                'group_rank': score.group_rank
            }
            for score in ranking_data.feature_scores
        ])

        # Add rank column (already sorted by group F1)
        results_df['feature_rank'] = range(1, len(results_df) + 1)

        # Add metadata
        results_df['model'] = ranking_data.model_name
        results_df['timestamp'] = ranking_data.timestamp
        results_df['iteration'] = ranking_data.iteration

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == '.csv':
            if output_path.exists():
                existing_df = pd.read_csv(output_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(output_path, index=False)
            logger.debug(f"Saved {len(ranking_data.feature_scores)} group-derived rankings to {output_path.name}")
        else:
            if output_path.exists():
                existing_df = pd.read_excel(output_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_excel(output_path, sheet_name='Group Derived Features', index=False)
            logger.debug(f"Saved {len(ranking_data.feature_scores)} group-derived rankings to {output_path.name}")

    except Exception as e:
        logger.error(f"Failed to save group-derived features: {str(e)}")
        raise IOError(f"Group-derived feature ranking save failed: {str(e)}")
