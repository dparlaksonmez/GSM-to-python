"""
Robust Rank Aggregation utilities for GSM pipeline.

Purpose:
    Combine multiple ranked group/feature lists into a single, stable ranking.

Implementation:
    Uses a Robust Rank Aggregation (RRA) approach based on order statistics.
    Missing items in a list are treated as worst rank for that list.

Key Functions:
    - aggregate_group_ranks_rra: Aggregates group rankings across iterations
    - aggregate_feature_ranks_rra: Aggregates feature rankings across iterations
    - load_ranked_groups_from_excel: Load ranked groups from Excel file
    - load_ranked_features_from_excel: Load ranked features from Excel file
    - save_aggregated_group_ranking: Save aggregated group rankings to Excel
    - save_aggregated_feature_ranking: Save aggregated feature rankings to Excel
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from scipy.stats import beta


##### DATACLASSES FOR GROUPS #####
@dataclass
class RankedGroupItem:
    """Single ranked group item within a list."""
    group_name: str
    rank: int
    list_size: int


@dataclass
class RankedGroupList:
    """A ranked group list from a single source (iteration or metric)."""
    source_id: str
    items: List[RankedGroupItem]


@dataclass
class AggregatedGroupRank:
    """Aggregated rank output for a single group."""
    group_name: str
    aggregated_p_value: float
    aggregated_score: float
    average_rank: float
    occurrences: int


@dataclass
class AggregatedGroupRanking:
    """Container for aggregated group rankings."""
    items: List[AggregatedGroupRank]


##### DATACLASSES FOR FEATURES #####
@dataclass
class RankedFeatureItem:
    """Single ranked feature item within a list."""
    feature_name: str
    rank: int
    list_size: int
    importance_score: float = 0.0


@dataclass
class RankedFeatureList:
    """A ranked feature list from a single source (iteration)."""
    source_id: str
    items: List[RankedFeatureItem]


@dataclass
class AggregatedFeatureRank:
    """Aggregated rank output for a single feature."""
    feature_name: str
    aggregated_p_value: float
    aggregated_score: float
    average_rank: float
    average_importance: float
    occurrences: int


@dataclass
class AggregatedFeatureRanking:
    """Container for aggregated feature rankings."""
    items: List[AggregatedFeatureRank]


def aggregate_group_ranks_rra(
    ranked_lists: List[RankedGroupList]
) -> AggregatedGroupRanking:
    """Aggregate ranked group lists using Robust Rank Aggregation (RRA)."""
    if not ranked_lists:
        return AggregatedGroupRanking(items=[])

    total_lists = len(ranked_lists)
    group_names = set()
    rank_maps = []
    list_sizes = []

    for ranked_list in ranked_lists:
        mapping = {}
        for item in ranked_list.items:
            mapping[item.group_name] = item
            group_names.add(item.group_name)
        rank_maps.append(mapping)
        list_sizes.append(len(ranked_list.items))

    aggregated_items: List[AggregatedGroupRank] = []

    for group_name in sorted(group_names):
        normalized_ranks = []
        raw_ranks = []
        occurrences = 0

        for rank_map, list_size in zip(rank_maps, list_sizes):
            item = rank_map.get(group_name)
            if item is None:
                rank = list_size + 1 if list_size > 0 else 1
            else:
                rank = item.rank
                occurrences += 1

            raw_ranks.append(rank)
            normalized_ranks.append(rank / list_size if list_size > 0 else 1.0)

        normalized_ranks.sort()
        p_values = []
        for k, r in enumerate(normalized_ranks, start=1):
            p_values.append(beta.cdf(r, k, total_lists - k + 1))
        aggregated_p = min(p_values) if p_values else 1.0
        aggregated_score = -np.log10(aggregated_p) if aggregated_p > 0 else float("inf")
        average_rank = sum(raw_ranks) / len(raw_ranks) if raw_ranks else 0.0

        aggregated_items.append(
            AggregatedGroupRank(
                group_name=group_name,
                aggregated_p_value=aggregated_p,
                aggregated_score=aggregated_score,
                average_rank=average_rank,
                occurrences=occurrences
            )
        )

    aggregated_items.sort(
        key=lambda x: (x.aggregated_p_value, x.average_rank)
    )

    return AggregatedGroupRanking(items=aggregated_items)


def save_aggregated_group_ranking(
    ranking: AggregatedGroupRanking,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Save aggregated group ranking to an Excel file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Group Name": item.group_name,
                "Aggregated P-Value": item.aggregated_p_value,
                "Aggregated Score": item.aggregated_score,
                "Average Rank": item.average_rank,
                "Occurrences": item.occurrences
            }
            for idx, item in enumerate(ranking.items)
        ])
        df.to_excel(output_path, index=False)
        logger.debug(f"Saved aggregated groups: {output_path.name}")
    except Exception as exc:
        logger.error(f"❌ Failed to save aggregated group rankings: {exc}")
        raise


##### FEATURE AGGREGATION #####
def aggregate_feature_ranks_rra(
    ranked_lists: List[RankedFeatureList]
) -> AggregatedFeatureRanking:
    """Aggregate ranked feature lists using Robust Rank Aggregation (RRA)."""
    if not ranked_lists:
        return AggregatedFeatureRanking(items=[])

    total_lists = len(ranked_lists)
    feature_names = set()
    rank_maps: List[Dict[str, RankedFeatureItem]] = []
    list_sizes = []

    for ranked_list in ranked_lists:
        mapping = {}
        for item in ranked_list.items:
            mapping[item.feature_name] = item
            feature_names.add(item.feature_name)
        rank_maps.append(mapping)
        list_sizes.append(len(ranked_list.items))

    aggregated_items: List[AggregatedFeatureRank] = []

    for feature_name in sorted(feature_names):
        normalized_ranks = []
        raw_ranks = []
        importance_scores = []
        occurrences = 0

        for rank_map, list_size in zip(rank_maps, list_sizes):
            item = rank_map.get(feature_name)
            if item is None:
                rank = list_size + 1 if list_size > 0 else 1
            else:
                rank = item.rank
                importance_scores.append(item.importance_score)
                occurrences += 1

            raw_ranks.append(rank)
            normalized_ranks.append(rank / list_size if list_size > 0 else 1.0)

        normalized_ranks.sort()
        p_values = []
        for k, r in enumerate(normalized_ranks, start=1):
            p_values.append(beta.cdf(r, k, total_lists - k + 1))
        aggregated_p = min(p_values) if p_values else 1.0
        aggregated_score = -np.log10(aggregated_p) if aggregated_p > 0 else float("inf")
        average_rank = sum(raw_ranks) / len(raw_ranks) if raw_ranks else 0.0
        average_importance = np.mean(importance_scores) if importance_scores else 0.0

        aggregated_items.append(
            AggregatedFeatureRank(
                feature_name=feature_name,
                aggregated_p_value=aggregated_p,
                aggregated_score=aggregated_score,
                average_rank=average_rank,
                average_importance=average_importance,
                occurrences=occurrences
            )
        )

    aggregated_items.sort(
        key=lambda x: (x.aggregated_p_value, x.average_rank)
    )

    return AggregatedFeatureRanking(items=aggregated_items)


def save_aggregated_feature_ranking(
    ranking: AggregatedFeatureRanking,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Save aggregated feature ranking to an Excel file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Feature Name": item.feature_name,
                "Aggregated P-Value": item.aggregated_p_value,
                "Aggregated Score": item.aggregated_score,
                "Average Rank": item.average_rank,
                "Average Importance": item.average_importance,
                "Occurrences": item.occurrences
            }
            for idx, item in enumerate(ranking.items)
        ])
        df.to_excel(output_path, index=False)
        logger.debug(f"Saved aggregated features: {output_path.name}")
    except Exception as exc:
        logger.error(f"❌ Failed to save aggregated feature rankings: {exc}")
        raise


##### LOADERS FROM EXCEL #####
def load_ranked_groups_from_excel(
    excel_path: Path,
    logger: logging.Logger
) -> List[RankedGroupList]:
    """
    Load ranked group data from Excel file and convert to RankedGroupList objects.
    
    Args:
        excel_path: Path to ranked_groups_all_iterations.xlsx
        logger: Logger instance
    
    Returns:
        List of RankedGroupList objects, one per iteration
    """
    if not excel_path.exists():
        logger.warning(f"Ranked groups file missing: {excel_path.name}")
        return []
    
    try:
        df = pd.read_excel(excel_path)
        
        if df.empty:
            logger.warning("Ranked groups file is empty")
            return []
        
        ranked_lists = []
        
        for iteration in df['Iteration'].unique():
            iter_df = df[df['Iteration'] == iteration].copy()
            iter_df = iter_df.sort_values('Rank', ascending=True)
            
            list_size = len(iter_df)
            items = [
                RankedGroupItem(
                    group_name=str(row['Group Name']),
                    rank=int(row['Rank']),
                    list_size=list_size
                )
                for _, row in iter_df.iterrows()
            ]
            
            ranked_lists.append(
                RankedGroupList(
                    source_id=f"iteration_{iteration}",
                    items=items
                )
            )
        
        logger.debug(f"Loaded {len(ranked_lists)} group lists")
        return ranked_lists
        
    except Exception as e:
        logger.error(f"❌ Failed to load ranked groups from Excel: {e}")
        return []


def load_ranked_features_from_excel(
    excel_path: Path,
    logger: logging.Logger
) -> List[RankedFeatureList]:
    """
    Load ranked feature data from Excel file and convert to RankedFeatureList objects.
    
    Args:
        excel_path: Path to ranked_features_all_iterations.xlsx
        logger: Logger instance
    
    Returns:
        List of RankedFeatureList objects, one per iteration
    """
    if not excel_path.exists():
        logger.warning(f"Ranked features file missing: {excel_path.name}")
        return []
    
    try:
        df = pd.read_excel(excel_path)
        
        if df.empty:
            logger.warning("Ranked features file is empty")
            return []
        
        ranked_lists = []
        
        for iteration in df['iteration'].unique():
            iter_df = df[df['iteration'] == iteration].copy()
            iter_df = iter_df.sort_values('importance_score', ascending=False)
            iter_df['rank'] = range(1, len(iter_df) + 1)
            
            list_size = len(iter_df)
            items = [
                RankedFeatureItem(
                    feature_name=str(row['feature_name']),
                    rank=int(row['rank']),
                    list_size=list_size,
                    importance_score=float(row.get('importance_score', 0.0))
                )
                for _, row in iter_df.iterrows()
            ]
            
            ranked_lists.append(
                RankedFeatureList(
                    source_id=f"iteration_{iteration}",
                    items=items
                )
            )
        
        logger.debug(f"Loaded {len(ranked_lists)} feature lists")
        return ranked_lists
        
    except Exception as e:
        logger.error(f"❌ Failed to load ranked features from Excel: {e}")
        return []


##### BEST AVERAGED RANKINGS FROM MODELING RESULTS #####
def compute_best_averaged_groups(
    all_results: List[Dict],
    logger: logging.Logger
) -> List[Dict]:
    """
    Compute the best averaged groups based on modeling results JSON data.
    
    Groups are ranked by how often they appear in top configurations
    and their average F1 score contribution.
    
    Args:
        all_results: Loaded modeling_results_all_iterations.json data
        logger: Logger instance
    
    Returns:
        List of dicts with group statistics, sorted by average F1 score
    """
    group_stats: Dict[str, Dict] = {}
    
    for iteration_data in all_results:
        for result in iteration_data.get('results', []):
            used_groups = result.get('used_groups', [])
            f1_score = result.get('f1_score', 0.0)
            
            for group_name in used_groups:
                if group_name not in group_stats:
                    group_stats[group_name] = {
                        'f1_scores': [],
                        'occurrences': 0
                    }
                group_stats[group_name]['f1_scores'].append(f1_score)
                group_stats[group_name]['occurrences'] += 1
    
    best_groups = []
    for group_name, stats in group_stats.items():
        best_groups.append({
            'Group Name': group_name,
            'Average F1 Score': np.mean(stats['f1_scores']),
            'Std F1 Score': np.std(stats['f1_scores']),
            'Occurrences': stats['occurrences'],
            'Best F1 Score': max(stats['f1_scores']),
            'Worst F1 Score': min(stats['f1_scores'])
        })
    
    best_groups.sort(key=lambda x: x['Average F1 Score'], reverse=True)
    logger.debug(f"Stats for {len(best_groups)} groups")
    return best_groups


def compute_best_averaged_features(
    all_results: List[Dict],
    logger: logging.Logger
) -> List[Dict]:
    """
    Compute the best averaged features based on feature importance across all results.
    
    Args:
        all_results: Loaded modeling_results_all_iterations.json data
        logger: Logger instance
    
    Returns:
        List of dicts with feature statistics, sorted by average importance
    """
    feature_stats: Dict[str, Dict] = {}
    
    for iteration_data in all_results:
        for result in iteration_data.get('results', []):
            feature_importance = result.get('feature_importance', {})
            
            for feature_name, importance in feature_importance.items():
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'importance_scores': [],
                        'occurrences': 0
                    }
                feature_stats[feature_name]['importance_scores'].append(importance)
                feature_stats[feature_name]['occurrences'] += 1
    
    best_features = []
    for feature_name, stats in feature_stats.items():
        best_features.append({
            'Feature Name': feature_name,
            'Average Importance': np.mean(stats['importance_scores']),
            'Std Importance': np.std(stats['importance_scores']),
            'Occurrences': stats['occurrences'],
            'Max Importance': max(stats['importance_scores']),
            'Min Importance': min(stats['importance_scores'])
        })
    
    best_features.sort(key=lambda x: x['Average Importance'], reverse=True)
    logger.debug(f"Stats for {len(best_features)} features")
    return best_features


def save_best_averaged_rankings(
    output_dir: Path,
    all_results: List[Dict],
    logger: logging.Logger
) -> None:
    """
    Save best averaged groups and features to Excel files.
    
    Args:
        output_dir: Output directory
        all_results: Loaded modeling_results_all_iterations.json data
        logger: Logger instance
    """
    # Compute and save best averaged groups
    best_groups = compute_best_averaged_groups(all_results, logger)
    if best_groups:
        groups_df = pd.DataFrame(best_groups)
        groups_df.insert(0, 'Rank', range(1, len(groups_df) + 1))
        groups_path = output_dir / "best_averaged_groups.xlsx"
        groups_df.to_excel(groups_path, index=False)
        logger.debug(f"Saved averaged groups: {groups_path.name}")
    
    # Compute and save best averaged features
    best_features = compute_best_averaged_features(all_results, logger)
    if best_features:
        features_df = pd.DataFrame(best_features)
        features_df.insert(0, 'Rank', range(1, len(features_df) + 1))
        features_path = output_dir / "best_averaged_features.xlsx"
        features_df.to_excel(features_path, index=False)
        logger.debug(f"Saved averaged features: {features_path.name}")


##### MODEL FEATURE IMPORTANCE RRA #####
def extract_model_feature_importance_lists(
    all_results: List[Dict],
    logger: logging.Logger
) -> List[RankedFeatureList]:
    """
    Extract per-iteration feature importance rankings from modeling results JSON.

    For each iteration, the model's native feature importances (e.g. XGBoost gain)
    are aggregated across all group-count steps by taking the maximum importance
    each feature received. The features are then ranked by descending importance,
    producing one RankedFeatureList per iteration.

    Args:
        all_results: Loaded modeling_results_all_iterations.json data
        logger: Logger instance

    Returns:
        List of RankedFeatureList objects, one per iteration

    Example:
        >>> with open("modeling_results_all_iterations.json") as f:
        ...     data = json.load(f)
        >>> lists = extract_model_feature_importance_lists(data, logger)
        >>> print(f"Got {len(lists)} iteration lists")
    """
    ranked_lists: List[RankedFeatureList] = []

    for idx, iteration_data in enumerate(all_results):
        iteration_id = iteration_data.get('iteration', idx + 1)

        # Collect max importance per feature across all group-count steps
        feature_max_importance: Dict[str, float] = {}
        for result in iteration_data.get('results', []):
            for feature_name, importance in result.get('feature_importance', {}).items():
                current_max = feature_max_importance.get(feature_name, 0.0)
                feature_max_importance[feature_name] = max(current_max, importance)

        if not feature_max_importance:
            continue

        # Sort by importance (descending) and assign ranks
        sorted_features = sorted(
            feature_max_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        list_size = len(sorted_features)
        items = [
            RankedFeatureItem(
                feature_name=name,
                rank=rank,
                list_size=list_size,
                importance_score=importance
            )
            for rank, (name, importance) in enumerate(sorted_features, start=1)
        ]

        ranked_lists.append(
            RankedFeatureList(
                source_id=f"iteration_{iteration_id}",
                items=items
            )
        )

    logger.debug(
        f"Extracted model feature importance lists from {len(ranked_lists)} iterations"
    )
    return ranked_lists


def save_model_feature_importance_per_iteration(
    all_results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
) -> Path:
    """
    Save per-iteration model feature importances to a single Excel file.

    Each iteration gets its own rows. Features are ranked by descending
    importance within each iteration. This file is useful for inspecting
    which genes the model relied on in each train/test split.

    Args:
        all_results: Loaded modeling_results_all_iterations.json data
        output_dir: Directory to save the output file
        logger: Logger instance

    Returns:
        Path to the saved Excel file
    """
    ranked_lists = extract_model_feature_importance_lists(all_results, logger)

    rows = []
    for ranked_list in ranked_lists:
        for item in ranked_list.items:
            rows.append({
                'iteration': ranked_list.source_id,
                'feature_name': item.feature_name,
                'rank': item.rank,
                'importance_score': item.importance_score
            })

    output_path = output_dir / "model_feature_importance_all_iterations.xlsx"
    if rows:
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False)
        logger.debug(
            f"Saved model feature importances: {output_path.name} "
            f"({len(ranked_lists)} iterations, {len(rows)} total entries)"
        )
    else:
        logger.warning("No model feature importance data to save")

    return output_path


def aggregate_model_feature_importance_rra(
    all_results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
) -> Optional[AggregatedFeatureRanking]:
    """
    Run RRA on per-iteration model feature importances and save the result.

    This applies Robust Rank Aggregation to the XGBoost (or other model)
    native feature importances, identifying genes that are consistently
    important across multiple random train/test splits.

    Low aggregated p-value = gene is reliably ranked high regardless of
    which samples are in the train vs test set — a strong biomarker signal.

    Args:
        all_results: Loaded modeling_results_all_iterations.json data
        output_dir: Directory to save the aggregated result
        logger: Logger instance

    Returns:
        AggregatedFeatureRanking or None if no data available
    """
    # Save per-iteration file for inspection
    save_model_feature_importance_per_iteration(all_results, output_dir, logger)

    # Extract and aggregate
    ranked_lists = extract_model_feature_importance_lists(all_results, logger)
    if not ranked_lists:
        logger.warning("No model feature importance lists for RRA aggregation")
        return None

    aggregated = aggregate_feature_ranks_rra(ranked_lists)

    # Save aggregated result
    save_aggregated_feature_ranking(
        aggregated,
        output_dir / "aggregated_model_feature_importance_rra.xlsx",
        logger
    )
    logger.info(
        f"📊 Model feature importance RRA: {len(aggregated.items)} features aggregated "
        f"from {len(ranked_lists)} iterations"
    )
    return aggregated