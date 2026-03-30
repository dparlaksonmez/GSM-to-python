"""Shared post-processing helpers for GSM workflow variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.modeling.run_modeling import ModelingResult
from src.scoring.metrics import MetricsData
from src.utils.rank_aggregation import (
    RankedFeatureItem,
    RankedFeatureList,
    RankedGroupItem,
    RankedGroupList,
    aggregate_feature_ranks_rra,
    aggregate_group_ranks_rra,
    aggregate_model_feature_importance_rra,
    compute_best_averaged_features,
    compute_best_averaged_groups,
    save_aggregated_feature_ranking,
    save_aggregated_group_ranking,
    save_best_averaged_rankings,
)


IterationRecord = Dict[str, Any]


def _strip_non_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_non_serializable(item)
            for key, item in value.items()
            if key != 'fitted_model'
        }
    if isinstance(value, list):
        return [_strip_non_serializable(item) for item in value]
    return value


def serialize_modeling_result(modeling_result: ModelingResult) -> Dict[str, Any]:
    payload = {
        key: value
        for key, value in modeling_result.__dict__.items()
        if key != 'fitted_model'
    }
    return _strip_non_serializable(payload)


# Build aggregation-ready payloads once so downstream reporting can stay in memory.
def serialize_iteration_results(iteration_results: List[IterationRecord]) -> List[Dict[str, Any]]:
    return [
        {
            'metadata': {
                'iteration': result['iteration'],
                'random_seed': result['random_seed'],
            },
            'results': [
                serialize_modeling_result(model_result)
                for model_result in result['modeling_results']
            ],
        }
        for result in iteration_results
    ]


# Convert per-iteration scoring output directly into RRA inputs without rereading files.
def build_ranked_group_lists(iteration_results: List[IterationRecord]) -> List[RankedGroupList]:
    ranked_lists: List[RankedGroupList] = []
    for result in iteration_results:
        ranked_groups: List[MetricsData] = result.get('ranked_groups', [])
        if not ranked_groups:
            continue
        list_size = len(ranked_groups)
        items = [
            RankedGroupItem(
                group_name=str(group.name),
                rank=rank,
                list_size=list_size,
            )
            for rank, group in enumerate(ranked_groups, start=1)
        ]
        ranked_lists.append(
            RankedGroupList(
                source_id=f"iteration_{result['iteration']}",
                items=items,
            )
        )
    return ranked_lists


def save_ranked_groups_combined(iteration_results: List[IterationRecord], output_dir: Path, logger) -> None:
    rows: List[Dict[str, Any]] = []
    for result in iteration_results:
        for rank, group in enumerate(result.get('ranked_groups', []), start=1):
            rows.append(
                {
                    'Rank': rank,
                    'Group Name': group.name,
                    'Accuracy': group.accuracy,
                    'F1 Score': group.f1,
                    'Iteration': result['iteration'],
                }
            )
    if not rows:
        return
    pd.DataFrame(rows).to_excel(output_dir / 'ranked_groups_all_iterations.xlsx', index=False)
    logger.debug(f'Saved combined ranked groups ({len(rows)} rows)')


def build_ranked_feature_lists(all_results: List[Dict[str, Any]]) -> List[RankedFeatureList]:
    ranked_lists: List[RankedFeatureList] = []
    for iteration_data in all_results:
        metadata = iteration_data.get('metadata', {})
        iteration = metadata.get('iteration')
        results_list = iteration_data.get('results', [])
        if not results_list:
            continue
        best_result = max(
            results_list,
            key=lambda result: (result.get('num_features_used', 0), result.get('f1_score', 0.0)),
        )
        feature_importance = best_result.get('feature_importance', {})
        if not feature_importance:
            continue
        sorted_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        list_size = len(sorted_features)
        items = [
            RankedFeatureItem(
                feature_name=str(feature_name),
                rank=rank,
                list_size=list_size,
                importance_score=float(importance),
            )
            for rank, (feature_name, importance) in enumerate(sorted_features, start=1)
        ]
        ranked_lists.append(
            RankedFeatureList(
                source_id=f'iteration_{iteration}',
                items=items,
            )
        )
    return ranked_lists


def save_ranked_features_combined(ranked_feature_lists: List[RankedFeatureList], output_dir: Path, logger) -> None:
    rows: List[Dict[str, Any]] = []
    for ranked_list in ranked_feature_lists:
        iteration = int(str(ranked_list.source_id).split('_')[-1])
        for item in ranked_list.items:
            rows.append(
                {
                    'feature_name': item.feature_name,
                    'importance_score': item.importance_score,
                    'iteration': iteration,
                }
            )
    if not rows:
        return
    pd.DataFrame(rows).to_excel(output_dir / 'ranked_features_all_iterations.xlsx', index=False)
    logger.debug(f'Saved combined ranked features ({len(rows)} rows)')


def ensure_results_json(results_json_path: Path, all_results_data: List[Dict[str, Any]], logger) -> None:
    """Persist the lightweight JSON payload only when a downstream file-based step needs it."""
    if results_json_path.exists():
        return
    results_json_path.write_text(json.dumps(all_results_data, indent=2), encoding='utf-8')
    logger.info('Saved lightweight results JSON for downstream reporting steps')


def run_postprocessing_aggregations(
    iteration_results: List[IterationRecord],
    output_dir: Path,
    logger,
) -> Dict[str, Any]:
    """Run the shared post-processing pipeline entirely from in-memory iteration results."""
    all_results_data = serialize_iteration_results(iteration_results)
    ranked_group_lists = build_ranked_group_lists(iteration_results)
    ranked_feature_lists = build_ranked_feature_lists(all_results_data)
    aggregated_groups = None
    aggregated_features = None
    robust_rank_groups = None
    robust_rank_features = None

    try:
        save_ranked_groups_combined(iteration_results, output_dir, logger)
    except Exception as exc:
        logger.warning(f'Failed to build ranked groups workbook: {exc}')

    try:
        save_ranked_features_combined(ranked_feature_lists, output_dir, logger)
    except Exception as exc:
        logger.warning(f'Failed to build ranked features workbook: {exc}')

    try:
        save_best_averaged_rankings(output_dir, all_results_data, logger)
        aggregated_groups = compute_best_averaged_groups(all_results_data, logger)
        aggregated_features = compute_best_averaged_features(all_results_data, logger)
    except Exception as exc:
        logger.warning(f'Failed to compute averaged rankings: {exc}')

    try:
        if ranked_group_lists:
            aggregated_group_ranking = aggregate_group_ranks_rra(ranked_group_lists)
            save_aggregated_group_ranking(
                aggregated_group_ranking,
                output_dir / 'aggregated_group_ranking_rra.xlsx',
                logger,
            )
            robust_rank_groups = [
                {
                    'Group Name': item.group_name,
                    'Aggregated P-Value': item.aggregated_p_value,
                    'Aggregated Score': item.aggregated_score,
                    'Average Rank': item.average_rank,
                    'Occurrences': item.occurrences,
                }
                for item in aggregated_group_ranking.items
            ]
    except Exception as exc:
        logger.warning(f'Failed to aggregate group rankings: {exc}')

    try:
        if ranked_feature_lists:
            aggregated_feature_ranking = aggregate_feature_ranks_rra(ranked_feature_lists)
            save_aggregated_feature_ranking(
                aggregated_feature_ranking,
                output_dir / 'aggregated_feature_ranking_rra.xlsx',
                logger,
            )
            robust_rank_features = [
                {
                    'Feature Name': item.feature_name,
                    'Aggregated P-Value': item.aggregated_p_value,
                    'Aggregated Score': item.aggregated_score,
                    'Average Rank': item.average_rank,
                    'Average Importance': item.average_importance,
                    'Occurrences': item.occurrences,
                }
                for item in aggregated_feature_ranking.items
            ]
    except Exception as exc:
        logger.warning(f'Failed to aggregate feature rankings: {exc}')

    try:
        aggregate_model_feature_importance_rra(all_results_data, output_dir, logger)
    except Exception as exc:
        logger.warning(f'Failed to aggregate model feature importances: {exc}')

    return {
        'all_results_data': all_results_data,
        'aggregated_groups': aggregated_groups,
        'aggregated_features': aggregated_features,
        'robust_rank_groups': robust_rank_groups,
        'robust_rank_features': robust_rank_features,
    }
