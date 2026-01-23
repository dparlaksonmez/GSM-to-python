"""
Robust Rank Aggregation utilities for GSM pipeline.

Purpose:
    Combine multiple ranked group lists into a single, stable ranking.

Implementation:
    Uses a Robust Rank Aggregation (RRA) approach based on order statistics.
    Missing groups in a list are treated as worst rank for that list.
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from scipy.stats import beta


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
        logger.info(f"✅ Saved aggregated group rankings to: {output_path}")
    except Exception as exc:
        logger.error(f"❌ Failed to save aggregated group rankings: {exc}")
        raise