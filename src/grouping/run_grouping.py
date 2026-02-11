"""
🧬 Core Grouping Module for GSM Pipeline 🧬

This module organizes features into logical groups based on biological relationships.

Key Functions:
-------------
- run_grouping: Main entry point for grouping process
- validate_grouping_data: Validates input data format and content

Example Usage:
-------------
>>> grouping_data = pd.read_csv('group_config.csv')
>>> grouped_features = run_grouping(grouping_data, group_feature_mappings, logger)
>>> print(f"Created {len(grouped_features)} feature groups")
"""

import logging
from typing import List
import pandas as pd

from src.grouping.grouping_utils import DEFAULT_GENE_COLUMN_NAME, DEFAULT_GROUP_COLUMN_NAME, GroupFeatureMappingData, create_group_feature_mapping


def run_grouping(
    grouping_data: pd.DataFrame,
    filtered_features: List[str],
    logger: logging.Logger,
    gene_column_name: str = DEFAULT_GENE_COLUMN_NAME,
    group_column_name: str = DEFAULT_GROUP_COLUMN_NAME,
) -> List[GroupFeatureMappingData]:
    """
    Run the grouping process to organize features into their respective groups.
    
    Args:
        grouping_data: DataFrame with gene and group columns
        filtered_features: List of feature names that passed preliminary filtering
        logger: Logger instance for tracking progress
    
    Returns:
        List[GroupFeatureMappingData]: Organized feature groups
    
    Raises:
        ValueError: If input validation fails
    """
    logger.info(f"Grouping: {len(filtered_features)} filtered features")

    # Create initial group mappings from all grouping data
    all_group_mappings = create_group_feature_mapping(grouping_data, 
                                                      gene_column_name=gene_column_name,
                                                      group_column_name=group_column_name,
                                                      logger=logger)
    
    # Filter groups to only include features that passed preliminary filtering
    # Save original feature lists for fallback if t-test is too aggressive
    filtered_set = set(filtered_features)
    group_feature_mappings = []
    original_feature_lists = {group.group_name: list(group.feature_list) for group in all_group_mappings}
    
    for group in all_group_mappings:
        # Keep only features that are in the filtered set
        filtered_feature_list = [f for f in group.feature_list if f in filtered_set]
        
        # Only include groups that have at least one filtered feature
        if filtered_feature_list:
            group.feature_list = filtered_feature_list
            group_feature_mappings.append(group)
    
    # Fallback: if no groups survived filtering (t-test was too strict or
    # the few surviving features don't overlap with any group), fall back
    # to using ALL features present in the grouping data (bypass t-test).
    if not group_feature_mappings:
        logger.warning(
            f"⚠️ No groups survived after t-test filtering "
            f"({len(filtered_features)} filtered features matched 0 groups). "
            "Falling back to using all features from grouping data (no t-test filter)."
        )
        # Restore original feature lists and use all groups
        for group in all_group_mappings:
            group.feature_list = original_feature_lists[group.group_name]
        group_feature_mappings = [
            group for group in all_group_mappings if group.feature_list
        ]
    
    total_features = sum(len(group.feature_list) for group in group_feature_mappings)
    avg_group_size = total_features / len(group_feature_mappings) if group_feature_mappings else 0
    logger.info(f"Groups: {len(group_feature_mappings)}/{len(all_group_mappings)} | avg {avg_group_size:.1f} features")
    try:
        validate_grouping_data(grouping_data, logger)
        
        if not group_feature_mappings:
            raise ValueError("No valid feature groups provided")
        
        return group_feature_mappings
        
    except Exception as e:
        logger.error(f"Grouping error: {str(e)}")
        raise


def validate_grouping_data(
    grouping_data: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """
    Validate the grouping data format and content.
    
    Args:
        grouping_data: DataFrame containing grouping configuration
        logger: Logger instance for tracking validation

    Raises:
        ValueError: If validation fails
    """
    
    if grouping_data.empty:
        raise ValueError("❌ Grouping data is empty")
        
    # TODO: check here with your specific column requirements
    # required_columns = ['feature_id', 'group_name']
    # missing_cols = [col for col in required_columns if col not in grouping_data.columns]
    # if missing_cols:
    #     raise ValueError(f"❌ Missing required columns: {', '.join(missing_cols)}")
    
    logger.debug("Grouping data validated")