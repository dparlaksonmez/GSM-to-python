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

from src.grouping.grouping_utils import GroupFeatureMappingData, create_group_feature_mapping


def run_grouping(
    grouping_data: pd.DataFrame,
    logger: logging.Logger
) -> List[GroupFeatureMappingData]:
    """
    Run the grouping process to organize features into their respective groups.
    
    Args:
        grouping_data: DataFrame with columns ['feature_id', 'group_name']
        group_feature_mappings: List of existing group-feature mappings
        logger: Logger instance for tracking progress
    
    Returns:
        List[GroupFeatureMappingData]: Organized feature groups
    
    Raises:
        ValueError: If input validation fails
    """
    logger.info(f"#" * 50)
    logger.info(f"🔄 Starting grouping process...")

    group_feature_mappings = create_group_feature_mapping(grouping_data, logger)
        
    # Log information about the groups
    total_features = sum(len(group.feature_list) for group in group_feature_mappings)
    avg_group_size = total_features / len(group_feature_mappings) if group_feature_mappings else 0
    logger.info(f"📊 Total features in groups: {total_features}")
    logger.info(f"📊 Average group size: {avg_group_size:.2f} features")
    try:
        validate_grouping_data(grouping_data, logger)
        
        if not group_feature_mappings:
            raise ValueError("❌ No valid feature groups provided")
        
        logger.info(f"✅ Successfully processed {len(group_feature_mappings)} feature groups")
        return group_feature_mappings
        
    except Exception as e:
        logger.error(f"❌ Error in grouping process: {str(e)}")
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
    
    logger.info("✅ Grouping data validation successful")