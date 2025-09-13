"""
🧬 Grouping Utilities Module 🧬

This module provides data structures and utility functions for feature grouping operations.

Key Data Structures:
------------------
- GroupFeatureMappingData: Maps groups to features and vice versa

Key Functions:
-------------
- create_group_feature_mapping: Creates mapping between groups and their features

Example Usage:
------------
>>> grouping_data = pd.read_csv('disease_gene_associations.csv')
>>> group_mappings = create_group_feature_mapping(grouping_data)
>>> print(f"Created {len(group_mappings)} feature groups")
"""

import pandas as pd
import logging
from typing import List, Optional
from dataclasses import dataclass, field

# Default column names for grouping data
DEFAULT_GROUP_COLUMN_NAME = "diseaseName"
DEFAULT_GENE_COLUMN_NAME = "geneSymbol"


@dataclass
class GroupFeatureMappingData:
    """
    Data structure representing the mapping between a group and its features.
    
    Attributes:
        group_name: Unique identifier for the group
        feature_list: List of features belonging to this group
    """
    group_name: str
    feature_list: List[str] = field(default_factory=list)


def create_group_feature_mapping(
    grouping_data: pd.DataFrame,
    group_column_name: str = DEFAULT_GROUP_COLUMN_NAME,
    gene_column_name: str = DEFAULT_GENE_COLUMN_NAME,
    logger: Optional[logging.Logger] = None
) -> List[GroupFeatureMappingData]:
    """
    Create mappings between groups and their constituent features.
    
    Args:
        grouping_data: DataFrame with at least two columns - one for group names 
                      and one for feature/gene names
        logger: Optional logger for progress tracking
    
    Returns:
        List of GroupFeatureMappingData objects representing group-feature mappings
    
    Example:
        >>> data = pd.DataFrame({
        >>>     'diseaseName': ['Cancer', 'Cancer', 'Diabetes'],
        >>>     'geneSymbol': ['BRCA1', 'TP53', 'INS']
        >>> })
        >>> mappings = create_group_feature_mapping(data)
        >>> print(f"Cancer group has {len(mappings[0].feature_list)} genes")
    """
    if logger:
        logger.info("🔄 Creating group-feature mappings...")
    
    # Validate input data
    if grouping_data.empty:
        if logger:
            logger.error("❌ Empty grouping data provided")
        return []
    
    # Check if required columns exist
    required_columns = [group_column_name, gene_column_name]
    for column in required_columns:
        if column not in grouping_data.columns:
            if logger:
                logger.error(f"❌ Required column '{column}' not found in grouping data")
            return []
    
    # Create dictionary to store group-feature mappings
    group_mappings = {}
    
    # Process each row in the grouping data
    for _, row in grouping_data.iterrows():
        group_name = str(row[group_column_name])
        feature_name = str(row[gene_column_name])
        
        # Skip if either is empty
        if not group_name or not feature_name:
            continue
        
        # Create new group if it doesn't exist
        if group_name not in group_mappings:
            group_mappings[group_name] = GroupFeatureMappingData(group_name=group_name)
        
        # Add feature to the group
        group_mappings[group_name].feature_list.append(feature_name)
    
    # Convert dictionary to list
    result = list(group_mappings.values())
    
    if logger:
        logger.info(f"✅ Created {len(result)} group-feature mappings")
        
        # Log some statistics for the top 5 largest groups
        if result:
            sorted_groups = sorted(result, key=lambda g: len(g.feature_list), reverse=True)
            logger.info("📊 Top 5 largest groups:")
            for i, group in enumerate(sorted_groups[:5], 1):
                logger.info(f"  {i}. {group.group_name}: {len(group.feature_list)} features")
    
    return result