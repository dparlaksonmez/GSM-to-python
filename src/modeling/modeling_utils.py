"""
Utility functions for data preprocessing and model preparation in the GSM pipeline.

Key Functions:
- preprocess_features: Standardizes numeric features and encodes categorical variables
- filter_best_groups: Filters gene groups based on scoring thresholds
- validate_data: Validates input data structure and types
- handle_missing_values: Implements strategies for handling missing data
"""

from typing import List, Dict, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from src.grouping.grouping_utils import GroupFeatureMappingData


def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validates input DataFrame structure and content.
    
    Args:
        data: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if data.empty:
        return False, "DataFrame is empty"
        
    if data.isnull().all().any():
        return False, "DataFrame contains columns with all null values"
        
    return True, ""

def handle_missing_values(
    data: pd.DataFrame,
    strategy: str = 'mean',
    fill_value: Optional[Union[str, float]] = None
) -> pd.DataFrame:
    """
    Handles missing values in the dataset.
    
    Args:
        data: Input DataFrame
        strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')
        fill_value: Value to use if strategy is 'constant'
        
    Returns:
        DataFrame with handled missing values
    """
    numeric_data = data.select_dtypes(include=[np.number])
    categorical_data = data.select_dtypes(include=['object', 'category'])
    
    # Handle numeric columns
    if not numeric_data.empty:
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        data[numeric_data.columns] = imputer.fit_transform(numeric_data)
    
    # Handle categorical columns
    if not categorical_data.empty:
        data[categorical_data.columns] = categorical_data.fillna('missing')
    
    return data

def preprocess_features(
    data: pd.DataFrame,
    logger,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    scaler_type: str = 'standard',
) -> pd.DataFrame:
    """
    Preprocess features by scaling numeric columns and encoding categorical ones.
    
    Args:
        data: Input DataFrame
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        scaler_type: Type of scaler to use ('standard' or 'robust')
        handle_nulls: Whether to handle missing values before preprocessing
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        ValueError: If input validation fails or invalid parameters are provided
    """
    # Validate input data
    is_valid, error_message = validate_data(data)
    if not is_valid:
        raise ValueError(error_message)
    
    # Create copy to avoid modifying original data
    processed_data = data.copy()
    
    # Identify column types if not provided
    if numeric_columns is None:
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
    if categorical_columns is None:
        categorical_columns = processed_data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
    
    # Scale numeric features
    if numeric_columns:
        scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        try:
            processed_data[numeric_columns] = scaler.fit_transform(
                processed_data[numeric_columns]
            )
        except Exception as e:
            logger.error(f"Error scaling numeric features: {str(e)}")
            raise
    
    # Encode categorical features
    for col in categorical_columns:
        try:
            processed_data[col] = pd.Categorical(processed_data[col]).codes
        except Exception as e:
            logger.error(f"Error encoding categorical column {col}: {str(e)}")
            raise
        
    return processed_data

def filter_features_of_groups(
    group_list: List[str],
    group_feature_mapping: List[GroupFeatureMappingData],
    data_x: pd.DataFrame,
    logger: logging.Logger
) -> List[str]:
    """
    Filters features of gene groups based on the provided group list.
    
    Args:
        group_list: List of gene group names to filter
        data_x: DataFrame containing gene expression data
        logger: Logger for logging information and errors
        
    Returns:
        List of filtered gene group names
    """
    # Validate input data
    is_valid, error_message = validate_data(data_x)
    if not is_valid:
        logger.error(f"Invalid input data: {error_message}")
        raise ValueError(error_message)
    
    # Filter features
    all_filtered_features = []
    for group_name in group_list:
        group_data = next((group for group in group_feature_mapping if group.group_name == group_name), None)
        if group_data is None:
            logger.warning(f"Group {group_name} not found in group_feature_mapping")
            continue
        
        # Filter features based on group data — keep only features present in the data
        group_features = [feature for feature in group_data.feature_list if feature in data_x.columns]
        if not group_features:
            logger.warning(f"No features found for group {group_name}")
            continue
        
        all_filtered_features.extend(group_features)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in all_filtered_features:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)
    return unique_features