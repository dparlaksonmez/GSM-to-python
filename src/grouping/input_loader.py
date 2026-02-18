# grouping/group_loader.py

"""
Functions for loading and validating grouping input data
"""
from typing import Tuple
import pandas as pd
from src.data_processing.data_loader import _detect_separator

def load_grouping_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate grouping data from file.
    
    Auto-detects the column separator (comma, tab, etc.).
    
    Args:
        filepath (str): Path to the grouping data file
        
    Returns:
        pd.DataFrame: Validated grouping data with feature_name and group_name columns
    """
    sep = _detect_separator(filepath)
    grouping_data = pd.read_csv(filepath, sep=sep)
    required_columns = {'feature_name', 'group_name'}
    
    if not required_columns.issubset(grouping_data.columns):
        missing = required_columns - set(grouping_data.columns)
        raise ValueError(f"Grouping data missing required columns: {missing}")
    
    return grouping_data

def validate_main_data(main_data: pd.DataFrame, 
                      grouping_data: pd.DataFrame) -> None:
    """
    Validate that main data contains all features referenced in grouping data.
    
    Args:
        main_data (pd.DataFrame): Main dataset
        grouping_data (pd.DataFrame): Grouping information
        
    Raises:
        ValueError: If validation fails
    """
    missing_features = set(grouping_data['feature_name']) - set(main_data.columns)
    if missing_features:
        raise ValueError(f"Main data missing features referenced in grouping data: {missing_features}")

def prepare_data(main_data_path: str, 
                grouping_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate both main and grouping data.
    
    Args:
        main_data_path (str): Path to main data file
        grouping_data_path (str): Path to grouping data file
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Validated main data and grouping data
    """
    main_data = pd.read_csv(main_data_path, sep=_detect_separator(main_data_path))
    grouping_data = load_grouping_data(grouping_data_path)
    validate_main_data(main_data, grouping_data)
    return main_data, grouping_data