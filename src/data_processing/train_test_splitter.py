"""
Data Splitting Module for GSM Pipeline

This module handles the splitting of datasets into training, validation, and test sets
for the GSM bioinformatics pipeline. It ensures proper stratification and validation
of input data.

Key Functions:
    - split_data: Split data into train/test or train/validation/test sets
    - validate_split_params: Validate splitting parameters

Example Usage:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'gene1': [1,2,3,4,5],
    ...     'gene2': [2,3,4,5,6],
    ...     'target': [0,1,0,1,0]
    ... })
    >>> X_train, X_test, y_train, y_test = split_data(
    ...     data, 'target', test_size=0.2, stratify=True
    ... )
"""

from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class TrainTestValSplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None



def validate_split_params(
    data: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: Optional[float] = None
) -> None:
    """Validate parameters for data splitting."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Data must be a pandas DataFrame, current type: {type(data)}")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if val_size is not None and not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    
    if val_size is not None and test_size + val_size >= 1:
        raise ValueError("Sum of test_size and val_size must be less than 1")

def split_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify: bool = False,
    random_state: Optional[int] = 42
) -> TrainTestValSplitData:
    """
    Split the dataset into training, validation (optional), and testing sets.

    Args:
        data: The dataset to split
        target_column: The name of the target column in the dataset
        test_size: Proportion of the dataset to use for testing (default: 0.2)
        val_size: Proportion of the dataset to use for validation (default: None)
        stratify: Whether to maintain class distribution in splits (default: False)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        If val_size is None:
            X_train, X_test, y_train, y_test
        If val_size is specified:
            X_train, X_val, X_test, y_train, y_val, y_test

    Raises:
        ValueError: If input parameters are invalid
        TypeError: If data is not a pandas DataFrame
    """
    validate_split_params(data, target_column, test_size, val_size)

    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    stratify_data = y if stratify else None

    if val_size is not None:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_data,
            random_state=random_state
        )
        
        # Second split: separate validation set from remaining data
        val_ratio = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=stratify_temp,
            random_state=random_state
        )
        
        return TrainTestValSplitData(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            X_val=X_val, y_val=y_val
        )
    
    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_data,
        random_state=random_state
    )
    
    return TrainTestValSplitData(X_train=X_train, 
                                 X_test=X_test, 
                                 y_train=y_train, 
                                 y_test=y_test)