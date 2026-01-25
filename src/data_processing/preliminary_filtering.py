"""
Preliminary filtering of gene expression data using t-test.

This module implements preliminary filtering to identify differentially expressed genes
between two conditions (e.g., disease vs. control) using t-tests. It provides functions for:
- Loading data
- Performing t-tests
- Filtering genes based on p-value thresholds

Key Functions:
    - preliminary_filter: Main function to perform preliminary filtering on gene expression data

Usage Example:
    >>> import pandas as pd
    >>> X = pd.DataFrame(np.random.randn(100, 1000))  # 100 samples, 1000 genes
    >>> y = np.random.binomial(1, 0.5, 100)  # Binary labels
    >>> results = preliminary_filter(X, y, threshold=0.05)
"""

import numpy as np
import pandas as pd
from typing import Union
from src.feature_selection.ttest_filter import select_features, TTestResults

def preliminary_ttest_filter(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    logger,
    threshold: float = 0.05,
    equal_var: bool = False,
    initial_feature_filter_size: int = 0
) -> TTestResults:
    """
    Perform preliminary filtering on gene expression data using t-tests.

    Args:
        X: Gene expression data (samples × features)
        y: Binary labels indicating groups (0 or 1)
        threshold: P-value threshold for filtering
        feature_names: Optional list of feature names
        equal_var: Whether to assume equal variances

    Returns:
        TTestResults object containing filtered features and statistics

    Example:
        >>> X = np.random.randn(100, 1000)  # 100 samples, 1000 genes
        >>> y = np.random.binomial(1, 0.5, 100)  # Binary labels
        >>> results = preliminary_filter(X, y, threshold=0.05)
    """
    try:
        # Select features using t-test
        results = select_features(X, 
                                  y, 
                                  threshold=threshold, 
                                  equal_var=equal_var, 
                                  initial_feature_filter_size=0,
                                  logger=logger)
        
        return results

    except Exception as e:
        logger.error(f"Error in preliminary_filter: {str(e)}")
        raise
