"""
T-test based feature selection for gene expression data.

🧬 Purpose: This module filters important genes using statistical t-tests
📊 Main functionality: Compares gene expression between two groups (e.g., disease vs healthy)

Key Functions:
    🔍 perform_ttest: Runs t-test analysis
    ⚖️ filter_by_pvalue: Selects significant genes
    📏 adjust_pvalues: Corrects for multiple testing

For non-Python researchers:
- This is like doing many t-tests in Excel, but automated
- The module handles all statistical corrections automatically
- You just need to provide your data and get filtered genes back
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy import stats
from typing import Union, Optional, Tuple
from dataclasses import dataclass

# Default value for initial feature filter size
DEFAULT_INITIAL_FEATURE_FILTER_SIZE = 1000

##### Data Structure Definitions #####

@dataclass
class GeneExpressionData:
    """Container for gene expression data and labels."""
    expression_matrix: Union[np.ndarray, pd.DataFrame]  # Genes × Samples matrix
    labels: Union[np.ndarray, pd.Series]  # Binary labels (e.g., disease/healthy)
    feature_names: Optional[np.ndarray] = None  # Gene names/IDs

@dataclass
class TTestResults:
    """Container for t-test results."""
    statistics: np.ndarray  # T-test statistics for each gene
    pvalues: np.ndarray  # Raw p-values
    adjusted_pvalues: np.ndarray  # Corrected p-values
    selected_features: np.ndarray  # Indices of selected genes
    selected_feature_names: Optional[np.ndarray] = None  # Names of selected genes

@dataclass
class TTestParameters:
    """Configuration for t-test analysis."""
    threshold: float = 0.05  # P-value cutoff
    equal_var: bool = False  # Whether to assume equal variances
    correction_method: str = 'fdr_bh'  # Multiple testing correction method
    initial_feature_filter_size: int = DEFAULT_INITIAL_FEATURE_FILTER_SIZE  # Max features to keep (0 = no limit)

def perform_ttest(
    data: GeneExpressionData,
    parameters: TTestParameters,
    logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform t-test for each gene between two groups.
    
    Args:
        data: GeneExpressionData object containing expression data and labels
        config: TTestConfig object with analysis parameters
        logger: Logger object for error tracking
    """
    try:
        # Validate inputs
        unique_labels = np.unique(data.labels)
        if len(unique_labels) != 2:
            logger.error(f"Invalid number of unique labels: {unique_labels}")
            raise ValueError(f"Expected 2 unique labels, got {len(unique_labels)}: {unique_labels}")

        # Split data into groups
        group1_mask = data.labels == unique_labels[0]
        group2_mask = data.labels == unique_labels[1]
        
        group1_data = data.expression_matrix[group1_mask]
        group2_data = data.expression_matrix[group2_mask]

        # Perform t-test
        logger.info(f"##### Running t-test on {data.expression_matrix.shape[1]} genes #####")
        t_stats, p_vals = stats.ttest_ind(
            group1_data, 
            group2_data,
            equal_var=parameters.equal_var,
            axis=0
        )

        return np.array(t_stats, dtype=np.float64), np.array(p_vals, dtype=np.float64)

    except Exception as e:
        logger.error(f"❌ Error in t-test calculation: {str(e)}")
        raise

def adjust_pvalues(
    pvalues: np.ndarray,
    config: TTestParameters,
    logger
) -> np.ndarray:
    """Apply multiple testing correction to p-values."""
    try:
        logger.info(f"Applying {config.correction_method} correction")
        return multipletests(pvalues, method=config.correction_method)[1]
    except Exception as e:
        logger.error(f"❌ Error in p-value adjustment: {str(e)}")
        raise

def filter_by_pvalue(
    data: GeneExpressionData,
    pvalues: np.ndarray,
    config: TTestParameters,
    logger
) -> TTestResults:
    """Filter genes based on significance threshold."""
    try:
        # Adjust p-values
        adjusted_pvals = adjust_pvalues(pvalues, config, logger)
        
        # First filter by p-value threshold
        significant_mask = adjusted_pvals < config.threshold
        
        # Only apply top-K filter if initial_feature_filter_size > 0
        if config.initial_feature_filter_size > 0:
            # Take top K features by p-value
            sorted_indices = np.argsort(adjusted_pvals)
            top_k_mask = np.zeros_like(significant_mask)
            top_k_mask[sorted_indices[:config.initial_feature_filter_size]] = True
            # Combine both filters
            final_mask = significant_mask & top_k_mask
        else:
            # Use only significance filter if no size limit
            final_mask = significant_mask
            
        selected_features = np.arange(len(pvalues))[final_mask]
        
        # Get feature names if available
        selected_names = (data.feature_names[final_mask] 
                         if data.feature_names is not None else None)

        logger.info(f"✅ Selected {sum(significant_mask)} significant genes")
        # TODO: mention about the p-value threshold
        if config.initial_feature_filter_size > 0:
            logger.info(f"✅ Final selection: {sum(final_mask)} genes after size filter (initial_feature_filter_size={config.initial_feature_filter_size})")
        
        return TTestResults(
            statistics=pvalues,
            pvalues=pvalues,
            adjusted_pvalues=adjusted_pvals,
            selected_features=selected_features,
            selected_feature_names=selected_names
        )

    except Exception as e:
        logger.error(f"❌ Error in gene filtering: {str(e)}")
        raise

def select_features(
    expression_data: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, pd.Series],
    feature_names: Optional[np.ndarray] = None,
    threshold: float = 0.05,
    equal_var: bool = False,
    logger=None
) -> TTestResults:
    """
    Complete pipeline for t-test based gene selection.
    
    For non-Python users:
    - Input your gene expression matrix (genes in columns)
    - Provide binary labels (e.g., 0 for control, 1 for disease)
    - Optionally provide gene names
    - Get back a list of significant genes
    """
    try:
        # Package input data
        data = GeneExpressionData(
            expression_matrix=np.asarray(expression_data),
            labels=np.asarray(labels),
            feature_names=feature_names
        )
        
        # Create config
        config = TTestParameters(
            threshold=threshold,
            equal_var=equal_var
        )
        
        # Run analysis
        t_stats, p_vals = perform_ttest(data, config, logger)
        results = filter_by_pvalue(data, p_vals, config, logger)
        
        return results

    except Exception as e:
        if logger:
            logger.error(f"❌ Error in feature selection pipeline: {str(e)}")
        raise