"""
GSM with Z-Score Pipeline
=========================

This script implements the GSM pipeline using Z-score based group scoring.
It integrates data loading, preprocessing, preliminary filtering, and group scoring.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import joblib
from typing import List, Tuple

# Import project modules
from src.workflows import GSM_workflow_config as config
from src.data_processing.data_loader import load_input_file, load_group_file
from src.data_processing.handle_missing_values import drop_missing_values
from src.data_processing.preliminary_filtering import preliminary_ttest_filter
from src.data_processing.data_preprocess import convert_labels_to_binary
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("GSM_ZScore")

def compute_group_sum(gene_list, t_gene_df):
    matched = t_gene_df[t_gene_df['gene'].isin(gene_list)]
    return matched['t_value'].sum()


def _prepare_group_dataframe(
    group_data: pd.DataFrame,
    gene_column_name: str,
    group_column_name: str,
    selected_genes: List[str],
    logger,
) -> pd.DataFrame:
    """Normalize grouping columns and filter to selected genes."""
    if gene_column_name in group_data.columns and group_column_name in group_data.columns:
        group_data_renamed = group_data.rename(
            columns={gene_column_name: 'gene', group_column_name: 'group'}
        )
    else:
        logger.warning(
            f"Configured grouping columns ({gene_column_name}, {group_column_name}) not found. "
            "Using first two columns."
        )
        group_data_renamed = group_data.iloc[:, [0, 1]].copy()
        group_data_renamed.columns = ['gene', 'group']

    # Filter group_data to only include genes that are in selected_genes
    group_data_filtered = group_data_renamed[group_data_renamed['gene'].isin(selected_genes)]
    return group_data_filtered


def zscore_group_filter(
    filter_results,
    feature_names: List[str],
    group_data: pd.DataFrame,
    *,
    gene_column_name: str,
    group_column_name: str,
    q_value_threshold: float,
    logger,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Apply z-score based group filtering using existing t-test results.

    Returns a tuple of (selected_feature_names, significant_group_df).
    """
    selected_indices = filter_results.selected_features
    if filter_results.selected_feature_names is not None:
        selected_genes = list(filter_results.selected_feature_names)
    else:
        selected_genes = [feature_names[idx] for idx in selected_indices]

    # Build t-stat DataFrame for selected genes
    t_stats = filter_results.statistics
    if t_stats is None or len(t_stats) == 0:
        logger.warning("No t-statistics available; skipping z-score group filtering.")
        return [], pd.DataFrame()
    selected_t_stats = t_stats[selected_indices]
    t_gene = pd.DataFrame({'gene': selected_genes, 't_value': selected_t_stats})

    group_data_filtered = _prepare_group_dataframe(
        group_data, gene_column_name, group_column_name, selected_genes, logger
    )

    # Group by 'group' and collect 'genes'
    group_df = group_data_filtered.groupby('group')['gene'].apply(list).reset_index()
    if group_df.empty:
        logger.warning("No groups with selected genes found for z-score filtering.")
        return [], pd.DataFrame()

    group_df['sizes'] = group_df['gene'].apply(len)

    logger.info(f"Calculating group scores (Z-values) for {len(group_df)} groups...")
    group_df['t_sum'] = group_df['gene'].apply(lambda genes: compute_group_sum(genes, t_gene))

    # Z-value calculation (matching original logic: sqrt(n) * Sum(t))
    group_df['z_value'] = np.sqrt(group_df['sizes']) * group_df['t_sum']

    # Calculate p-values
    group_df['p_value'] = 2 * (1 - norm.cdf(abs(group_df['z_value'])))

    # FDR Correction
    p_values = group_df['p_value'].values
    _, q_values, _, _ = multipletests(p_values, method='fdr_bh')
    group_df['q_value'] = q_values

    significant_df_z = group_df[group_df['q_value'] < q_value_threshold]
    logger.info(f"Found {len(significant_df_z)} significant groups (q < {q_value_threshold}).")

    if len(significant_df_z) == 0:
        return [], significant_df_z

    # Extract unique features from significant groups
    all_group_elements = [item for sublist in significant_df_z['gene'] for item in sublist]
    unique_elements = sorted(set(all_group_elements))
    selected_features = unique_elements
    return selected_features, significant_df_z

def run_zscore_pipeline():
    logger.info("Starting GSM Z-Score Pipeline...")

    # 1. Load Data
    logger.info(f"Loading expression data from {config.INPUT_EXPRESSION_DATA}...")
    input_data = load_input_file(config.INPUT_EXPRESSION_DATA, separator=config.MAIN_DATA_FILE_SEPARATOR)
    
    logger.info(f"Loading grouping data from {config.INPUT_GROUP_DATA}...")
    group_data = load_group_file(config.INPUT_GROUP_DATA, separator=config.GROUPING_FILE_SEPARATOR)
    
    # 2. Handle Missing Values
    logger.info("Handling missing values...")
    input_data = drop_missing_values(input_data)
    
    # 3. Preprocessing and Filtering
    if config.LABEL_COLUMN_NAME not in input_data.columns:
         raise ValueError(f"Column '{config.LABEL_COLUMN_NAME}' not found in input data")

    # Convert labels to binary
    input_data = convert_labels_to_binary(
        input_data, 
        config.LABEL_COLUMN_NAME, 
        config.CLASS_LABELS_NEGATIVE, 
        config.CLASS_LABELS_POSITIVE
    )

    # Separate features and target
    X = input_data.drop(columns=[config.LABEL_COLUMN_NAME])
    y = input_data[config.LABEL_COLUMN_NAME]
    
    # Ensure numeric features
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # 4. Preliminary Filtering (T-test)
    logger.info("Performing preliminary filtering...")
    
    filter_results = preliminary_ttest_filter(
        X, y, 
        logger=logger, 
        threshold=config.TTEST_THRESHOLD
    )
    
    selected_indices = filter_results.selected_features
    selected_genes = X.columns[selected_indices]
    selected_t_stats = filter_results.statistics[selected_indices]
    
    logger.info(f"Selected {len(selected_genes)} genes after filtering.")
    
    # Create DataFrame of Genes and T-values for downstream processing
    t_gene = pd.DataFrame({'Gene': selected_genes, 't_Value': selected_t_stats})
    
    # 5. Grouping and Scoring
    logger.info("Processing grouping data...")
    
    if config.GENE_COLUMN_NAME in group_data.columns and config.GROUP_COLUMN_NAME in group_data.columns:
        group_data_renamed = group_data.rename(columns={config.GENE_COLUMN_NAME: 'gene', config.GROUP_COLUMN_NAME: 'group'})
    else:
        logger.warning(f"Configured grouping columns ({config.GENE_COLUMN_NAME}, {config.GROUP_COLUMN_NAME}) not found. Using first two columns.")
        group_data_renamed = group_data.iloc[:, [0, 1]].copy()
        group_data_renamed.columns = ['gene', 'group']

    # Filter group_data to only include genes that are in selected_genes
    group_data_filtered = group_data_renamed[group_data_renamed['gene'].isin(selected_genes)]
    
    # Group by 'group' and collect 'genes'
    group_df = group_data_filtered.groupby('group')['gene'].apply(list).reset_index()
    group_df['sizes'] = group_df['gene'].apply(len)
    
    logger.info(f"Found {len(group_df)} groups with matching genes.")
    
    # Calculate group scores
    logger.info("Calculating group scores (Z-values)...")
    
    group_df['mean_t'] = group_df['gene'].apply(lambda genes: compute_group_sum(genes, t_gene))
    
    # Z-value calculation (matching original logic: sqrt(n) * Sum(t))
    group_df['z_value'] = np.sqrt(group_df['sizes']) * group_df['mean_t']
    
    # Calculate p-values
    group_df['p_value'] = 2 * (1 - norm.cdf(abs(group_df['z_value'])))
    
    # FDR Correction
    p_values = group_df['p_value'].values
    if len(p_values) > 0:
        _, q_values, _, _ = multipletests(p_values, method='fdr_bh')
        group_df['q_value'] = q_values
        
        # Filter by Q-Value
        significant_df_z = group_df[group_df['q_value'] < config.Q_VALUE_THRESHOLD]
        
        logger.info(f"Found {len(significant_df_z)} significant groups (q < {config.Q_VALUE_THRESHOLD}).")
        
        if len(significant_df_z) == 0:
            logger.warning("No significant groups found.")
            return input_data[[config.LABEL_COLUMN_NAME]] 
            
        # Extract unique features from significant groups
        all_group_elements = [item for sublist in significant_df_z['gene'] for item in sublist]
        unique_elements = sorted(set(all_group_elements))
        
        # Final Selection
        selected_features = [col for col in unique_elements if col in input_data.columns]
        
        logger.info(f"Final feature set size: {len(selected_features)}")
        
        new_df = pd.concat(
            [input_data[config.LABEL_COLUMN_NAME], input_data[selected_features]],
            axis=1
        )
        
        print("Shape of the new DataFrame:", new_df.shape)
        return new_df
    else:
        logger.warning("No groups to process.")
        return input_data

if __name__ == "__main__":
    run_zscore_pipeline()
