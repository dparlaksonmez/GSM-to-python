"""
GSM with Z-Score Pipeline
=========================

This module provides both:
1. A pure in-memory z-score filter that can be used inside a workflow.
2. A CLI-style convenience pipeline for standalone execution.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from src.workflows import GSM_zscore_workflow_config as config
from src.data_processing.data_loader import load_input_file, load_group_file
from src.data_processing.handle_missing_values import drop_missing_values
from src.data_processing.preliminary_filtering import preliminary_ttest_filter
from src.data_processing.data_preprocess import convert_labels_to_binary
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("GSM_ZScore")


def compute_group_sum(gene_list, t_gene_df):
    matched = t_gene_df[t_gene_df["gene"].isin(gene_list)]
    return matched["t_value"].sum()



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
            columns={gene_column_name: "gene", group_column_name: "group"}
        )
    else:
        logger.warning(
            f"Configured grouping columns ({gene_column_name}, {group_column_name}) not found. "
            "Using first two columns."
        )
        group_data_renamed = group_data.iloc[:, [0, 1]].copy()
        group_data_renamed.columns = ["gene", "group"]

    return group_data_renamed[group_data_renamed["gene"].isin(selected_genes)]



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
    Apply z-score based group filtering using already-computed t-test results.

    Returns:
        Tuple of (selected_feature_names, significant_group_df).
    """
    selected_indices = filter_results.selected_features
    if filter_results.selected_feature_names is not None:
        selected_genes = list(filter_results.selected_feature_names)
    else:
        selected_genes = [feature_names[idx] for idx in selected_indices]

    # Reuse upstream t-test statistics so we do not recompute filtering work here.
    t_stats = filter_results.statistics
    if t_stats is None or len(t_stats) == 0:
        logger.warning("No t-statistics available; skipping z-score group filtering.")
        return [], pd.DataFrame()

    selected_t_stats = t_stats[selected_indices]
    t_gene = pd.DataFrame({"gene": selected_genes, "t_value": selected_t_stats})

    group_data_filtered = _prepare_group_dataframe(
        group_data,
        gene_column_name,
        group_column_name,
        selected_genes,
        logger,
    )

    group_df = group_data_filtered.groupby("group")["gene"].apply(list).reset_index()
    if group_df.empty:
        logger.warning("No groups with selected genes found for z-score filtering.")
        return [], pd.DataFrame()

    group_df["sizes"] = group_df["gene"].apply(len)
    logger.info(f"Calculating group scores (Z-values) for {len(group_df)} groups...")
    group_df["t_sum"] = group_df["gene"].apply(lambda genes: compute_group_sum(genes, t_gene))
    group_df["z_value"] = np.sqrt(group_df["sizes"]) * group_df["t_sum"]
    group_df["p_value"] = 2 * (1 - norm.cdf(abs(group_df["z_value"])))

    p_values = group_df["p_value"].values
    _, q_values, _, _ = multipletests(p_values, method="fdr_bh")
    group_df["q_value"] = q_values

    significant_df_z = group_df[group_df["q_value"] < q_value_threshold].copy()
    logger.info(f"Found {len(significant_df_z)} significant groups (q < {q_value_threshold}).")

    if significant_df_z.empty:
        return [], significant_df_z

    all_group_elements = [item for sublist in significant_df_z["gene"] for item in sublist]
    selected_features = sorted(set(all_group_elements))
    return selected_features, significant_df_z



# This thin wrapper keeps the workflow path fully in-memory instead of reloading files per iteration.
def run_zscore_filter(
    filter_results,
    feature_names: List[str],
    group_data: pd.DataFrame,
    *,
    gene_column_name: str,
    group_column_name: str,
    q_value_threshold: float,
    logger,
) -> Tuple[List[str], pd.DataFrame]:
    """Public in-memory entry point for z-score filtering inside GSM workflows."""
    return zscore_group_filter(
        filter_results,
        feature_names,
        group_data,
        gene_column_name=gene_column_name,
        group_column_name=group_column_name,
        q_value_threshold=q_value_threshold,
        logger=logger,
    )



def run_zscore_pipeline():
    logger.info("Starting GSM Z-Score Pipeline...")

    logger.info(f"Loading expression data from {config.INPUT_EXPRESSION_DATA}...")
    input_data = load_input_file(config.INPUT_EXPRESSION_DATA, separator=config.MAIN_DATA_FILE_SEPARATOR)

    logger.info(f"Loading grouping data from {config.INPUT_GROUP_DATA}...")
    group_data = load_group_file(config.INPUT_GROUP_DATA, separator=config.GROUPING_FILE_SEPARATOR)

    logger.info("Handling missing values...")
    input_data = drop_missing_values(input_data)

    if config.LABEL_COLUMN_NAME not in input_data.columns:
        raise ValueError(f"Column '{config.LABEL_COLUMN_NAME}' not found in input data")

    input_data = convert_labels_to_binary(
        input_data,
        config.LABEL_COLUMN_NAME,
        config.CLASS_LABELS_NEGATIVE,
        config.CLASS_LABELS_POSITIVE,
    )

    X = input_data.drop(columns=[config.LABEL_COLUMN_NAME])
    y = input_data[config.LABEL_COLUMN_NAME]
    X = X.apply(pd.to_numeric, errors="coerce")

    logger.info("Performing preliminary filtering...")
    filter_results = preliminary_ttest_filter(
        X,
        y,
        logger=logger,
        threshold=config.TTEST_THRESHOLD,
    )

    selected_features, significant_df_z = run_zscore_filter(
        filter_results,
        list(X.columns),
        group_data,
        gene_column_name=config.GENE_COLUMN_NAME,
        group_column_name=config.GROUP_COLUMN_NAME,
        q_value_threshold=config.Q_VALUE_THRESHOLD,
        logger=logger,
    )

    if not selected_features:
        logger.warning("No significant groups found after z-score filtering.")
        return input_data[[config.LABEL_COLUMN_NAME]]

    logger.info(f"Final feature set size: {len(selected_features)}")
    if not significant_df_z.empty:
        logger.info(f"Retained {len(significant_df_z)} significant groups after z-score filtering.")

    new_df = pd.concat(
        [input_data[config.LABEL_COLUMN_NAME], input_data[selected_features]],
        axis=1,
    )

    print("Shape of the new DataFrame:", new_df.shape)
    return new_df


if __name__ == "__main__":
    run_zscore_pipeline()
