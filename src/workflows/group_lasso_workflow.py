"""
Group Lasso Workflow 🧬

Purpose:
    Implements a feature-selection workflow using the Group Lasso algorithm.
    Identifies important groups of genes (e.g., disease-associated gene sets)
    that are collectively predictive of a target phenotype.

    OVERLAP HANDLING:
    Standard Group Lasso requires each feature to belong to exactly one group.
    In DisGeNET grouping data, ~86 % of genes belong to multiple disease groups
    (TP53 alone appears in 1,319 groups).  This module solves the overlap by
    DUPLICATING features: a gene in N groups becomes N columns (one per group).
    After model fitting, duplicated coefficients are aggregated back to original
    gene names for reporting.

Key Functions:
    - group_lasso_workflow():       Main orchestrator — multi-iteration evaluation.
    - load_and_prepare_data():      Loads data, duplicates overlapping genes,
                                    builds non-overlapping group structure.
    - run_single_iteration():       One train/test split with scaling + model fit.
    - train_group_lasso():          Fits LogisticGroupLasso model.
    - evaluate_model():             Computes classification metrics on test data.
    - extract_selected_genes():     Maps expanded features → original gene names.
    - save_results():               Persists metrics, gene frequency, group info.

Example Usage:
    python -m src.workflows.group_lasso_workflow
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from group_lasso import LogisticGroupLasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# --- Project-specific imports ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.workflows.group_lasso_workflow_config import GroupLassoConfig
from src.data_processing.data_loader import _detect_separator
from src.utils.biological_validation import (
    query_enrichr,
    query_string_db,
    query_disgenet,
    save_enrichr_results,
    save_string_results,
    save_disgenet_results,
    save_validation_summary,
    save_biological_validation_explanation,
    ValidationReport,
)


##### CUSTOM EXCEPTIONS #####

class GroupLassoError(Exception):
    """Base exception for Group Lasso workflow errors."""


class DataLoadError(GroupLassoError):
    """Raised when input data cannot be loaded or validated."""


class GroupStructureError(GroupLassoError):
    """Raised when the grouping file has an invalid structure."""


##### DATA STRUCTURES #####

@dataclass
class FeatureDuplicationResult:
    """Result of duplicating overlapping features for Group Lasso.

    Stores the mapping metadata needed to convert between the expanded
    (duplicated) feature space and the original gene names.
    """
    groups: np.ndarray              # Group ID for each expanded column (no overlaps)
    expanded_names: list            # Column names like "TP53__Glioblastoma"
    original_gene_map: dict         # {expanded_name: original_gene_name}
    id_to_group: dict               # {group_id: group_name}
    group_to_id: dict               # {group_name: group_id}
    n_original_features: int        # Columns before duplication
    n_expanded_features: int        # Columns after duplication
    n_duplicated_genes: int         # Genes that appeared in >1 group
    n_groups_used: int              # Groups with at least one gene in expression data


@dataclass
class IterationMetrics:
    """Classification metrics from a single train/test iteration."""
    iteration: int
    seed: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    specificity: float
    auc_roc: float
    n_selected_features: int
    n_selected_groups: int


@dataclass
class AggregatedResults:
    """Aggregated results across all iterations."""
    iterations: List[IterationMetrics] = field(default_factory=list)
    mean_f1: float = 0.0
    std_f1: float = 0.0
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_accuracy: float = 0.0
    mean_selected_features: float = 0.0
    mean_selected_groups: float = 0.0


##### MAIN WORKFLOW #####

def group_lasso_workflow(
    config: GroupLassoConfig,
    logger: logging.Logger,
    output_dir: Path,
    *,
    n_iterations: int = 10,
) -> AggregatedResults:
    """
    Main workflow orchestrating the Group Lasso feature selection process.

    Runs multiple iterations with different random seeds to produce robust
    performance estimates, matching the GSM pipeline evaluation protocol.

    Args:
        config: GroupLassoConfig with file paths and hyperparameters.
        logger: Logger instance for tracking progress.
        output_dir: Directory where results will be saved.
        n_iterations: Number of random train/test splits (default: 10).

    Returns:
        AggregatedResults with per-iteration and summary metrics.
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING GROUP LASSO WORKFLOW")
    logger.info(f"   Iterations: {n_iterations} | Base seed: {config.random_seed}")
    logger.info("=" * 80)

    ##### STEP 1: Load and prepare data (once) #####
    logger.info("🔄 Loading and preparing data...")
    X, y, dup_result = load_and_prepare_data(config, logger)
    _log_duplication_summary(dup_result, logger)

    ##### STEP 2: Run iterations #####
    # Scaler is fit on train data INSIDE each iteration to avoid data leakage.
    all_metrics: List[IterationMetrics] = []
    all_selected_genes: List[List[str]] = []

    for i in range(n_iterations):
        iteration_seed = config.random_seed + i * 44
        logger.info(f"\n{'─' * 60}")
        logger.info(f"📊 Iteration {i + 1}/{n_iterations}  (seed={iteration_seed})")
        logger.info(f"{'─' * 60}")

        metrics, selected_genes = run_single_iteration(
            X=X,
            y=y,
            dup_result=dup_result,
            config=config,
            iteration=i + 1,
            iteration_seed=iteration_seed,
            logger=logger,
        )
        all_metrics.append(metrics)
        all_selected_genes.append(selected_genes)

    ##### STEP 3: Aggregate and save results #####
    results = _aggregate_results(all_metrics)
    _log_aggregated_results(results, logger)
    save_results(results, all_selected_genes, dup_result, output_dir, logger)

    ##### STEP 4: Biological validation (optional) #####
    if getattr(config, "run_biological_validation", False):
        try:
            top_genes = _get_top_genes_for_validation(
                all_selected_genes, config.biological_validation_top_genes,
            )
            if top_genes:
                run_gl_biological_validation(
                    top_genes=top_genes,
                    output_dir=output_dir,
                    config=config,
                    logger=logger,
                )
            else:
                logger.warning("⚠️  Bio validation skipped: no genes selected")
        except Exception as e:
            logger.warning(f"⚠️  Biological validation failed: {e}")

    logger.info("=" * 80)
    logger.info("🏁 GROUP LASSO WORKFLOW COMPLETE")
    logger.info("=" * 80)
    return results


##### DATA LOADING AND OVERLAP HANDLING #####

def load_and_prepare_data(
    config: GroupLassoConfig,
    logger: logging.Logger,
) -> tuple:
    """
    Load expression and grouping data, then build the duplication map.

    Group Lasso requires non-overlapping groups.  When a gene belongs to N
    disease groups we will later create N copies of its expression column —
    one per group — each assigned to exactly one group ID.  Here we build
    only the *metadata* (FeatureDuplicationResult); the actual expanded
    matrix is constructed per-iteration in ``build_expanded_matrix``.

    Returns:
        (X, y, dup_result): Raw expression DataFrame, target array, and
        FeatureDuplicationResult with the expansion instructions.
    """
    X, y = _load_expression_data(config, logger)
    group_df = _load_grouping_data(config, logger)
    dup_result = _build_duplication_map(X, group_df, config, logger)
    return X, y, dup_result


def _load_expression_data(
    config: GroupLassoConfig,
    logger: logging.Logger,
) -> tuple:
    """Load expression matrix and separate features from target."""
    logger.info(f"  📂 Loading main data from: {config.main_data_path}")
    try:
        sep = _detect_separator(config.main_data_path)
        main_df = pd.read_csv(config.main_data_path, sep=sep)
    except FileNotFoundError:
        raise DataLoadError(f"Main data file not found: {config.main_data_path}")

    # Strip whitespace and stray quotes from column names.
    main_df.columns = main_df.columns.str.strip().str.strip('"').str.strip()

    if config.target_column not in main_df.columns:
        raise DataLoadError(
            f"Target column '{config.target_column}' not found in data. "
            f"Available columns (first 5): {list(main_df.columns[:5])}..."
        )

    y_raw = main_df[config.target_column]
    X = main_df.drop(columns=[config.target_column])

    # Handle missing values.
    if config.missing_value_handling == "remove_rows":
        combined = pd.concat([X, y_raw], axis=1).dropna()
        X = combined.drop(columns=[config.target_column])
        y_raw = combined[config.target_column]
        logger.info(f"  Removed rows with NaN → {X.shape[0]} samples remain")
    elif config.missing_value_handling == "fill_mean":
        X = X.fillna(X.mean(numeric_only=True))
        logger.info("  Filled NaN with column means")

    # Convert class labels to 0/1.
    y, uniques = pd.factorize(y_raw)
    logger.info(f"  Classes: {list(uniques)} → mapped to {list(range(len(uniques)))}")
    logger.info(f"  Expression matrix: {X.shape[0]} samples × {X.shape[1]} genes")

    return X, y


def _load_grouping_data(
    config: GroupLassoConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and validate the gene-to-group mapping file."""
    logger.info(f"  📂 Loading grouping data from: {config.group_data_path}")
    try:
        sep = _detect_separator(config.group_data_path)
        group_df = pd.read_csv(config.group_data_path, sep=sep, header=0)
    except FileNotFoundError:
        raise DataLoadError(f"Group data file not found: {config.group_data_path}")

    # Strip whitespace and stray carriage returns from all string columns.
    for col in group_df.columns:
        if group_df[col].dtype == object:
            group_df[col] = group_df[col].str.strip().str.strip('\r')
    group_df.columns = [c.strip().strip('\r') for c in group_df.columns]

    # Validate structure.
    if config.group_name_column not in group_df.columns:
        raise GroupStructureError(
            f"Group column '{config.group_name_column}' not found. "
            f"Available: {list(group_df.columns)}"
        )
    if len(group_df.columns) != 2:
        raise GroupStructureError(
            f"Expected 2 columns in grouping file, got {len(group_df.columns)}: "
            f"{list(group_df.columns)}"
        )

    feature_col = [c for c in group_df.columns if c != config.group_name_column][0]
    logger.info(f"  Feature column: '{feature_col}' | Group column: '{config.group_name_column}'")
    logger.info(f"  Total gene-group pairs: {len(group_df):,}")

    return group_df


def _build_duplication_map(
    X: pd.DataFrame,
    group_df: pd.DataFrame,
    config: GroupLassoConfig,
    logger: logging.Logger,
) -> FeatureDuplicationResult:
    """
    Build the metadata for duplicating genes that belong to multiple groups.

    For each (gene, group) pair in the grouping file, if the gene exists in
    the expression matrix, register a column named ``gene__group`` assigned
    to that single group.  This converts overlapping groups into
    non-overlapping ones that Group Lasso can handle correctly.

    Filtering controls (from config):
    - **min_genes_per_group**: disease-groups with fewer genes present in the
      expression data are dropped (reduces noise and speeds training).
    - **max_groups_per_gene**: genes in more than N groups keep only the top-N
      assignments (sorted alphabetically by group name for determinism).

    The actual numeric matrix is built later per train/test split via
    ``build_expanded_matrix`` to keep memory usage under control.

    Args:
        X: Expression DataFrame (samples × genes).
        group_df: Two-column DataFrame (feature_id, group_name).
        config: Configuration with column names and filter thresholds.
        logger: Logger instance.

    Returns:
        FeatureDuplicationResult with the expansion metadata.
    """
    feature_col = [c for c in group_df.columns if c != config.group_name_column][0]
    available_genes = set(X.columns)

    # Keep only gene-group pairs where the gene has expression data.
    relevant = group_df[group_df[feature_col].isin(available_genes)].copy()
    logger.info(f"  Gene-group pairs with expression data: {len(relevant):,}")

    # --- Filter 1: remove small groups ---
    min_size = getattr(config, "min_genes_per_group", 0)
    if min_size > 0:
        group_sizes = relevant.groupby(config.group_name_column)[feature_col].nunique()
        kept_groups = set(group_sizes[group_sizes >= min_size].index)
        before = len(relevant)
        relevant = relevant[relevant[config.group_name_column].isin(kept_groups)]
        logger.info(
            f"  Filter min_genes_per_group >= {min_size}: "
            f"{len(kept_groups):,} groups kept, "
            f"{before - len(relevant):,} pairs removed"
        )

    # --- Filter 2: cap duplications per gene ---
    max_dup = getattr(config, "max_groups_per_gene", 0)
    if max_dup > 0:
        # Sort so the cap is deterministic (alphabetical by group name).
        relevant = relevant.sort_values(config.group_name_column)
        relevant = relevant.groupby(feature_col).head(max_dup).reset_index(drop=True)
        logger.info(
            f"  Filter max_groups_per_gene <= {max_dup}: "
            f"{len(relevant):,} pairs remain"
        )

    # Build group mappings from the *filtered* set.
    unique_groups = relevant[config.group_name_column].unique()
    group_to_id = {name: idx for idx, name in enumerate(unique_groups)}
    id_to_group = {idx: name for name, idx in group_to_id.items()}

    # Count how many genes sit in more than one group after filtering.
    gene_group_counts = relevant.groupby(feature_col)[config.group_name_column].nunique()
    n_multi = int((gene_group_counts > 1).sum())
    n_single = int((gene_group_counts == 1).sum())
    logger.info(f"  Genes in 1 group: {n_single:,} | Genes in >1 group: {n_multi:,}")

    # Build the expanded column list: one entry per (gene, group) pair.
    expanded_names: List[str] = []
    expanded_groups: List[int] = []
    original_gene_map: Dict[str, str] = {}

    for _, row in relevant.iterrows():
        gene = row[feature_col]
        group_name = row[config.group_name_column]
        col_name = f"{gene}__{group_name}"
        expanded_names.append(col_name)
        expanded_groups.append(group_to_id[group_name])
        original_gene_map[col_name] = gene

    groups_used = set(expanded_groups)

    return FeatureDuplicationResult(
        groups=np.array(expanded_groups),
        expanded_names=expanded_names,
        original_gene_map=original_gene_map,
        id_to_group=id_to_group,
        group_to_id=group_to_id,
        n_original_features=len(available_genes),
        n_expanded_features=len(expanded_names),
        n_duplicated_genes=n_multi,
        n_groups_used=len(groups_used),
    )


def build_expanded_matrix(
    X: pd.DataFrame,
    dup_result: FeatureDuplicationResult,
) -> np.ndarray:
    """
    Build the expanded feature matrix from original expression data.

    Each column in the expanded matrix is a copy of the original gene's
    expression values, placed into its assigned group slot.

    Args:
        X: Original expression DataFrame (samples × genes).
        dup_result: Duplication mapping from _build_duplication_map.

    Returns:
        np.ndarray of shape (n_samples, n_expanded_features).
    """
    n_samples = X.shape[0]
    n_cols = len(dup_result.expanded_names)
    expanded = np.empty((n_samples, n_cols), dtype=np.float64)

    for j, col_name in enumerate(dup_result.expanded_names):
        original_gene = dup_result.original_gene_map[col_name]
        expanded[:, j] = X[original_gene].values

    return expanded


def _log_duplication_summary(
    dup_result: FeatureDuplicationResult,
    logger: logging.Logger,
) -> None:
    """Log a human-readable summary of the feature duplication."""
    ratio = dup_result.n_expanded_features / max(dup_result.n_original_features, 1)
    logger.info("📋 Feature duplication summary:")
    logger.info(f"   Original genes in expression data: {dup_result.n_original_features:,}")
    logger.info(f"   Expanded columns (gene × group):   {dup_result.n_expanded_features:,}  ({ratio:.1f}×)")
    logger.info(f"   Genes duplicated (multi-group):     {dup_result.n_duplicated_genes:,}")
    logger.info(f"   Active disease groups:              {dup_result.n_groups_used:,}")


##### SINGLE ITERATION #####

def run_single_iteration(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    dup_result: FeatureDuplicationResult,
    config: GroupLassoConfig,
    iteration: int,
    iteration_seed: int,
    logger: logging.Logger,
) -> tuple:
    """
    Run one train/test iteration of the Group Lasso pipeline.

    Steps:
    1. Split data into train/test (stratified).
    2. Build expanded matrices and normalize (fit on train only).
    3. Train Group Lasso model.
    4. Evaluate on test set.
    5. Extract selected genes (aggregated from duplicates).

    Returns:
        (IterationMetrics, list of selected original gene names).
    """
    # Step 1: Stratified train/test split.
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=iteration_seed,
        stratify=y,
    )
    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    logger.info(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

    # Step 2: Build expanded matrices and normalize (scaler fit on train only).
    X_train_exp = build_expanded_matrix(X_train_raw, dup_result)
    X_test_exp = build_expanded_matrix(X_test_raw, dup_result)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_exp)
    X_test_scaled = scaler.transform(X_test_exp)

    # Step 3: Train model.
    model = train_group_lasso(
        X_train_scaled, y_train, dup_result.groups, config, logger,
    )

    # Step 4: Evaluate.
    metrics = evaluate_model(
        model, X_test_scaled, y_test,
        iteration=iteration,
        seed=iteration_seed,
        groups=dup_result.groups,
        logger=logger,
    )

    # Step 5: Extract selected original gene names.
    selected_genes = extract_selected_genes(model, dup_result, logger)

    return metrics, selected_genes


##### MODEL TRAINING #####

def train_group_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    config: GroupLassoConfig,
    logger: logging.Logger,
) -> LogisticGroupLasso:
    """Initialize and train the LogisticGroupLasso model.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training target array.
        groups: Group-ID array (one entry per expanded column).
        config: Hyperparameter configuration.
        logger: Logger instance.

    Returns:
        Fitted LogisticGroupLasso model.
    """
    model = LogisticGroupLasso(
        groups=groups,
        group_reg=config.group_reg,
        l1_reg=config.l1_reg,
        n_iter=config.n_iter,
        tol=config.tol,
        scale_reg=config.scale_reg,
        subsampling_scheme=config.subsampling_scheme,
        fit_intercept=config.fit_intercept,
        warm_start=config.warm_start,
        random_state=config.random_seed,
        supress_warning=True,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logger.info(f"  Model trained in {elapsed:.1f}s")

    return model


##### EVALUATION #####

def evaluate_model(
    model: LogisticGroupLasso,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    iteration: int,
    seed: int,
    groups: np.ndarray,
    logger: logging.Logger,
) -> IterationMetrics:
    """Compute classification metrics on the test set.

    Args:
        model: Fitted LogisticGroupLasso.
        X_test: Scaled test feature matrix.
        y_test: True test labels.
        iteration: Current iteration number.
        seed: Seed used for this iteration's split.
        groups: Group-ID array for counting selected groups.
        logger: Logger instance.

    Returns:
        IterationMetrics dataclass with all computed scores.
    """
    y_pred = model.predict(X_test)

    # AUC-ROC (from predicted probabilities when available).
    auc = _safe_auc(model, X_test, y_test)

    # Confusion matrix → specificity.
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Count selected features / groups from the sparsity mask.
    mask = model.sparsity_mask_
    n_sel_features = int(np.sum(mask))
    n_sel_groups = len(np.unique(groups[mask])) if n_sel_features > 0 else 0

    metrics = IterationMetrics(
        iteration=iteration,
        seed=seed,
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred, average='weighted'),
        precision=precision_score(y_test, y_pred, average='weighted', zero_division=0),
        recall=recall_score(y_test, y_pred, average='weighted', zero_division=0),
        specificity=specificity,
        auc_roc=auc,
        n_selected_features=n_sel_features,
        n_selected_groups=n_sel_groups,
    )

    logger.info(
        f"  F1={metrics.f1:.3f} | AUC={metrics.auc_roc:.3f} | "
        f"Acc={metrics.accuracy:.3f} | "
        f"Features={n_sel_features} | Groups={n_sel_groups}"
    )
    return metrics


def _safe_auc(
    model: LogisticGroupLasso,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Compute AUC-ROC safely, returning 0.0 on failure."""
    try:
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            return float(roc_auc_score(y_test, y_prob[:, 1]))
        return float(roc_auc_score(y_test, y_prob.ravel()))
    except (AttributeError, ValueError):
        return 0.0


##### GENE EXTRACTION & COEFFICIENT AGGREGATION #####

def extract_selected_genes(
    model: LogisticGroupLasso,
    dup_result: FeatureDuplicationResult,
    logger: logging.Logger,
) -> List[str]:
    """
    Map selected expanded features back to original gene names.

    After duplication, a gene like TP53 may have columns TP53__Glioblastoma,
    TP53__Leukemia, etc.  If *any* duplicate is selected, the original gene
    is considered selected.

    Returns:
        Sorted list of unique original gene names that were selected.
    """
    mask = model.sparsity_mask_
    if not np.any(mask):
        logger.info("  ⚠️  No features selected (all zeroed out).")
        return []

    selected_expanded = [
        name for name, sel in zip(dup_result.expanded_names, mask) if sel
    ]
    selected_originals = sorted(set(
        dup_result.original_gene_map[name] for name in selected_expanded
    ))
    logger.info(f"  Selected {len(selected_expanded)} expanded → {len(selected_originals)} unique genes")
    return selected_originals


def compute_gene_coefficients(
    model: LogisticGroupLasso,
    dup_result: FeatureDuplicationResult,
) -> pd.DataFrame:
    """
    Aggregate coefficients from duplicated columns back to original genes.

    For each original gene, sums the absolute coefficients across all its
    duplicated columns.

    Returns:
        DataFrame with columns: gene, coefficient_sum, n_groups_selected.
    """
    mask = model.sparsity_mask_
    empty = pd.DataFrame(columns=["gene", "coefficient_sum", "n_groups_selected"])
    if not np.any(mask):
        return empty

    coef = _extract_coef_vector(model, len(dup_result.expanded_names))

    gene_data: Dict[str, dict] = {}
    for j, (name, selected) in enumerate(zip(dup_result.expanded_names, mask)):
        if not selected:
            continue
        gene = dup_result.original_gene_map[name]
        if gene not in gene_data:
            gene_data[gene] = {"coef_sum": 0.0, "n_groups": 0}
        gene_data[gene]["coef_sum"] += abs(coef[j])
        gene_data[gene]["n_groups"] += 1

    rows = [
        {"gene": g, "coefficient_sum": d["coef_sum"], "n_groups_selected": d["n_groups"]}
        for g, d in gene_data.items()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("coefficient_sum", ascending=False).reset_index(drop=True)
    return df


def _extract_coef_vector(model: LogisticGroupLasso, n_features: int) -> np.ndarray:
    """Get a 1-D coefficient vector from the model (handles multi-class shapes)."""
    coef = model.coef_.flatten()
    if len(coef) == n_features:
        return coef
    coef_matrix = model.coef_
    if coef_matrix.shape[0] == n_features:
        coef_matrix = coef_matrix.T
    if coef_matrix.ndim > 1 and coef_matrix.shape[0] > 1:
        return coef_matrix[1]
    return coef_matrix.flatten()


##### AGGREGATION HELPERS #####

def _aggregate_results(metrics_list: List[IterationMetrics]) -> AggregatedResults:
    """Compute mean ± std across iterations."""
    f1s = [m.f1 for m in metrics_list]
    aucs = [m.auc_roc for m in metrics_list]
    accs = [m.accuracy for m in metrics_list]
    feats = [m.n_selected_features for m in metrics_list]
    groups = [m.n_selected_groups for m in metrics_list]

    return AggregatedResults(
        iterations=metrics_list,
        mean_f1=float(np.mean(f1s)),
        std_f1=float(np.std(f1s)),
        mean_auc=float(np.mean(aucs)),
        std_auc=float(np.std(aucs)),
        mean_accuracy=float(np.mean(accs)),
        mean_selected_features=float(np.mean(feats)),
        mean_selected_groups=float(np.mean(groups)),
    )


def _log_aggregated_results(results: AggregatedResults, logger: logging.Logger) -> None:
    """Log a summary table of aggregated results."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 AGGREGATED RESULTS")
    logger.info(f"   Mean F1:       {results.mean_f1:.4f} ± {results.std_f1:.4f}")
    logger.info(f"   Mean AUC-ROC:  {results.mean_auc:.4f} ± {results.std_auc:.4f}")
    logger.info(f"   Mean Accuracy: {results.mean_accuracy:.4f}")
    logger.info(f"   Mean Features: {results.mean_selected_features:.1f}")
    logger.info(f"   Mean Groups:   {results.mean_selected_groups:.1f}")
    logger.info("=" * 60)


##### SAVE RESULTS #####

def save_results(
    results: AggregatedResults,
    all_selected_genes: List[List[str]],
    dup_result: FeatureDuplicationResult,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Save all results to the output directory.

    Creates:
        - group_lasso_results.json   (summary + per-iteration metrics)
        - iteration_metrics.xlsx     (metrics table)
        - gene_selection_frequency.xlsx (genes ranked by selection count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON: full metrics + duplication info ---
    metrics_data = {
        "summary": {
            "mean_f1": results.mean_f1,
            "std_f1": results.std_f1,
            "mean_auc": results.mean_auc,
            "std_auc": results.std_auc,
            "mean_accuracy": results.mean_accuracy,
            "mean_selected_features": results.mean_selected_features,
            "mean_selected_groups": results.mean_selected_groups,
        },
        "iterations": [asdict(m) for m in results.iterations],
        "duplication_info": {
            "n_original_features": dup_result.n_original_features,
            "n_expanded_features": dup_result.n_expanded_features,
            "n_duplicated_genes": dup_result.n_duplicated_genes,
            "n_groups_used": dup_result.n_groups_used,
        },
    }
    json_path = output_dir / "group_lasso_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"💾 Saved metrics to: {json_path}")

    # --- Excel: per-iteration metrics ---
    metrics_df = pd.DataFrame([asdict(m) for m in results.iterations])
    excel_path = output_dir / "iteration_metrics.xlsx"
    metrics_df.to_excel(excel_path, index=False, engine="openpyxl")
    logger.info(f"💾 Saved iteration metrics to: {excel_path}")

    # --- Excel: gene selection frequency ---
    gene_freq: Dict[str, int] = {}
    for genes in all_selected_genes:
        for g in genes:
            gene_freq[g] = gene_freq.get(g, 0) + 1

    if gene_freq:
        freq_df = pd.DataFrame(
            sorted(gene_freq.items(), key=lambda x: -x[1]),
            columns=["gene", "selection_count"],
        )
        n_iters = len(all_selected_genes)
        freq_df["selection_fraction"] = freq_df["selection_count"] / n_iters
        freq_path = output_dir / "gene_selection_frequency.xlsx"
        freq_df.to_excel(freq_path, index=False, engine="openpyxl")
        logger.info(f"💾 Saved gene frequency to: {freq_path}")
    else:
        logger.warning("⚠️  No genes were selected in any iteration.")


##### BIOLOGICAL VALIDATION #####

def _get_top_genes_for_validation(
    all_selected_genes: List[List[str]],
    top_n: int,
) -> List[str]:
    """Return the top-N genes ranked by selection frequency across iterations.

    Args:
        all_selected_genes: Per-iteration lists of selected gene names.
        top_n: Maximum number of genes to return.

    Returns:
        List of gene names sorted by descending frequency.
    """
    gene_freq: Dict[str, int] = {}
    for genes in all_selected_genes:
        for g in genes:
            gene_freq[g] = gene_freq.get(g, 0) + 1

    sorted_genes = sorted(gene_freq.items(), key=lambda x: -x[1])
    return [gene for gene, _count in sorted_genes[:top_n]]


def run_gl_biological_validation(
    *,
    top_genes: List[str],
    output_dir: Path,
    config: GroupLassoConfig,
    logger: logging.Logger,
) -> ValidationReport:
    """Run Enrichr / STRING-db / DisGeNET validation on Group Lasso results.

    Uses the same API utilities as the GSM pipeline but takes the gene list
    directly from Group Lasso selection frequency rather than from RRA files.

    Args:
        top_genes: Ordered list of gene symbols to validate.
        output_dir: Run output directory (e.g. output/glasso_...).
        config: GroupLassoConfig (for DisGeNET API key, grouping path).
        logger: Logger instance.

    Returns:
        ValidationReport dataclass with all query results.
    """
    logger.info("🧬 Running biological validation on top GL-selected genes...")
    logger.info(f"   Genes to validate ({len(top_genes)}): {', '.join(top_genes[:5])}...")

    validation_dir = output_dir / "biological_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Save explanation file.
    save_biological_validation_explanation(output_dir, logger)

    report = ValidationReport(input_genes=top_genes)

    # --- Enrichr ---
    try:
        enrichr_results = query_enrichr(top_genes, logger)
        report.enrichr_results = enrichr_results
        save_enrichr_results(enrichr_results, validation_dir, logger)
        logger.info(f"  ✅ Enrichr: {len(enrichr_results)} enrichment terms found")
    except Exception as e:
        logger.warning(f"  ⚠️ Enrichr query failed: {e}")

    # --- STRING-db ---
    try:
        string_data = query_string_db(top_genes, logger)
        report.string_interactions = string_data["interactions"]
        report.string_network_url = string_data["network_url"]
        save_string_results(string_data, validation_dir, logger)
        logger.info(f"  ✅ STRING-db: {len(report.string_interactions)} interactions")
    except Exception as e:
        logger.warning(f"  ⚠️ STRING-db query failed: {e}")

    # --- DisGeNET ---
    api_key = getattr(config, "disgenet_api_key", "")
    if api_key:
        try:
            disease_assocs = query_disgenet(top_genes, api_key, logger)
            report.disease_associations = disease_assocs
            save_disgenet_results(disease_assocs, validation_dir, logger)
            logger.info(f"  ✅ DisGeNET: {len(disease_assocs)} disease associations")
        except Exception as e:
            logger.warning(f"  ⚠️ DisGeNET query failed: {e}")
    else:
        logger.info("  ℹ️  DisGeNET skipped (no API key provided)")

    # --- Summary report ---
    save_validation_summary(report, validation_dir, logger)
    logger.info(f"  💾 Validation results saved to: {validation_dir}")
    return report


##### SCRIPT ENTRY POINT #####

if __name__ == "__main__":
    config = GroupLassoConfig()

    # Create timestamped output directory.
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    project_root_path = Path(__file__).resolve().parents[2]
    output_dir = project_root_path / "output" / f"glasso_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger.
    log_file = output_dir / "group_lasso_workflow.log"
    logger = setup_logger(str(log_file))

    # Run the workflow.
    group_lasso_workflow(config, logger, output_dir, n_iterations=10)
