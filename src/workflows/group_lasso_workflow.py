"""
Group Lasso Workflow 🧬

Purpose:
    This script implements a feature selection workflow using the Group Lasso algorithm.
    It is designed to identify important groups of features (e.g., genes belonging to the same pathway)
    that are collectively predictive of a target variable.

Key Functions:
    - group_lasso_workflow(): The main function that orchestrates the entire process.
    - load_and_prepare_data(): Loads the main dataset and grouping information, handles
      missing values, and creates the necessary data structures for the model.
    - train_group_lasso(): Initializes and trains the LogisticGroupLasso model.
    - log_results(): Evaluates the model, logs performance metrics, and saves the
      selected features and their group significance to output files.

Example Usage:
    To run the workflow with default settings, execute the script from the command line:
    python -m src.workflows.group_lasso_workflow
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from group_lasso import LogisticGroupLasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Project-specific imports ---
# Add the project root to the Python path to allow for absolute imports.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.workflows.group_lasso_workflow_config import GroupLassoConfig
from src.data_processing.data_loader import _detect_separator


##### MAIN WORKFLOW #####
def group_lasso_workflow(config: GroupLassoConfig, logger: logging.Logger, output_dir: Path):
    """
    Main workflow orchestrating the Group Lasso feature selection process.
    
    Steps:
    1. Sets the random seed for reproducibility.
    2. Loads and preprocesses the data.
    3. Splits the data into training and testing sets.
    4. Trains the Group Lasso model.
    5. Logs the results, including performance metrics and selected features/groups.
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING GROUP LASSO WORKFLOW")
    logger.info("=" * 80)

    # Step 1: Set random seed for consistent results across runs.
    np.random.seed(config.random_seed)
    logger.info(f"🎲 Random seed set to: {config.random_seed}")

    # Step 2: Load data from files and prepare it for the model.
    logger.info("🔄 Loading and preparing data...")
    X, y, groups, feature_names, id_to_group = load_and_prepare_data(config, logger)

    # Step 3: Split the dataset into training and testing sets.
    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_seed
    )
    logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples.")

    # Step 4: Train the Logistic Group Lasso model on the training data.
    logger.info("🤖 Training Group Lasso model...")
    model = train_group_lasso(X_train, y_train, groups, config, logger)

    # Step 5: Log performance and save the results to the output directory.
    logger.info("📝 Logging results and saving output files...")
    log_results(model, X_test, y_test, groups, feature_names, config, logger, output_dir, id_to_group)

    logger.info("=" * 80)
    logger.info("🏁 GROUP LASSO WORKFLOW COMPLETE")
    logger.info("=" * 80)

##### HELPER FUNCTIONS #####

def load_and_prepare_data(config: GroupLassoConfig, logger: logging.Logger):
    """
    Loads main and group data, preprocesses it, and creates the group structure.
    
    Returns:
        - X: Feature matrix (numpy array).
        - y: Target variable (numpy array).
        - groups: A numpy array where each element is the group ID for the corresponding feature.
        - feature_names: A list of feature names.
        - id_to_group: A dictionary mapping group IDs back to their original names.
    """
    # Load the main dataset containing features and target variable.
    logger.info(f"Loading main data from: {config.main_data_path}")
    try:
        sep = _detect_separator(config.main_data_path)
        main_df = pd.read_csv(config.main_data_path, sep=sep)
    except FileNotFoundError:
        logger.error(f"Main data file not found at: {config.main_data_path}")
        sys.exit(1)

    # Load the grouping data which defines feature-to-group relationships.
    logger.info(f"Loading group data from: {config.group_data_path}")
    try:
        sep = _detect_separator(config.group_data_path)
        group_df = pd.read_csv(config.group_data_path, sep=sep, header=0)
    except FileNotFoundError:
        logger.error(f"Group data file not found at: {config.group_data_path}")
        sys.exit(1)

    # Separate features (X) from the target variable (y).
    if config.target_column not in main_df.columns:
        logger.error(f"Target column '{config.target_column}' not found in main data.")
        sys.exit(1)
    y = main_df[config.target_column]
    X = main_df.drop(columns=[config.target_column])
    X.columns = X.columns.str.strip().str.strip('"')
    
    # Handle missing values based on the specified strategy.
    if config.missing_value_handling == "remove_rows":
        logger.info("Handling missing values by removing rows...")
        # Combine X and y to ensure rows are dropped consistently.
        combined_df = pd.concat([X, y], axis=1)
        combined_df.dropna(inplace=True)
        X = combined_df.drop(columns=[config.target_column])
        y = combined_df[config.target_column]
        logger.info(f"Shape of X after removing rows with missing values: {X.shape}")
    elif config.missing_value_handling == "fill_mean":
        logger.info("Handling missing values by filling with column mean...")
        X = X.fillna(X.mean())
    else:
        logger.warning(f"Unknown missing value handling strategy: {config.missing_value_handling}. Skipping.")

    # Convert categorical target labels into numerical format (e.g., 'neg'/'pos' -> 0/1).
    y, _ = pd.factorize(y)

    # --- Create Group Structure ---
    # Dynamically identify the feature and group columns from the grouping file.
    if config.group_name_column not in group_df.columns:
        logger.error(f"Group name column '{config.group_name_column}' not found in group data.")
        sys.exit(1)
    
    grouping_file_columns = group_df.columns.tolist()
    if len(grouping_file_columns) != 2:
        logger.error(f"Grouping file is expected to have 2 columns, but it has {len(grouping_file_columns)}.")
        sys.exit(1)
        
    feature_column = [col for col in grouping_file_columns if col != config.group_name_column][0]
    logger.info(f"Using '{feature_column}' as the feature column and '{config.group_name_column}' as the group column.")

    # Create mappings from group names to integer IDs and vice-versa.
    unique_groups = group_df[config.group_name_column].unique()
    group_to_id = {name: i for i, name in enumerate(unique_groups)}
    id_to_group = {i: name for name, i in group_to_id.items()}
    
    # log how many unique features were found
    logger.info(f"Identified {len(group_df[feature_column].unique())} unique features from the grouping file.")
    # log how many unique groups were found
    logger.info(f"Identified {len(unique_groups)} unique groups from the grouping file.")
    # log how many unique groups are in the data
    features_in_data = set(X.columns)
    groups_in_data = set(group_df[group_df[feature_column].isin(features_in_data)][
        config.group_name_column
    ].unique())
    logger.info(f"Out of these, {len(groups_in_data)} groups have features present in the main dataset.")

    # Map each feature to its corresponding group ID.
    feature_to_group_id = {}
    for _, row in group_df.iterrows():
        feature_to_group_id[row[feature_column]] = group_to_id.get(row[config.group_name_column])

    # Filter features in X to only include those present in the grouping file.
    grouped_features = [f for f in X.columns if f in feature_to_group_id]
    X_filtered = X[grouped_features]
    logger.info(f"Filtered features from {len(X.columns)} to {len(grouped_features)} that are in the grouping file.")

    # Normalize the feature data using Z-score scaling.
    logger.info("Normalizing data using Z-score scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    X_filtered = pd.DataFrame(X_scaled, index=X_filtered.index, columns=X_filtered.columns)
    logger.info("Data normalization complete.")

    # Create the final 'groups' array required by the GroupLasso model.
    feature_names = X_filtered.columns
    groups = [feature_to_group_id[feature] for feature in feature_names]
    groups = np.array(groups)

    logger.info(f"Loaded {X_filtered.shape[0]} samples with {X_filtered.shape[1]} features.")
    logger.info(f"Created {len(np.unique(groups))} groups for features.")

    return X_filtered.values, y, groups, feature_names.to_list(), id_to_group

def train_group_lasso(X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray, config: GroupLassoConfig, logger: logging.Logger) -> LogisticGroupLasso:
    """Initializes and trains the LogisticGroupLasso model."""
    # Initialize the model with parameters from the configuration.
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
        supress_warning=True
    )
    
    # Fit the model to the training data.
    logger.info("Fitting the LogisticGroupLasso model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    logger.info(f"Model training completed in {end_time - start_time:.2f} seconds.")
    
    return model

def log_results(model: LogisticGroupLasso, X_test: np.ndarray, y_test: np.ndarray, groups: np.ndarray, feature_names: List[str], config: GroupLassoConfig, logger: logging.Logger, output_dir: Path, id_to_group: Dict[int, str]):
    """
    Logs model performance, saves selected features, and saves group significance.
    """
    # --- Model Performance Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted') # Sensitivity
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # --- Calculate Specificity ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info("-" * 40)
    logger.info("📊 MODEL PERFORMANCE")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall (Sensitivity): {recall:.4f}")
    logger.info(f"  Specificity: {specificity:.4f}")
    logger.info("  Confusion Matrix:")
    logger.info(f"    {cm}")
    logger.info("-" * 40)

    # --- Save Confusion Matrix ---
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    cm_output_path = output_dir / "confusion_matrix.xlsx"
    try:
        cm_df.to_excel(cm_output_path, index=True, engine='openpyxl')
        logger.info(f"💾 Saved confusion matrix to: {cm_output_path}")
    except IOError as e:
        logger.error(f"❌ Failed to save confusion matrix: {e}")
        
    # --- Save Performance Metrics ---
    performance_metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
    }
    performance_df = pd.DataFrame([performance_metrics])
    performance_output_path = output_dir / "performance_metrics.xlsx"
    try:
        performance_df.to_excel(performance_output_path, index=False, engine='openpyxl')
        logger.info(f"💾 Saved performance metrics to: {performance_output_path}")
    except IOError as e:
        logger.error(f"❌ Failed to save performance metrics: {e}")

    # --- Save Classification Report ---
    report_df = pd.DataFrame(report).transpose()
    report_output_path = output_dir / "classification_report.xlsx"
    try:
        report_df.to_excel(report_output_path, index=True, engine='openpyxl')
        logger.info(f"💾 Saved classification report to: {report_output_path}")
    except IOError as e:
        logger.error(f"❌ Failed to save classification report: {e}")
    
    # --- Feature and Group Selection Analysis ---
    selected_features_mask = model.sparsity_mask_
    n_selected_features = np.sum(selected_features_mask)
    
    logger.info("🔍 FEATURE SELECTION RESULTS")
    logger.info(f"  Total features: {len(selected_features_mask)}")
    logger.info(f"  Selected features: {n_selected_features}")
    
    if n_selected_features > 0:
        selected_groups = np.unique(groups[selected_features_mask])
        logger.info(f"  Selected group length: {len(selected_groups)}")

        # --- Coefficient Extraction ---
        # The shape of `coef_` can vary, so we handle different cases to get a 1D array.
        coef_matrix = model.coef_
        if coef_matrix.shape[0] == len(feature_names):
            coef_matrix = coef_matrix.T
        if coef_matrix.ndim > 1 and coef_matrix.shape[0] > 1:
            coefficients = coef_matrix[1]  # Positive class coefficients for binary classification
        else:
            coefficients = coef_matrix.flatten()

        # Define a scaling factor for coefficients and significance.
        scaling_factor = 1_000_000
        logger.info(f"📈 Scaling coefficients and group significance by a factor of {scaling_factor:,}.")

        # --- Save Selected Features ---
        selected_feature_names = [name for name, selected in zip(feature_names, selected_features_mask) if selected]
        selected_coeffs = coefficients[selected_features_mask] * scaling_factor
        features_df = pd.DataFrame({"feature": selected_feature_names, "coefficient": selected_coeffs})
        
        # Sort features by the absolute value of their coefficient to rank by importance.
        features_df['abs_coefficient'] = features_df['coefficient'].abs()
        features_df = features_df.sort_values(by='abs_coefficient', ascending=False).drop(columns=['abs_coefficient'])
        
        output_path = output_dir / "selected_features.xlsx"
        try:
            features_df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"💾 Saved {len(features_df)} selected features to: {output_path}")
        except IOError as e:
            logger.error(f"❌ Failed to save selected features: {e}")

        # --- Calculate and Save Group Significance ---
        group_significance = []
        for group_id in np.unique(groups):
            if group_id == -1:  # Skip the un-regularized group
                continue
            
            group_mask = (groups == group_id)
            num_features_in_group = np.sum(group_mask)
            group_features_mask = selected_features_mask & group_mask
            
            # Calculate significance only if at least one feature from the group was selected.
            if np.any(group_features_mask):
                group_coeffs = coefficients[group_features_mask]
                # Significance is the sum of absolute coefficients of selected features in the group.
                significance = np.sum(np.abs(group_coeffs)) * scaling_factor
                group_name = id_to_group.get(group_id, f"Unknown Group {group_id}")
                group_significance.append((group_name, significance, num_features_in_group))

        if group_significance:
            group_df = pd.DataFrame(group_significance, columns=['group_name', 'significance', 'feature_count'])
            group_df = group_df.sort_values(by='significance', ascending=False)
            
            group_output_path = output_dir / "group_significance.xlsx"
            try:
                group_df.to_excel(group_output_path, index=False, engine='openpyxl')
                logger.info(f"💾 Saved group significance to: {group_output_path}")
            except IOError as e:
                logger.error(f"❌ Failed to save group significance: {e}")

    else:
        logger.info("  No features or groups were selected.")

if __name__ == "__main__":
    # This block runs when the script is executed directly.
    config = GroupLassoConfig()
    
    # Create a unique, timestamped directory for this run to store outputs.
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output" / f"glasso_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the logger to save logs to a file within the run-specific directory.
    log_file = output_dir / "group_lasso_workflow.log"
    logger = setup_logger(str(log_file))

    # Create the configuration object and execute the main workflow.
    
    group_lasso_workflow(config, logger, output_dir)
