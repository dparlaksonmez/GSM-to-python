"""
Group Lasso Workflow Configuration 🧬

Purpose:
    This file defines the configuration for the Group Lasso workflow using a dataclass.
    It centralizes all the parameters, making it easy to manage and modify the workflow's behavior.
"""

from dataclasses import dataclass
from typing import Union

# --- Constants ---
# Define a constant for the random seed to ensure reproducibility across the script.
RANDOM_SEED = 42

# --- Configuration ---

@dataclass
class GroupLassoConfig:
    """Configuration for the Group Lasso workflow."""
    # --- File Paths ---
    # Path to the main dataset containing features and the target variable.
    main_data_path: str = "data/main_data/GDS1962.csv"
    # Path to the file defining the groups for features.
    group_data_path: str = "data/grouping_data/cancer-DisGeNET_gedinet.txt"

    # --- Data Columns ---
    # Name of the column in the main data file to be used as the target variable.
    target_column: str = "class"
    # Name of the column in the group data file that defines the group names.
    group_name_column: str = "group_name"

    # --- Preprocessing ---
    # Strategy for handling missing values.
    # Options: "remove_rows", "fill_mean".
    missing_value_handling: str = "remove_rows"

    # --- Model Hyperparameters ---
    # Regularization parameter for group-level sparsity (lambda1).
    # Higher values lead to more groups being entirely removed.
    group_reg: float = 0.01
    # Regularization parameter for feature-level sparsity within groups (lambda2).
    # Higher values lead to more individual features being removed from the selected groups.
    l1_reg: float = 0.05
    # The maximum number of iterations to perform.
    n_iter: int = 1000
    # The convergence tolerance. The optimization will stop once the norm of the change in coefficients is less than this.
    tol: float = 1e-5
    # How to scale the group-wise regularization coefficients.
    # Options: "group_size", "none", "inverse_group_size".
    scale_reg: str = "group_size"
    # The subsampling rate for gradient and singular value computations. Can be a float (fraction), int (count), or 'sqrt'.
    subsampling_scheme: Union[None, float, int, str] = None
    # Whether to fit an intercept term in the model.
    fit_intercept: bool = True
    # If True, subsequent calls to fit will not re-initialize model parameters, speeding up hyperparameter search.
    warm_start: bool = False

    # --- General Settings ---
    # Seed for random number generators to ensure reproducibility.
    random_seed: int = RANDOM_SEED
    # Proportion of the dataset to be used for testing.
    test_size: float = 0.3
    # Base directory for all output files.
    output_dir: str = "output/group_lasso"
