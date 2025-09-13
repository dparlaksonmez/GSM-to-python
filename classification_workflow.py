"""
🤖 Classification_Workflow.py - Machine Learning Classification Pipeline

Purpose:
    Complete classification workflow implementing Monte Carlo Cross-Validation
    with multiple machine learning algorithms for gene expression analysis.
    Includes integrated data preprocessing capabilities.
    
Key Components:
    1. Data Preprocessing: Handles missing values and feature normalization
    2. Data Loading: Imports and preprocesses gene expression data
    3. Monte Carlo CV: Repeated random train/test splits for robust evaluation  
    4. Classification: Applies multiple ML algorithms from machine_learning folder
    5. Result    logger.info("📄 INPUT DATA INFORMATION:")
    logger.info(f"   📊 Dataset shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
    logger.info(f"   📁 Data type: {type(input_data).__name__}")
    logger.info(f"   🎯 Target column: '{target_column}'")
    
    # Add file path information if available from config
    if config is not None and hasattr(config, 'input_data_path'):
        logger.info(f"   📂 Source file: {config.input_data_path}")
        try:
            from pathlib import Path
            file_path = Path(config.input_data_path)
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / 1024**2
                logger.info(f"   📏 File size: {file_size_mb:.2f} MB")
                # Log file timestamp
                import datetime
                file_mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                logger.info(f"   🕒 File modified: {file_mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            pass  # Don't fail if we can't get file info
    
    # Log basic data quality information
    total_missing = input_data.isnull().sum().sum()
    missing_percentage = (total_missing / (input_data.shape[0] * input_data.shape[1])) * 100
    logger.info(f"   🔍 Missing values: {total_missing} ({missing_percentage:.2f}%)")
    
    # Log column information
    feature_columns = [col for col in input_data.columns if col != target_column]
    logger.info(f"   📏 Feature columns: {len(feature_columns)}")
    logger.info(f"   📈 Memory usage: {input_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")Aggregates metrics across all iterations and algorithms
    6. Excel Export: Saves comprehensive results to Excel files

Key Functions:
    🎯 classification_run: Main entry point - orchestrates the complete workflow
    � preprocess_classification_data: Applies missing value handling and normalization
    �🔄 monte_carlo_cv: Implements Monte Carlo cross-validation
    🤖 apply_all_classifiers: Runs all available ML algorithms
    📊 collect_results: Aggregates performance metrics
    📋 save_to_excel: Exports results to Excel files

Preprocessing Features:
    🗑️ Missing Value Handling: Drop rows or fill with mean/median/mode
    📊 Feature Normalization: Z-score, Min-Max, or Robust scaling
    ✅ Data Validation: Comprehensive input validation and error handling

Usage Example:
    >>> # Load input data
    >>> data = pd.read_csv("expression_data.csv")
    >>> logger = setup_logger()
    >>> 
    >>> # Configure and run classification workflow with preprocessing
    >>> classification_run(
    ...     input_data=data,
    ...     target_column="class",
    ...     n_iterations=100,
    ...     test_size=0.3,
    ...     apply_preprocessing=True,
    ...     missing_value_strategy="fill_mean",
    ...     normalization_method="zscore",
    ...     logger=logger
    ... )

Notes:
    - Ensures reproducibility through fixed random seeds
    - Implements comprehensive error handling and logging
    - Supports all algorithms in machine_learning folder
    - Exports detailed statistics and individual iteration results
    - Preprocessing is optional and configurable
"""

##### IMPORTS #####
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging

# Core ML and stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Configuration and utilities

from src.utils.logger import setup_logger
from classification_workflow_config import DEFAULT_CONFIG, ClassificationWorkflowConfig

# Machine Learning algorithms
from src.machine_learning.random_forest import train_random_forest, predict_random_forest
from src.machine_learning.decision_tree import train_decision_tree, predict_decision_tree
from src.machine_learning.xgboost import train_xgboost_classifier
from src.machine_learning.adaboost import train_adaboost, predict_adaboost
from src.machine_learning.logitboost import train_logitboost, predict_logitboost

# Data processing utilities
from src.data_processing.data_loader import load_input_file
from src.data_processing.data_preprocess import preprocess_data, determine_class_balance
from src.data_processing.normalization import normalize_data
from src.data_processing.handle_missing_values import drop_missing_values, fill_missing_values

# Use default values instead of importing from GSM config
OUTPUT_DIR = "output"
RANDOM_SEED = 42

##### DATA STRUCTURES #####
@dataclass
class ClassificationMetrics:
    """Comprehensive metrics for a single classification run."""
    algorithm: str
    iteration: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_time: float
    prediction_time: float
    confusion_matrix: List[List[int]] = field(default_factory=list)
    classification_report: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonteCarloResults:
    """Results from Monte Carlo Cross-Validation across all algorithms."""
    algorithm_metrics: Dict[str, List[ClassificationMetrics]]
    summary_statistics: Dict[str, Dict[str, float]]
    total_iterations: int
    algorithms_used: List[str]
    
@dataclass
class ClassificationConfig:
    """Configuration parameters for classification workflow."""
    n_iterations: int = 100
    test_size: float = 0.3
    random_seed: int = RANDOM_SEED
    stratify: bool = True
    target_column: str = "class"
    algorithms: List[str] = field(default_factory=lambda: [
        "RandomForest", "DecisionTree", "XGBoost", "AdaBoost", "LogitBoost"
    ])
    # Preprocessing parameters
    apply_preprocessing: bool = True
    missing_value_strategy: str = "drop"  # Options: "drop", "fill_mean", "fill_median", "fill_mode"
    normalization_method: str = "zscore"  # Options: "zscore", "minmax", "robust", "none"
    # Class balancing parameters
    apply_class_balancing: bool = False
    min_class_balance_ratio: float = 0.5
    sampling_method: str = "undersampling"  # Options: "undersampling", "oversampling"

##### UTILITY FUNCTIONS #####
def set_random_seed(seed: int, logger: logging.Logger) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"🎲 Random seed set to: {seed}")

def generate_iteration_seed(base_seed: int, iteration: int) -> int:
    """Generate unique seed for each iteration."""
    return base_seed + iteration * 1000

##### DATA PREPROCESSING #####
def preprocess_classification_data(
    data: pd.DataFrame,
    config: ClassificationConfig,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply preprocessing steps to the input data.
    
    Args:
        data: Input DataFrame with features and target column
        config: Configuration containing preprocessing parameters
        logger: Logger instance
        
    Returns:
        Preprocessed DataFrame
    """
    if not config.apply_preprocessing:
        logger.info("🚫 Preprocessing disabled - using raw data")
        return data
        
    logger.info("🔄 Starting data preprocessing...")
    processed_data = data.copy()
    
    # Step 1: Handle missing values
    logger.info(f"🔍 Checking for missing values: {processed_data.isnull().sum().sum()} total")
    
    if processed_data.isnull().sum().sum() > 0:
        if config.missing_value_strategy == "drop":
            logger.info("🗑️ Dropping rows with missing values...")
            processed_data = drop_missing_values(processed_data)
        elif config.missing_value_strategy.startswith("fill_"):
            strategy = config.missing_value_strategy.replace("fill_", "")
            logger.info(f"🔧 Filling missing values using {strategy} strategy...")
            processed_data = fill_missing_values(processed_data, strategy=strategy)
        
        logger.info(f"✅ After handling missing values: {processed_data.shape[0]} rows, {processed_data.isnull().sum().sum()} missing values")
    
    # Step 2: Normalize features (exclude target column)
    if config.normalization_method != "none":
        logger.info(f"📊 Applying {config.normalization_method} normalization...")
        
        # Separate features and target
        feature_columns = [col for col in processed_data.columns if col != config.target_column]
        features = processed_data[feature_columns]
        target = processed_data[config.target_column]
        
        # Apply normalization to features only
        try:
            # Create a temporary DataFrame with just features for normalization
            features_df = features.copy()
            normalized_features = features_df  # Initialize with original features
            
            # Apply normalization using the specific normalization function
            if config.normalization_method == "zscore":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                normalized_features = pd.DataFrame(
                    scaler.fit_transform(features_df),
                    columns=features_df.columns,
                    index=features_df.index
                )
            elif config.normalization_method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                normalized_features = pd.DataFrame(
                    scaler.fit_transform(features_df),
                    columns=features_df.columns,
                    index=features_df.index
                )
            elif config.normalization_method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                normalized_features = pd.DataFrame(
                    scaler.fit_transform(features_df),
                    columns=features_df.columns,
                    index=features_df.index
                )
            
            # Reconstruct the DataFrame with normalized features and original target
            processed_data = pd.concat([normalized_features, target], axis=1)
            logger.info(f"✅ Normalization completed using {config.normalization_method} method")
            
        except Exception as e:
            logger.warning(f"⚠️ Normalization failed: {str(e)}. Proceeding with unnormalized data.")
    
    # Step 3: Apply class balancing if enabled
    if config.apply_class_balancing:
        logger.info(f"⚖️ Applying class balancing using {config.sampling_method}...")
        try:
            processed_data = determine_class_balance(
                processed_data,
                logger=logger,
                min_class_balance_ratio=config.min_class_balance_ratio,
                sampling_method=config.sampling_method
            )
            logger.info(f"✅ Class balancing completed. Final data shape: {processed_data.shape}")
        except Exception as e:
            logger.warning(f"⚠️ Class balancing failed: {str(e)}. Proceeding with original data.")
    else:
        logger.info("⚖️ Class balancing disabled - maintaining original class distribution")
    
    logger.info(f"🎯 Preprocessing complete. Final data shape: {processed_data.shape}")
    return processed_data

##### MONTE CARLO CROSS-VALIDATION #####
def monte_carlo_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: ClassificationConfig,
    logger: logging.Logger
) -> MonteCarloResults:
    """
    Perform Monte Carlo Cross-Validation with multiple algorithms.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Configuration parameters
        logger: Logger instance
        
    Returns:
        MonteCarloResults containing all metrics and statistics
    """
    logger.info(f"🎯 Starting Monte Carlo CV with {config.n_iterations} iterations")
    logger.info(f"📊 Algorithms: {', '.join(config.algorithms)}")
    
    # Handle label encoding for string labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"🏷️ Original labels: {sorted(y.unique())} → Encoded labels: {np.unique(y_encoded).tolist()}")
    
    # Initialize results storage
    algorithm_metrics = {alg: [] for alg in config.algorithms}
    
    # Perform iterations
    for iteration in range(1, config.n_iterations + 1):
        logger.info(f"=" * 50)
        logger.info(f"🔄 Iteration {iteration}/{config.n_iterations}")
        
        # Set iteration-specific seed
        iteration_seed = generate_iteration_seed(config.random_seed, iteration)
        set_random_seed(iteration_seed, logger)
        
        # Split data using encoded labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=config.test_size,
            stratify=y_encoded if config.stratify else None,
            random_state=iteration_seed
        )
        
        logger.info(f"📈 Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Apply all algorithms
        iteration_results = apply_all_classifiers(
            X_train, X_test, y_train, y_test,
            config.algorithms, iteration, logger
        )
        
        # Store results
        for result in iteration_results:
            algorithm_metrics[result.algorithm].append(result)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(algorithm_metrics, logger)
    
    return MonteCarloResults(
        algorithm_metrics=algorithm_metrics,
        summary_statistics=summary_stats,
        total_iterations=config.n_iterations,
        algorithms_used=config.algorithms
    )

##### CLASSIFICATION ALGORITHMS #####
def apply_all_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    algorithms: List[str],
    iteration: int,
    logger: logging.Logger
) -> List[ClassificationMetrics]:
    """
    Apply all specified classification algorithms to the data.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        algorithms: List of algorithm names to apply
        iteration: Current iteration number
        logger: Logger instance
        
    Returns:
        List of ClassificationMetrics for each algorithm
    """
    results = []
    
    for algorithm in algorithms:
        try:
            logger.info(f"🤖 Training {algorithm}...")
            
            # Train and predict with timing
            start_time = time.time()
            model = train_algorithm(algorithm, X_train, y_train, logger)
            training_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = predict_algorithm(algorithm, model, X_test, logger)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_metrics(
                y_test, y_pred, algorithm, iteration,
                training_time, prediction_time, logger
            )
            
            results.append(metrics)
            logger.info(f"✅ {algorithm} completed - F1: {metrics.f1_score:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error with {algorithm}: {str(e)}")
            # Create empty metrics for failed algorithm
            failed_metrics = ClassificationMetrics(
                algorithm=algorithm, iteration=iteration,
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                auc_score=0.0, training_time=0.0, prediction_time=0.0
            )
            results.append(failed_metrics)
    
    return results

def train_algorithm(algorithm: str, X_train: pd.DataFrame, y_train: pd.Series, logger: logging.Logger):
    """Train the specified algorithm."""
    if algorithm == "RandomForest":
        return train_random_forest(X_train, y_train, random_state=42)
    elif algorithm == "DecisionTree":
        return train_decision_tree(X_train, y_train, random_state=42)
    elif algorithm == "XGBoost":
        return train_xgboost_classifier(X_train, y_train)
    elif algorithm == "AdaBoost":
        return train_adaboost(X_train, y_train, random_state=42)
    elif algorithm == "LogitBoost":
        return train_logitboost(X_train, y_train, random_state=42)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def predict_algorithm(algorithm: str, model, X_test: pd.DataFrame, logger: logging.Logger):
    """Make predictions with the specified algorithm."""
    if algorithm == "RandomForest":
        return predict_random_forest(model, X_test)
    elif algorithm == "DecisionTree":
        return predict_decision_tree(model, X_test)
    elif algorithm == "XGBoost":
        return model.predict(X_test)
    elif algorithm == "AdaBoost":
        return predict_adaboost(model, X_test)
    elif algorithm == "LogitBoost":
        return predict_logitboost(model, X_test)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

##### METRICS CALCULATION #####
def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    algorithm: str,
    iteration: int,
    training_time: float,
    prediction_time: float,
    logger: logging.Logger
) -> ClassificationMetrics:
    """Calculate comprehensive metrics for classification results."""
    
    # Basic metrics - convert to float explicitly
    # Since we're now using numeric labels (0, 1), we can use default parameters
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    
    # AUC score (if possible)
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        auc = 0.0
    
    # Confusion matrix
    try:
        cm_array = confusion_matrix(y_true, y_pred)
        cm = [[int(val) for val in row] for row in cm_array.tolist()]
    except Exception:
        cm = []
    
    # Classification report
    try:
        class_report = classification_report(y_true, y_pred, output_dict=True)
        if not isinstance(class_report, dict):
            class_report = {}
    except Exception:
        class_report = {}
    
    return ClassificationMetrics(
        algorithm=algorithm,
        iteration=iteration,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_score=auc,
        training_time=training_time,
        prediction_time=prediction_time,
        confusion_matrix=cm,
        classification_report=class_report
    )

##### RESULTS ANALYSIS #####
def calculate_summary_statistics(
    algorithm_metrics: Dict[str, List[ClassificationMetrics]],
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """Calculate summary statistics for each algorithm across all iterations."""
    
    logger.info("📊 Calculating summary statistics...")
    summary_stats = {}
    
    for algorithm, metrics_list in algorithm_metrics.items():
        if not metrics_list:
            continue
            
        # Extract metric values
        accuracies = [m.accuracy for m in metrics_list]
        precisions = [m.precision for m in metrics_list]
        recalls = [m.recall for m in metrics_list]
        f1_scores = [m.f1_score for m in metrics_list]
        auc_scores = [m.auc_score for m in metrics_list]
        train_times = [m.training_time for m in metrics_list]
        pred_times = [m.prediction_time for m in metrics_list]
        
        # Calculate statistics
        summary_stats[algorithm] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'training_time_mean': np.mean(train_times),
            'training_time_std': np.std(train_times),
            'prediction_time_mean': np.mean(pred_times),
            'prediction_time_std': np.std(pred_times)
        }
        
        logger.info(f"✅ {algorithm} - Mean F1: {summary_stats[algorithm]['f1_mean']:.4f} ± {summary_stats[algorithm]['f1_std']:.4f}")
    
    return summary_stats

##### EXCEL EXPORT FUNCTIONS #####
def save_to_excel(
    results: MonteCarloResults,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Save all results to Excel files with multiple sheets."""
    
    logger.info("📋 Saving results to Excel files...")
    
    # Create Excel writer
    excel_path = output_dir / "classification_results.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Summary Statistics
        save_summary_sheet(results, writer, logger)
        
        # Sheet 2: Detailed Results
        save_detailed_sheet(results, writer, logger)
        
        # Sheet 3: Algorithm Comparison
        save_comparison_sheet(results, writer, logger)
    
    logger.info(f"✅ Results saved to: {excel_path}")

def save_summary_sheet(results: MonteCarloResults, writer, logger: logging.Logger) -> None:
    """Save summary statistics to Excel sheet."""
    
    summary_df = pd.DataFrame(results.summary_statistics).T
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=True)
    logger.info("📊 Summary statistics sheet created")

def save_detailed_sheet(results: MonteCarloResults, writer, logger: logging.Logger) -> None:
    """Save detailed iteration results to Excel sheet."""
    
    detailed_results = []
    
    for algorithm, metrics_list in results.algorithm_metrics.items():
        for metrics in metrics_list:
            detailed_results.append({
                'Algorithm': metrics.algorithm,
                'Iteration': metrics.iteration,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1_Score': metrics.f1_score,
                'AUC_Score': metrics.auc_score,
                'Training_Time': metrics.training_time,
                'Prediction_Time': metrics.prediction_time
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
    logger.info("📋 Detailed results sheet created")

def save_comparison_sheet(results: MonteCarloResults, writer, logger: logging.Logger) -> None:
    """Save algorithm comparison to Excel sheet."""
    
    comparison_data = []
    
    for algorithm in results.algorithms_used:
        stats = results.summary_statistics.get(algorithm, {})
        comparison_data.append({
            'Algorithm': algorithm,
            'Mean_Accuracy': stats.get('accuracy_mean', 0),
            'Std_Accuracy': stats.get('accuracy_std', 0),
            'Mean_F1': stats.get('f1_mean', 0),
            'Std_F1': stats.get('f1_std', 0),
            'Mean_Precision': stats.get('precision_mean', 0),
            'Mean_Recall': stats.get('recall_mean', 0),
            'Mean_AUC': stats.get('auc_mean', 0),
            'Mean_Training_Time': stats.get('training_time_mean', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean_F1', ascending=False)
    comparison_df.to_excel(writer, sheet_name='Algorithm_Comparison', index=False)
    logger.info("🏆 Algorithm comparison sheet created")

##### MAIN WORKFLOW FUNCTION #####
def classification_run(
    input_data: pd.DataFrame,
    *,  # Force named parameters
    config: Optional[ClassificationWorkflowConfig] = None,
    target_column: str = "class",
    n_iterations: int = 100,
    test_size: float = 0.3,
    algorithms: Optional[List[str]] = None,
    random_seed: int = RANDOM_SEED,
    output_dir: Optional[Path] = None,
    logger_path: Optional[Path] = None,
    # Preprocessing parameters
    apply_preprocessing: bool = True,
    missing_value_strategy: str = "drop",
    normalization_method: str = "zscore",
    # Class balancing parameters
    apply_class_balancing: bool = False,
    min_class_balance_ratio: float = 0.5,
    sampling_method: str = "undersampling"
) -> MonteCarloResults:
    """
    Main entry point for the classification workflow with integrated preprocessing.
    
    Args:
        input_data: Gene expression matrix with target column
        config: Configuration object (overrides individual parameters if provided)
        target_column: Name of the target/class column
        n_iterations: Number of Monte Carlo iterations
        test_size: Proportion of data for testing
        algorithms: List of algorithms to use (None for all)
        random_seed: Base random seed for reproducibility
        output_dir: Directory to save results
        logger_path: Path for log file
        apply_preprocessing: Whether to apply data preprocessing steps
        missing_value_strategy: Strategy for handling missing values ("drop", "fill_mean", "fill_median", "fill_mode")
        normalization_method: Method for feature normalization ("zscore", "minmax", "robust", "none")
        apply_class_balancing: Whether to apply class balancing to the dataset
        min_class_balance_ratio: Minimum acceptable ratio between minority and majority classes (0.5 means 1:2 ratio)
        sampling_method: Method for balancing classes ("undersampling", "oversampling")
        
    Returns:
        MonteCarloResults containing all metrics and statistics
        
    Example:
        >>> # Basic usage with preprocessing and class balancing
        >>> results = classification_run(
        ...     input_data=data,
        ...     target_column="disease_status",
        ...     apply_preprocessing=True,
        ...     missing_value_strategy="fill_mean",
        ...     normalization_method="zscore",
        ...     apply_class_balancing=True,
        ...     sampling_method="undersampling"
        ... )
    """
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / time.strftime("classification_%Y_%m_%d-%H_%M_%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    if logger_path is None:
        logger_path = output_dir / "classification_workflow.log"
    logger = setup_logger(str(logger_path))
    
    # Log output directory information
    logger.info("")
    logger.info("💾 OUTPUT CONFIGURATION:")
    logger.info(f"   📁 Output directory: {output_dir}")
    logger.info(f"   📝 Log file: {output_dir / 'classification_workflow.log'}")
    logger.info(f"   📊 Results will be saved to Excel files in output directory")
    
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING CLASSIFICATION WORKFLOW")
    logger.info("=" * 80)
    
    # Log input data information
    logger.info("📄 INPUT DATA INFORMATION:")
    logger.info(f"   📊 Dataset shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
    logger.info(f"   📁 Data type: {type(input_data).__name__}")
    logger.info(f"   🎯 Target column: '{target_column}'")
    
    # Log basic data quality information
    total_missing = input_data.isnull().sum().sum()
    missing_percentage = (total_missing / (input_data.shape[0] * input_data.shape[1])) * 100
    logger.info(f"   � Missing values: {total_missing} ({missing_percentage:.2f}%)")
    
    # Log column information
    feature_columns = [col for col in input_data.columns if col != target_column]
    logger.info(f"   📏 Feature columns: {len(feature_columns)}")
    logger.info(f"   📈 Memory usage: {input_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Use config object if provided, otherwise use individual parameters
    if config is not None:
        logger.info("")
        logger.info("⚙️ CONFIGURATION FROM CONFIG OBJECT:")
        logger.info(f"   📄 Input data path: {getattr(config, 'input_data_path', 'Not specified')}")
        logger.info(f"   🎯 Target column: {config.target_column}")
        logger.info(f"   🔄 Iterations: {config.n_iterations}")
        logger.info(f"   📈 Test size: {config.test_size}")
        logger.info(f"   🎲 Random seed: {config.random_seed}")
        logger.info(f"   🤖 Algorithms: {', '.join(config.algorithms)}")
        logger.info(f"   📊 Stratify: {getattr(config, 'stratify', 'Not specified')}")
        logger.info(f"   💾 Save to Excel: {getattr(config, 'save_to_excel', 'Not specified')}")
        logger.info(f"   🗣️ Verbose: {getattr(config, 'verbose', 'Not specified')}")
        
        # Extract class balancing parameters if available
        if hasattr(config, 'apply_class_balancing'):
            apply_class_balancing = config.apply_class_balancing
            min_class_balance_ratio = config.min_class_balance_ratio
            sampling_method = config.sampling_method
            logger.info(f"   ⚖️ Apply class balancing: {apply_class_balancing}")
            logger.info(f"   📊 Min class balance ratio: {min_class_balance_ratio}")
            logger.info(f"   🔄 Sampling method: {sampling_method}")
        
        target_column = config.target_column
        n_iterations = config.n_iterations
        test_size = config.test_size
        random_seed = config.random_seed
        algorithms = config.algorithms if algorithms is None else algorithms
    else:
        logger.info("")
        logger.info("⚙️ CONFIGURATION FROM PARAMETERS:")
        logger.info(f"   🎯 Target column: {target_column}")
        logger.info(f"   🔄 Iterations: {n_iterations}")
        logger.info(f"   📈 Test size: {test_size}")
        logger.info(f"   🎲 Random seed: {random_seed}")
        selected_algorithms = algorithms if algorithms is not None else ["RandomForest", "DecisionTree", "XGBoost", "AdaBoost", "LogitBoost"]
        logger.info(f"   🤖 Algorithms: {', '.join(selected_algorithms)}")
    
    # Log preprocessing configuration
    logger.info("")
    logger.info("🔧 PREPROCESSING CONFIGURATION:")
    logger.info(f"   ✅ Apply preprocessing: {apply_preprocessing}")
    logger.info(f"   🗑️ Missing value strategy: {missing_value_strategy}")
    logger.info(f"   📊 Normalization method: {normalization_method}")
    logger.info(f"   ⚖️ Apply class balancing: {apply_class_balancing}")
    logger.info(f"   📊 Min class balance ratio: {min_class_balance_ratio}")
    logger.info(f"   🔄 Sampling method: {sampling_method}")
    
    # Record start time for performance tracking
    start_time = time.time()
    logger.info(f"   ⏰ Workflow started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Default algorithms
    if algorithms is None:
        algorithms = ["RandomForest", "DecisionTree", "XGBoost", "AdaBoost", "LogitBoost"]
    
    # Create internal configuration for monte_carlo_cv
    internal_config = ClassificationConfig(
        n_iterations=n_iterations,
        test_size=test_size,
        random_seed=random_seed,
        target_column=target_column,
        algorithms=algorithms,
        apply_preprocessing=apply_preprocessing,
        missing_value_strategy=missing_value_strategy,
        normalization_method=normalization_method,
        apply_class_balancing=apply_class_balancing,
        min_class_balance_ratio=min_class_balance_ratio,
        sampling_method=sampling_method
    )
    
    # Apply preprocessing if enabled
    logger.info("🔄 Preparing data...")
    if target_column not in input_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Apply preprocessing steps
    preprocessed_data = preprocess_classification_data(input_data, internal_config, logger)
    
    X = preprocessed_data.drop(target_column, axis=1)
    y = preprocessed_data[target_column]
    
    # Enhanced data information logging
    logger.info("")
    logger.info("📊 FINAL PROCESSED DATA SUMMARY:")
    logger.info(f"   📏 Feature matrix shape: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"   🎯 Target column: '{target_column}' with {len(y.unique())} unique classes")
    
    # Target distribution with percentages
    target_counts = y.value_counts().to_dict()
    target_percentages = y.value_counts(normalize=True).to_dict()
    logger.info("   📈 Target distribution:")
    for class_name, count in target_counts.items():
        percentage = target_percentages[class_name] * 100
        logger.info(f"      • {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Data type information
    numeric_features = X.select_dtypes(include=[np.number]).shape[1]
    categorical_features = X.select_dtypes(exclude=[np.number]).shape[1]
    logger.info(f"   🔢 Numeric features: {numeric_features}")
    logger.info(f"   📝 Categorical features: {categorical_features}")
    
    # Memory usage
    feature_memory_mb = X.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"   💾 Feature matrix memory: {feature_memory_mb:.2f} MB")    # Run Monte Carlo Cross-Validation
    results = monte_carlo_cv(X, y, internal_config, logger)
    
    # Save results to Excel
    save_to_excel(results, output_dir, logger)
    
    # Log final summary
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info("=" * 50)
    logger.info("🏁 Classification Workflow Complete!")
    logger.info(f"⏰ Total execution time: {execution_time:.2f} seconds")
    logger.info(f"📊 Total iterations: {results.total_iterations}")
    logger.info(f"🤖 Algorithms tested: {len(results.algorithms_used)}")
    
    # Print best performing algorithm
    best_algorithm = max(
        results.summary_statistics.keys(),
        key=lambda alg: results.summary_statistics[alg].get('f1_mean', 0)
    )
    best_f1 = results.summary_statistics[best_algorithm]['f1_mean']
    logger.info(f"🏆 Best algorithm: {best_algorithm} (F1: {best_f1:.4f})")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info("=" * 50)
    
    return results

if __name__ == "__main__":
    # Example usage with configuration file
    logger = None
    try:
        # Load configuration
        config = DEFAULT_CONFIG
        
        # Setup main logger
        logger = setup_logger("main_execution.log")
        logger.info(f"📂 Loading data from: {config.input_data_path}")
        
        # Load test data
        data = pd.read_csv(config.input_data_path)
        logger.info(f"✅ Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Run classification workflow using config object
        results = classification_run(
            input_data=data,
            config=config
        )
        
        print("✅ Classification workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if logger:
            logger.error(f"❌ Error: {str(e)}")
