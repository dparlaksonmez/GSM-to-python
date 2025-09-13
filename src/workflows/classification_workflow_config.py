"""
🔧 Classification Workflow Configuration

Purpose:
    Configuration parameters for the machine learning classification workflow.
    Contains all settings needed to run the Monte Carlo Cross-Validation pipeline.

Configuration Sections:
    - Data Configuration: Input data and target settings
    - Monte Carlo Settings: Cross-validation parameters
    - Algorithm Selection: ML algorithms to include in the workflow
    - Output Settings: Results saving and formatting options
    - Performance Settings: Random seeds and reproducibility

Usage Example:
    >>> from classification_workflow_config import ClassificationWorkflowConfig
    >>> config = ClassificationWorkflowConfig()
    >>> print(f"Running {config.n_iterations} iterations with {config.test_size} test split")
"""

##### IMPORTS #####
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

##### CONFIGURATION DATACLASS #####
@dataclass
class ClassificationWorkflowConfig:
    """
    Complete configuration for the classification workflow.
    
    This dataclass contains all parameters needed to run the Monte Carlo
    Cross-Validation pipeline with multiple machine learning algorithms.
    """
    
    ##### DATA CONFIGURATION #####
    # Input data file path (relative to project root)
    input_data_path: str = "data/test/test_main_data.csv"
    # input_data_path: str = "data/main_data/GDS2545.csv"

    
    # Target column name in the dataset
    target_column: str = "class"
    
    ##### MONTE CARLO CROSS-VALIDATION SETTINGS #####
    # Number of Monte Carlo iterations to perform
    n_iterations: int = 5
    
    # Proportion of data to use for testing (0.0 to 1.0)
    test_size: float = 0.3
    
    # Whether to stratify the train/test split (maintains class distribution)
    stratify: bool = True
    
    ##### CLASS BALANCING SETTINGS #####
    # Whether to apply class balancing to the dataset
    apply_class_balancing: bool = True
    
    # Minimum acceptable ratio between minority and majority classes
    # (0.5 means classes can be at most 1:2)
    min_class_balance_ratio: float = 0.5
    
    # Method for balancing classes
    # Options: 'undersampling', 'oversampling'
    sampling_method: str = 'undersampling'
    
    ##### ALGORITHM SELECTION #####
    # List of algorithms to include in the classification workflow
    # Available options: "RandomForest", "DecisionTree", "XGBoost", "AdaBoost", "LogitBoost"
    algorithms: List[str] = field(default_factory=lambda: [
        "RandomForest", 
        "DecisionTree", 
        "XGBoost", 
        "AdaBoost", 
        "LogitBoost"
    ])
    
    ##### PERFORMANCE SETTINGS #####
    # Base random seed for reproducibility
    random_seed: int = 42
    
    ##### OUTPUT SETTINGS #####
    # Whether to save detailed results to Excel files
    save_to_excel: bool = True
    
    # Whether to print detailed progress information
    verbose: bool = True

##### CONFIGURATION INSTANCE #####
# Default configuration instance that can be imported and used directly
DEFAULT_CONFIG = ClassificationWorkflowConfig()

##### CONFIGURATION PRESETS #####
@dataclass
class QuickTestConfig(ClassificationWorkflowConfig):
    """Quick test configuration with fewer iterations for development."""
    n_iterations: int = 10
    test_size: float = 0.2
    algorithms: List[str] = field(default_factory=lambda: ["RandomForest", "DecisionTree"])

@dataclass
class ComprehensiveConfig(ClassificationWorkflowConfig):
    """Comprehensive configuration with more iterations for final analysis."""
    n_iterations: int = 100
    test_size: float = 0.3
    algorithms: List[str] = field(default_factory=lambda: [
        "RandomForest", "DecisionTree", "XGBoost", "AdaBoost", "LogitBoost"
    ])

@dataclass
class ProductionConfig(ClassificationWorkflowConfig):
    """Production configuration with high iteration count for robust results."""
    n_iterations: int = 200
    test_size: float = 0.25
    algorithms: List[str] = field(default_factory=lambda: [
        "RandomForest", "DecisionTree", "XGBoost", "AdaBoost", "LogitBoost"
    ])

##### PRESET INSTANCES #####
QUICK_TEST_CONFIG = QuickTestConfig()
COMPREHENSIVE_CONFIG = ComprehensiveConfig()
PRODUCTION_CONFIG = ProductionConfig()