"""
Configuration settings for the GSM Bioinformatics Pipeline.

This module contains all configuration parameters used throughout the pipeline,
organized by processing stage. Each section contains related parameters with
detailed documentation.

Key Configuration Sections:
- Input/Output: Data file paths and formats
- Pipeline Control: High-level pipeline behavior settings
- Model Settings: ML model configuration
- Feature Selection: Parameters for feature filtering and selection
- Data Processing: Data normalization and sampling parameters
- Performance: Execution settings like parallelization
"""
# TODO: Should I keep config file as .py or convert to .yaml?

from typing import Literal
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Input/Output Configuration
# ============================================================================

# Get the project directory. This assumes the config.py file is in the 'src' directory.
project_dir = Path(__file__).resolve().parents[1]

NUMBER_OF_ITERATIONS = 1

### TEST DATA ###
INPUT_EXPRESSION_DATA = "data/test/test_main_data.csv"
INPUT_GROUP_DATA = "data/test/test_grouping_data.csv"
# # Group Settings
GROUP_COLUMN_NAME = "diseaseName"
GENE_COLUMN_NAME = "geneSymbol"

### REAL DATA ###
# INPUT_EXPRESSION_DATA = "data/main_data/GDS2545.csv"
# INPUT_EXPRESSION_DATA = "data/main_data/GDS1962.csv"

# INPUT_GROUP_DATA = "data/grouping_data/cancer-DisGeNET.txt"
# GROUP_COLUMN_NAME = "group_name"
# GENE_COLUMN_NAME = "feature_id"


OUTPUT_DIR = project_dir / "output"



# MIN_GENES_PER_GROUP = 10 #TODO: should I keep it?
# MAX_GENES_PER_GROUP = 1000

# ============================================================================
# Data Processing Settings
# ============================================================================

LABEL_COLUMN_NAME = 'class'
NORMALIZATION_METHOD = 'zscore'  # Options: 'minmax', 'zscore', 'robust'
TRAIN_TEST_SPLIT_RATIO = 0.7
CLASS_LABELS_POSITIVE = "pos"
CLASS_LABELS_NEGATIVE = "neg"
MIN_CLASS_BALANCE_RATIO = 0.5  # Minimum acceptable ratio between minority and majority classes 
    # (0.5 means classes can be at most 1:2)
SAMPLING_METHOD = 'undersampling'  # Options: 'undersampling', 'oversampling', 'hybrid'
# TODO: implement sampling methods

# ============================================================================
# Pipeline Control
# ============================================================================

RANDOM_SEED = 44
CROSS_VALIDATION_FOLDS = 5
SAVE_INTERMEDIATE_RESULTS = True

# ============================================================================
# Model Configuration
# ============================================================================

# Supported model types
ModelType = Literal['DecisionTree', 'RandomForest', 'SVM', 'KNN', 'MLP']

MODEL_NAME : ModelType = 'RandomForest'


# ============================================================================
# Feature Selection Settings
# ============================================================================

INITIAL_FEATURE_FILTER_SIZE = 1000  # 0 to disable initial filtering
BEST_GROUPS_TO_KEEP = 10
# MIN_VARIANCE_THRESHOLD = 0.01 #TODO: should I keep it?
SELECTION_METHOD = 't_test'  # Options: 't_test', 'f_test', 'mutual_info'

# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGING_OUTPUT_FILE = 'pipeline.log'
