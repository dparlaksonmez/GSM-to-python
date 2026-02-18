"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GSM PIPELINE CONFIGURATION FILE                           ║
║                                                                              ║
║  This is the MAIN configuration file for the GSM Bioinformatics Pipeline.   ║
║  Edit the values below to customize your analysis.                          ║
║                                                                              ║
║  📖 For detailed documentation, see: DOCS/RUNNING.md                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Literal
from pathlib import Path

# Auto-detect project directory (do not modify)
project_dir = Path(__file__).resolve().parents[2]


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         1. INPUT DATA SETTINGS                             ║
# ║                                                                            ║
# ║  Configure your input files here. The pipeline requires:                   ║
# ║    • Expression data: Gene expression matrix (samples × genes)             ║
# ║    • Grouping data: Gene-to-group mappings (e.g., from DisGeNET)           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  EXPRESSION DATA FILE                                                    │
# │  Format: CSV or TSV with samples as rows, genes as columns               │
# │  Must include a class label column (see LABEL_COLUMN_NAME below)         │
# └──────────────────────────────────────────────────────────────────────────┘
INPUT_EXPRESSION_DATA = "data/main_data/GDS2545.csv"
MAIN_DATA_FILE_SEPARATOR = "auto"       # "auto" detects separator; or use "," / "\t"

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  GROUPING DATA FILE                                                      │
# │  Format: Two columns - gene identifiers and group names                  │
# │  Example sources: DisGeNET, KEGG, GO, custom gene sets                   │
# └──────────────────────────────────────────────────────────────────────────┘
INPUT_GROUP_DATA = "data/grouping_data/cancer-DisGeNET_gedinet.txt"
GROUPING_FILE_SEPARATOR = "auto"        # "auto" detects separator; or use "," / "\t"

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  COLUMN NAMES                                                            │
# │  Specify the column names in your grouping file                          │
# └──────────────────────────────────────────────────────────────────────────┘
GENE_COLUMN_NAME = "feature_id"         # Column containing gene identifiers
GROUP_COLUMN_NAME = "group_name"        # Column containing group/pathway names

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  OUTPUT DIRECTORY                                                        │
# │  Results will be saved in timestamped subfolders                         │
# └──────────────────────────────────────────────────────────────────────────┘
OUTPUT_DIR = project_dir / "output"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      2. PIPELINE CONTROL SETTINGS                          ║
# ║                                                                            ║
# ║  Control how the pipeline runs: iterations, reproducibility, outputs       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ITERATIONS                                                              │
# │  More iterations = more robust results, but longer runtime               │
# │  Recommended: 50-100 for publication, 10 for testing                     │
# └──────────────────────────────────────────────────────────────────────────┘
NUMBER_OF_ITERATIONS = 100

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  REPRODUCIBILITY                                                         │
# │  Set random seed for reproducible results                                │
# └──────────────────────────────────────────────────────────────────────────┘
RANDOM_SEED = 44

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  CROSS-VALIDATION                                                        │
# │  Number of folds for K-fold cross-validation during group scoring        │
# └──────────────────────────────────────────────────────────────────────────┘
CROSS_VALIDATION_FOLDS = 3

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  SAVE OPTIONS                                                            │
# └──────────────────────────────────────────────────────────────────────────┘
SAVE_INTERMEDIATE_RESULTS = True        # Save detailed JSON results


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      3. DATA PROCESSING SETTINGS                           ║
# ║                                                                            ║
# ║  Configure how your data is processed and split                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  CLASS LABELS                                                            │
# │  Specify the column and values used for classification                   │
# └──────────────────────────────────────────────────────────────────────────┘
LABEL_COLUMN_NAME = "class"             # Column containing class labels
CLASS_LABELS_POSITIVE = "pos"           # Label for positive class (e.g., disease)
CLASS_LABELS_NEGATIVE = "neg"           # Label for negative class (e.g., control)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  NORMALIZATION                                                           │
# │  Options: 'zscore' (recommended), 'minmax', 'robust'                     │
# └──────────────────────────────────────────────────────────────────────────┘
NORMALIZATION_METHOD = "zscore"

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  TRAIN/TEST SPLIT                                                        │
# │  Proportion of data used for training (rest used for testing)            │
# └──────────────────────────────────────────────────────────────────────────┘
TRAIN_TEST_SPLIT_RATIO = 0.7            # 70% train, 30% test


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      4. CLASS BALANCING SETTINGS                            ║
# ║                                                                            ║
# ║  Handle imbalanced datasets by balancing class distributions               ║
# ║  Applied AFTER normalization, BEFORE the iteration loop                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ENABLE/DISABLE                                                          │
# │  Set to True to balance classes before training                          │
# └──────────────────────────────────────────────────────────────────────────┘
APPLY_CLASS_BALANCING = True

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  MINIMUM CLASS BALANCE RATIO                                             │
# │  If minority/majority ratio is below this, balancing is applied          │
# │  0.5 means classes can be at most 1:2 before triggering                  │
# └──────────────────────────────────────────────────────────────────────────┘
MIN_CLASS_BALANCE_RATIO = 0.5

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  SAMPLING METHOD                                                         │
# │  Options: 'undersampling' (recommended), 'oversampling'                  │
# │    • undersampling: Reduces majority class to match minority             │
# │    • oversampling:  Duplicates minority class to match majority          │
# └──────────────────────────────────────────────────────────────────────────┘
SAMPLING_METHOD = "undersampling"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      5. FEATURE SELECTION SETTINGS                         ║
# ║                                                                            ║
# ║  Control how genes are filtered and groups are selected                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  T-TEST FILTERING                                                        │
# │  Genes with p-value above threshold are excluded                         │
# │  Lower = more stringent filtering                                        │
# └──────────────────────────────────────────────────────────────────────────┘
TTEST_THRESHOLD = 0.05                  # P-value threshold (0.05 = 95% confidence)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  INITIAL FEATURE FILTER                                                  │
# │  Optional: Limit to top N features before grouping                       │
# │  Set to 0 to disable (use all genes passing t-test)                      │
# └──────────────────────────────────────────────────────────────────────────┘
INITIAL_FEATURE_FILTER_SIZE = 0         # 0 = disabled

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  GROUP SELECTION                                                         │
# │  Number of top-ranked groups to use for final model                      │
# │  Higher = more features, potentially better but risking overfitting      │
# └──────────────────────────────────────────────────────────────────────────┘
BEST_GROUPS_TO_KEEP = 10


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      6. MACHINE LEARNING MODEL                             ║
# ║                                                                            ║
# ║  Select and configure the classification model                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  MODEL SELECTION                                                         │
# │  Available options:                                                      │
# │    • 'RandomForest'  - Recommended (best biological coherence)           │
# │    • 'XGBoost'       - Fast alternative (2.7× faster, lower bio signal)  │
# │    • 'DecisionTree'  - Simple, interpretable                             │
# │    • 'SVM'           - Support Vector Machine                            │
# │    • 'KNN'           - K-Nearest Neighbors                               │
# │    • 'MLP'           - Neural Network (Multi-Layer Perceptron)           │
# └──────────────────────────────────────────────────────────────────────────┘
ModelType = Literal['DecisionTree', 'RandomForest', 'XGBoost', 'SVM', 'KNN', 'MLP']
MODEL_NAME: ModelType = "RandomForest"

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  SCORING MODEL (for group ranking)                                       │
# │  This model is used to RANK gene groups during the scoring phase.        │
# │  It does NOT affect the final prediction model (MODEL_NAME above).       │
# │    • 'RandomForest'  - Recommended (best bio coherence, highest PPI)     │
# │    • 'XGBoost'       - 2.7× faster, slightly lower bio signal            │
# │    • 'DecisionTree'  - Fastest (4.2× faster, but ~6% lower F1)          │
# └──────────────────────────────────────────────────────────────────────────┘
ScoringModelType = Literal['DecisionTree', 'RandomForest', 'XGBoost']
SCORING_MODEL: ScoringModelType = "RandomForest"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      7. BIOLOGICAL VALIDATION (Optional)                   ║
# ║                                                                            ║
# ║  Query external databases to validate your top genes biologically          ║
# ║  This adds ~30-60 seconds to runtime due to API calls                      ║
# ║                                                                            ║
# ║  Databases queried:                                                        ║
# ║    • Enrichr: Pathway enrichment (KEGG, GO, Reactome, WikiPathways)        ║
# ║    • STRING-db: Protein-protein interaction networks                       ║
# ║    • DisGeNET: Gene-disease associations (requires free API key)           ║
# ║                                                                            ║
# ║  Output files saved in: output/<run>/biological_validation/                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ENABLE/DISABLE                                                          │
# │  Set to True to run biological validation after pipeline completes       │
# └──────────────────────────────────────────────────────────────────────────┘
RUN_BIOLOGICAL_VALIDATION = True

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  VALIDATION SETTINGS                                                     │
# └──────────────────────────────────────────────────────────────────────────┘
BIOLOGICAL_VALIDATION_TOP_GENES = 20    # Number of top genes to validate

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  DisGeNET API KEY (Optional)                                             │
# │  Get your free key at: https://www.disgenet.org/api/#/Authorization      │
# │  Leave empty ("") to skip DisGeNET queries                               │
# └──────────────────────────────────────────────────────────────────────────┘
DISGENET_API_KEY = ""


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      8. LOGGING SETTINGS                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

LOGGING_LEVEL = "INFO"                  # Options: 'DEBUG', 'INFO', 'WARNING'
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      QUICK START PRESETS                                   ║
# ║                                                                            ║
# ║  Uncomment ONE of the presets below to quickly switch configurations       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ────────────────────────────────────────────────────────────────────────────
# PRESET: Quick Test (for debugging/development)
# ────────────────────────────────────────────────────────────────────────────
# INPUT_EXPRESSION_DATA = "data/test/test_main_data.csv"
# INPUT_GROUP_DATA = "data/test/test_grouping_data.csv"
# NUMBER_OF_ITERATIONS = 3
# BEST_GROUPS_TO_KEEP = 5

# ────────────────────────────────────────────────────────────────────────────
# PRESET: Full Publication Run (robust results)
# ────────────────────────────────────────────────────────────────────────────
# NUMBER_OF_ITERATIONS = 100
# CROSS_VALIDATION_FOLDS = 5
# RUN_BIOLOGICAL_VALIDATION = True
