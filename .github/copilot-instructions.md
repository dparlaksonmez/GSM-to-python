# GSM Bioinformatics Pipeline — Development Guide 🧬

> **Stable conventions only** — This file contains coding standards,
> design principles, and documentation rules that rarely change.
>
> For the **project map** (every file, folder, key function, and current
> defaults) see **[`PROJECT_MAP.md`](../PROJECT_MAP.md)** in the repo root.
> That file is the living document that AI agents and contributors should
> update whenever files are added, removed, or renamed.

---

## Core Development Principles

### 1. Code Organization 🏗️
- Write modular, single-responsibility components
- Follow top-down development approach:
  1. Start with main pipeline function (gsm_pipeline.py)
  2. Implement major stage functions (filter.py, group.py, score.py, model.py)
  3. Break down each stage into smaller helper functions
  4. Create utility functions last
- Prefer pure functions over classes
- Use dataclasses for data structures and results (not for configuration)
- NEVER use dictionaries, tuples, or named tuples for data - always use dataclasses instead
- Do not create additonal file for dataclasses, define them in the same file where they are used
- If a function returns multiple values, use a dataclass to encapsulate them.

Additional guardrails:
- Keep functions short (<= 20 lines) with minimal nesting.
- Use clear, non-technical comments when explaining pipeline logic.
- Keep output files stable and backward-compatible unless stated.

Example of top-down hierarchy:
```python
# Level 1: Main Pipeline (gsm_pipeline.py)
def gsm_pipeline(input_data: GeneData, logger: Logger) -> Results:
    """Main pipeline orchestrating the entire GSM analysis."""
    filtered_data = filter_genes(input_data, logger)
    groups = group_genes(filtered_data, logger)
    scores = score_groups(groups, logger)
    return train_model(scores, logger)

# Level 2: Major Stage (group.py)
def group_genes(filtered_data: FilteredData, logger: Logger) -> list[GeneGroup]:
    """Major stage function for gene grouping."""
    normalized_data = normalize_expression(filtered_data)
    clusters = perform_clustering(normalized_data)
    return create_gene_groups(clusters)

# Level 3: Helper Functions (grouping/cluster_utils.py)
def perform_clustering(normalized_data: np.ndarray) -> np.ndarray:
    """Helper function for specific clustering logic."""
    # Implementation details
```

```python
from dataclasses import dataclass

@dataclass
class GeneGroup:
    name: str
    genes: list[str]
    score: float = 0.0
```

### 2. Code Clarity for Non-Python Experts
- Write self-documenting code
- Include detailed comments in plain English
- Add visual separators for code sections
```python
##### DATA PREPROCESSING #####
def preprocess_gene_data(data: GeneData, logger: Logger) -> ProcessedData:
    """Transform raw gene data into analysis-ready format.
    
    Example:
        raw_data = load_data("genes.csv")
        processed = preprocess_gene_data(raw_data, logger)
    """
    logger.info("🔄 Starting data preprocessing...")
```

### 3. Function Design Rules
- One function = one task
- Maximum 20 lines per function
- Maximum 2 levels of nesting
- Always use type hints
```python
def calculate_gene_score(
    gene_id: str,
    expression_data: np.ndarray,
    *,  # Force keyword arguments
    threshold: float = 0.05,
    logger: Logger
) -> float:
```

### 4. Error Handling & Logging
- Log all critical operations
- Use custom exceptions for domain-specific errors
```python
class GeneAnalysisError(Exception):
    """Base exception for gene analysis errors."""
    pass

def analyze_genes(genes: list[str], logger: Logger) -> GeneResults:
    try:
        logger.info(f"📊 Analyzing {len(genes)} genes...")
        # Analysis code here
    except ValueError as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise GeneAnalysisError(f"Gene analysis failed: {str(e)}")
```

### 5. Performance Optimization
- Use numpy for numerical operations
- Implement dask for large datasets
- Profile code regularly
```python
import dask.dataframe as dd

def process_large_dataset(file_path: str) -> dd.DataFrame:
    """Process gene expression data using dask for memory efficiency."""
    return dd.read_csv(file_path).map_partitions(process_partition)
```

## Documentation Standards

### File Headers
```python
"""
Gene Group Analysis Module 🧬

Purpose:
    Implements gene grouping algorithms based on expression patterns.

Key Functions:
    - group_genes(): Creates gene groups from expression data
    - score_groups(): Evaluates group significance
    - optimize_groups(): Refines group assignments

Example Usage:
    groups = group_genes(expression_data, logger)
    scores = score_groups(groups, logger)
"""
```

### Function Documentation
```python
def group_genes(
    expression_data: np.ndarray,
    *,
    min_size: int = 10,
    logger: Logger
) -> list[GeneGroup]:
    """Group genes based on expression patterns.

    Args:
        expression_data: Gene expression matrix (genes × samples)
        min_size: Minimum genes per group (default: 10)
        logger: Logger instance for tracking

    Returns:
        List of GeneGroup objects

    Example:
        >>> data = load_expression_data("data.csv")
        >>> groups = group_genes(data, min_size=15, logger=logger)
        >>> print(f"Found {len(groups)} gene groups")
    """
```

## Testing Requirements
- Write tests for all core functions
- Include edge cases
- Test with small datasets first
```python
def test_gene_grouping():
    """Test gene grouping with a small dataset."""
    test_data = load_test_data()
    groups = group_genes(test_data, min_size=5, logger=test_logger)
    assert len(groups) > 0, "Should create at least one group"
    assert all(len(g.genes) >= 5 for g in groups), "Groups should meet size requirement"
```

## Dependencies
Core: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, openpyxl
Bio-APIs: requests (Enrichr, STRING-db, DisGeNET)
UI: streamlit
Dev: pytest, ruff, pre-commit
Full list: `dependencies.txt`

---

## Key Conventions Quick-Reference
- **Default classifier**: Random Forest (seed = 44) for both scoring and modeling.
- **Iterations**: 100 per run, each with a new random seed.
- **FDR threshold**: α = 0.05, Welch t-test + Benjamini–Hochberg.
- **CV folds**: 5-fold stratified within each iteration.
- **Feature ranking**: Robust Rank Aggregation (Stuart et al.).
- **Excluded datasets**: GDS3268 (breast, 100 % zero-sig), GDS4206 (HCC, 92 % zero-sig). See `DATASET_EXCLUSIONS.md`.
- **Manuscripts**: Built programmatically via `scripts/build_manuscript_docx.py` from `reports_ARCHIVE/manuscript_data.json`.
- **Biological validation**: Enrichr + STRING-db + DisGeNET. Results in `output/<run>/biological_validation/`.

## When to Update PROJECT_MAP.md
Copilot (or any contributor) should update **`PROJECT_MAP.md`** (not this file) whenever:
- A new source file or folder is added/removed.
- A public function is renamed or its signature changes.
- Pipeline defaults (classifier, seed, FDR α, etc.) are altered.
- New scripts or workflow configs are introduced.

Update **this file** only when coding conventions, design principles, or
documentation standards change.