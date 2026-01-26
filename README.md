# G-S-M Bioinformatics Data Pipeline Project

## Purpose
Implementation of Grouping-Scoring-Modeling (GSM) pipeline for gene analysis. This pipeline helps researchers analyze gene expression data to identify meaningful patterns and make predictions.

## Project Overview
This project implements a modular data pipeline for bioinformatics analysis using the GSM approach:
1. **Load**: Load and validate input data
2. **Filter**: Preliminary gene filtering using t-test
3. **Group**: Group genes based on gene-group coupling data
4. **Score**: Evaluate groups using ML models
5. **Model**: Train ML models on best-performing groups
6. **Predict**: Generate predictions using trained models

## Documentation & Guides

We have detailed documentation available in the `DOCS/` folder:

- **[Installation Guide (WSL/Linux)](DOCS/INSTALL_WSL.md)**: Step-by-step setup instructions.
- **[Running the Pipeline](DOCS/RUNNING.md)**: How to use the Web UI and scripts.
- **[Development Guide](DOCS/DEVELOPMENT.md)**: Project structure and coding standards.
- **[GitHub Workflow](DOCS/GITHUB_WORKFLOW.md)**: How to contribute, branch, and submit PRs.
- **[GitHub Copilot Pro Guide](DOCS/COPILOT.md)**: How to install Copilot Pro and use it for coding.
- **[Troubleshooting](DOCS/TROUBLESHOOTING.md)**: Common fixes for installation and runtime errors.

## VS Code Extensions
Recommended extensions for this repo are listed in [EXTENSIONS.txt](EXTENSIONS.txt).

## Getting Started (Quick)

### Setup Video Tutorial
For a visual walkthrough of setting up this project, watch our tutorial video:

[![Project Setup Tutorial](https://img.youtube.com/vi/brYpWo7VfK0/0.jpg)](https://www.youtube.com/watch?v=brYpWo7VfK0&feature=youtu.be)

**Video Link**: [GSM Pipeline Setup Tutorial](https://www.youtube.com/watch?v=brYpWo7VfK0&feature=youtu.be)

### Quick Setup
1. **Clone the repository**:
    ```sh
    git clone https://github.com/shiny-apricot/GSM-to-python.git
    cd GSM-to-python
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r dependencies.txt
    ```

4. **Run the pipeline**:
    ```sh
    # Quick test (recommended first)
    python run_test.py
    
    # Full run with config settings
    python src/workflows/GSM_workflow.py
    
    # Batch run on all datasets
    python run_all_datasets.py --iterations 100
    ```

Optional UI (if present):
```sh
streamlit run src/ui/app.py
```

For detailed setup, see the [Installation Guide](DOCS/INSTALL_WSL.md).

## Running Options

### 1. Quick Test
```sh
python run_test.py                    # Test with sample data (3 iterations)
python run_test.py --iterations 5     # Custom iteration count
python run_test.py --real-data        # Test with real GDS2545 data
```

### 2. Single Dataset Run
Edit `src/workflows/GSM_workflow_config.py` to configure your data and settings, then:
```sh
python src/workflows/GSM_workflow.py
```

### 3. Batch Run (Multiple Datasets)
```sh
python run_all_datasets.py                    # All datasets, 100 iterations each
python run_all_datasets.py --iterations 50    # Custom iterations
python run_all_datasets.py --datasets GDS2545 GDS3257  # Specific datasets
python run_all_datasets.py --list             # List available datasets
```

For long-running batch jobs, use `screen`:
```sh
screen -S gsm_batch
python run_all_datasets.py --iterations 100
# Press Ctrl+A, then D to detach
screen -r gsm_batch  # To reattach
```

## Usage

### Running from Command Line
1. **Activate your virtual environment** (if not already active):
    ```sh
    # Windows
    venv\Scripts\activate
    
    # macOS and Linux
    source venv/bin/activate
    ```

2. **Run the pipeline**:
    ```sh
    python src/workflows/GSM_workflow.py
    ```
    **If it gives error, then try this:**
    ```
    python -m src.workflows.GSM_workflow
    ```

### Outputs
Each run creates a timestamped folder under `output/` containing:

**Core Results:**
- `summary_report.txt` - Human-readable summary with best results
- `modeling_results_all_iterations.json` - Detailed JSON results
- `modeling_results_statistics.xlsx` - Model performance statistics

**Rankings:**
- `ranked_groups_all_iterations.xlsx` - Group rankings per iteration (by F1 score)
- `ranked_features_individual_all_iterations.xlsx` - Feature rankings per iteration (by ML importance)
- `ranked_features_group_derived_all_iterations.xlsx` - Feature rankings per iteration (by group F1 score)
- `aggregated_group_ranking_rra.xlsx` - Robust Rank Aggregated groups
- `aggregated_feature_ranking_individual_rra.xlsx` - RRA features (ranked by ML importance)
- `aggregated_feature_ranking_group_derived_rra.xlsx` - RRA features (ranked by group F1) ⭐ **Recommended**
- `best_averaged_groups.xlsx` - Average group rankings
- `best_averaged_features.xlsx` - Average feature rankings

**Figures (in `figures/` subfolder):**
- Performance plots (boxplots, confidence intervals, heatmaps)
- Feature importance visualization
- Group count optimization analysis
- 15+ publication-ready figures

**Statistics Excel Files (in `figures/` subfolder):**
- `statistics_performance_by_groups.xlsx` - Detailed metrics by group count
- `statistics_feature_importance.xlsx` - Feature importance statistics
- `statistics_iteration_summary.xlsx` - Per-iteration performance
- `statistics_cv_confidence_intervals.xlsx` - Cross-validation and CI data

**Configuration:**
- `run_parameters.txt` - Actual parameters used for this run
- `config_used.py` - Copy of base config file
- `gsm_workflow.log` - Detailed run logs

**Optional (if enabled):**
- `biological_validation/` - Enrichr, STRING-db, DisGeNET results

## Documentation
- Maintain comprehensive docstrings
- Include usage examples
- Reference official library documentation
- Document data structures and workflows
- At the top of each file, include a summary of the file's purpose and functionality along with:
  - File's primary purpose and role in the pipeline, briefly
  - List of key functions/classes with brief descriptions
  - Usage examples where appropriate
  - Any important notes or caveats

## Troubleshooting

### Common Issues
1. **ImportError or ModuleNotFoundError**:
   - Verify your virtual environment is activated
   - Reinstall dependencies: `pip install -r dependencies.txt`

2. **Permission denied errors**:
   - Check file permissions: `chmod +x src/name_of_your_file`


## Contributing
1. Create a new branch
2. Make your changes
3. Commit your changes 
4. Push to the branch 
5. Create a new Pull Request

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please open an issue on the repository or contact the project maintainers.
