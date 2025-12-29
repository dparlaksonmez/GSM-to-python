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

2. **Install & Run**:
    ```sh
    # Create venv
    python3 -m venv venv
    source venv/bin/activate
    
    # Install deps
    pip install -r dependencies.txt
    
    # Run UI
    streamlit run src/ui/app.py
    ```

For detailed instructions, please refer to the [Installation Guide](DOCS/INSTALL_WSL.md).

    # Activate the conda environment
    conda activate gsm-env
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r dependencies.txt
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
