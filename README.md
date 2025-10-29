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

## Getting Started

### Setup Video Tutorial
For a visual walkthrough of setting up this project, watch our tutorial video:

[![Project Setup Tutorial](https://img.youtube.com/vi/brYpWo7VfK0/0.jpg)](https://www.youtube.com/watch?v=brYpWo7VfK0&feature=youtu.be)

**Video Link**: [GSM Pipeline Setup Tutorial](https://www.youtube.com/watch?v=brYpWo7VfK0&feature=youtu.be)

### Setting Up Your Environment
1. **Clone the repository**:
    ```sh
    git clone https://github.com/shiny-apricot/GSM-to-python.git
    cd GSM-to-python
    ```

2. **Create a virtual environment**:
   
   Using venv (Python's built-in virtual environment):
    ```sh
    # Create a virtual environment named 'venv'
    python -m venv venv
    
    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    
    # On macOS and Linux:
    source venv/bin/activate
    ```

   Using conda (if you prefer Anaconda/Miniconda):
    ```sh
    # Create a conda environment
    conda create -n gsm-env python=3.8
    
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
    python3 src/workflows/GSM_workflow.py
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
