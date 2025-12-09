# GSM Bioinformatics Pipeline UI

This directory contains the user interface for the GSM Bioinformatics Pipeline.

## Prerequisites

Ensure you have installed the required dependencies:

```bash
pip install -r dependencies.txt
```

## Running the UI

To launch the web interface, run the following command from the project root directory:

```bash
streamlit run src/ui/app.py
```

## Features

- **Data Upload**: Upload your gene expression data (CSV) and group definitions (CSV/TXT).
- **Configuration**: Customize pipeline parameters such as:
    - Number of Iterations
    - Train/Test Split Ratio
    - Normalization Method
    - Machine Learning Model
- **Column Mapping**: Specify column names for genes, groups, and class labels.
- **Visualization**: View interactive plots of F1 scores and average performance.
- **Summary Report**: Get a detailed summary of the best performing model configuration.

## Input Data Format

- **Expression Data**: CSV file where rows are samples and columns are genes (or vice versa, depending on your preprocessing). Must contain a label column.
- **Group Data**: CSV or Tab-separated file mapping genes to groups.

## Output

Results are saved in the `output/` directory, organized by timestamp. The UI displays the results from the current run.
