import os
import sys
import subprocess
import re
from pathlib import Path

def update_config_file(config_path: Path, new_data_path: str, q_value_threshold: float):
    """Updates the INPUT_EXPRESSION_DATA and Q_VALUE_THRESHOLD in the config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Use regex to find and replace the INPUT_EXPRESSION_DATA line
    content = re.sub(
        r'^INPUT_EXPRESSION_DATA\s*=\s*[\'"].*?[\'"]',
        f'INPUT_EXPRESSION_DATA = "{new_data_path}"',
        content,
        flags=re.MULTILINE
    )
    
    # Use regex to find and replace the Q_VALUE_THRESHOLD line
    content = re.sub(
        r'^Q_VALUE_THRESHOLD\s*=\s*[\d\.]+',
        f'Q_VALUE_THRESHOLD = {q_value_threshold}',
        content,
        flags=re.MULTILINE
    )
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    project_root = Path(__file__).resolve().parents[2]
    expression_data_dir = project_root / "data" / "expression_data"
    config_path = project_root / "src" / "workflows" / "GSM_zscore_workflow_config.py"
    workflow_script = project_root / "src" / "workflows" / "GSM_zscore_workflow.py"
    
    if not expression_data_dir.exists():
        print(f"Error: Directory {expression_data_dir} does not exist.")
        return

    # First run with 0.05, then run with 0.01
    q_value_thresholds = [0.05, 0.01]

    for threshold in q_value_thresholds:
        print(f"\n\n{'#'*80}")
        print(f"STARTING PIPELINE FOR Q_VALUE_THRESHOLD = {threshold}")
        print(f"{'#'*80}")
        
        # Iterate over all files in the expression data directory
        for file_path in expression_data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
                # Construct relative path formatted with forward slashes as expected by the config
                rel_path = f"data/expression_data/{file_path.name}"
                print(f"\n{'='*80}")
                print(f"Processing data file: {rel_path} | Q-Value: {threshold}")
                print(f"{'='*80}\n")
                
                # Update the configuration file
                update_config_file(config_path, rel_path, threshold)
                print(f"[info] Updated {config_path.name} with {rel_path} and Q_VALUE_THRESHOLD = {threshold}")
                
                # Run the workflow
                print(f"[run] Executing {workflow_script.name}...")
                try:
                    subprocess.run(
                        [sys.executable, str(workflow_script)],
                        check=True,
                        cwd=str(project_root)
                    )
                    print(f"[done] Finished processing {file_path.name} with Q-Value = {threshold}")
                except subprocess.CalledProcessError as e:
                    print(f"[error] Workflow execution failed for {file_path.name} (Q-Value = {threshold}): {e}")

if __name__ == "__main__":
    main()
