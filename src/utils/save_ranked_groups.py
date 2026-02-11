"""
📊 Utility module for saving ranked group results to Excel files
Key functions:
    - save_ranked_groups: Saves group rankings with metrics to Excel (ranks calculated automatically)
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
import logging
from pathlib import Path

from src.scoring.metrics import MetricsData


def save_ranked_groups(
    output_path: str,
    group_list: List[MetricsData],
    iteration: int,
    logger: logging.Logger
) -> None:
    """
    Saves ranked groups and their metrics to an Excel file.
    Ranks are calculated automatically based on accuracies.
    
    Args:
        output_path: Path to save Excel file
        group_list: List of MetricsData objects containing group metrics
        iteration: Current iteration number
        logger: Logger instance
    """
    try:
        logger.debug("Saving ranked groups...")
        
        df = pd.DataFrame({
            'Group Name': [m.name for m in group_list],
            'Accuracy': [m.accuracy for m in group_list],
            'F1 Score': [m.f1 for m in group_list],
            'Iteration': [iteration] * len(group_list),
        })
        
        df = df.sort_values('Accuracy', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Group Name', 'Accuracy', 'F1 Score', 'Iteration']]
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_p = Path(output_path)
        # If CSV requested, use CSV writing (better for very large files)
        if output_p.suffix.lower() == '.csv':
            if output_p.exists():
                existing_df = pd.read_csv(output_p)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(output_p, index=False)
            logger.debug(f"Saved ranked groups: {output_p.name}")
        else:
            # Excel behavior (append if exists)
            if output_p.exists():
                existing_df = pd.read_excel(output_p)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_excel(output_p, index=False)
            logger.debug(f"Saved ranked groups: {output_p.name}")
        
    except Exception as e:
        logger.error(f"Failed to save ranked groups: {str(e)}")
        raise
