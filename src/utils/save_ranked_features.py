"""
Feature Ranking Output Module 📊

Purpose:
    Save ranked feature results to Excel format with scores and metadata.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
from pathlib import Path
import logging

@dataclass
class FeatureRankingOutput:
    """Container for feature ranking output configuration."""
    output_path: Path
    feature_scores: List
    timestamp: str
    model_name: str
    iteration: int

def save_ranked_features(
    ranking_data: FeatureRankingOutput,
    logger: logging.Logger
) -> None:
    """
    Save ranked features and their scores to Excel.

    Args:
        ranking_data: FeatureRankingOutput containing ranking results
        logger: Logger instance for tracking

    Raises:
        IOError: If file writing fails
    """
    try:
        # Update path extension if not .xlsx
        output_path = ranking_data.output_path.with_suffix('.xlsx')
        logger.info(f"💾 Saving ranked features to {output_path}")

        # Create DataFrame from feature scores
        results_df = pd.DataFrame([
            {
                'feature_name': score.feature_name,
                'f1_score': score.f1_score,
                'importance_score': score.importance_score,
                'mutual_info': score.mutual_info
            }
            for score in ranking_data.feature_scores
        ])

        # Sort by importance score
        results_df = results_df.sort_values(
            by='importance_score',
            ascending=False
        ).reset_index(drop=True)

        # Add metadata
        results_df['model'] = ranking_data.model_name
        results_df['timestamp'] = ranking_data.timestamp
        results_df['iteration'] = ranking_data.iteration

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to a single sheet, appending rows across iterations
        if output_path.exists():
            existing_df = pd.read_excel(output_path)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)

        results_df.to_excel(output_path, sheet_name='Ranked Features', index=False)
        
        logger.info(f"✅ Successfully saved {len(results_df)} ranked features to Excel")

    except Exception as e:
        logger.error(f"❌ Failed to save ranked features: {str(e)}")
        raise IOError(f"Feature ranking save failed: {str(e)}")
