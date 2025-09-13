"""
🔄 Workflows Package

This package contains the main workflow implementations for the GSM pipeline.

Available workflows:
- GSM_workflow: Main GSM (Grouping-Scoring-Modeling) pipeline
- classification_workflow: General classification pipeline with Monte Carlo CV

Usage:
    from src.workflows import GSM_workflow
    from src.workflows import classification_workflow
"""

# Note: Individual modules should be imported directly to avoid dependency issues
# Example: from src.workflows.GSM_workflow import gsm_run