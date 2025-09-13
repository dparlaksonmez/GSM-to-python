"""
🧪 Test Class Balancing Integration

Purpose:
    Test script to verify that the class balancing functionality
    is properly integrated into the classification workflow.

Usage:
    python test_class_balancing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from classification_workflow_config import ClassificationWorkflowConfig
from classification_workflow import classification_run

# Create a simple test dataset with imbalanced classes
def create_test_data():
    """Create an imbalanced test dataset."""
    np.random.seed(42)
    
    # Create features
    n_samples = 1000
    n_features = 10
    
    # Create imbalanced classes (80% class 0, 20% class 1)
    n_class_0 = 800
    n_class_1 = 200
    
    # Generate features for each class
    X_class_0 = np.random.normal(0, 1, (n_class_0, n_features))
    X_class_1 = np.random.normal(1, 1, (n_class_1, n_features))
    
    # Combine data
    X = np.vstack([X_class_0, X_class_1])
    y = np.hstack([np.zeros(n_class_0), np.ones(n_class_1)])
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["class"] = y.astype(int)
    
    return df

def test_class_balancing():
    """Test the class balancing functionality."""
    print("🧪 Testing Class Balancing Integration")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data()
    print(f"📊 Original data shape: {test_data.shape}")
    print(f"📈 Original class distribution:")
    print(test_data["class"].value_counts())
    print()
    
    # Test 1: Without class balancing
    print("🔍 Test 1: Without class balancing")
    config_no_balance = ClassificationWorkflowConfig(
        n_iterations=3,
        test_size=0.3,
        apply_class_balancing=False,
        algorithms=["RandomForest"]
    )
    
    try:
        results_no_balance = classification_run(
            input_data=test_data,
            config=config_no_balance
        )
        print("✅ Test 1 passed: Workflow runs without class balancing")
    except Exception as e:
        print(f"❌ Test 1 failed: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: With undersampling
    print("🔍 Test 2: With undersampling")
    config_undersampling = ClassificationWorkflowConfig(
        n_iterations=3,
        test_size=0.3,
        apply_class_balancing=True,
        min_class_balance_ratio=0.8,
        sampling_method="undersampling",
        algorithms=["RandomForest"]
    )
    
    try:
        results_undersampling = classification_run(
            input_data=test_data,
            config=config_undersampling
        )
        print("✅ Test 2 passed: Workflow runs with undersampling")
    except Exception as e:
        print(f"❌ Test 2 failed: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: With oversampling
    print("🔍 Test 3: With oversampling")
    config_oversampling = ClassificationWorkflowConfig(
        n_iterations=3,
        test_size=0.3,
        apply_class_balancing=True,
        min_class_balance_ratio=0.8,
        sampling_method="oversampling",
        algorithms=["RandomForest"]
    )
    
    try:
        results_oversampling = classification_run(
            input_data=test_data,
            config=config_oversampling
        )
        print("✅ Test 3 passed: Workflow runs with oversampling")
    except Exception as e:
        print(f"❌ Test 3 failed: {str(e)}")
    
    print("\n🎉 Class balancing integration tests completed!")

if __name__ == "__main__":
    test_class_balancing()