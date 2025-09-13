"""
📋 Example Configuration with Class Balancing

Purpose:
    Demonstrates how to configure the classification workflow 
    to use class balancing functionality.

Usage:
    from example_class_balancing_config import BALANCED_CONFIG
    results = classification_run(input_data=data, config=BALANCED_CONFIG)
"""

from classification_workflow_config import ClassificationWorkflowConfig

##### EXAMPLE CONFIGURATIONS #####

# Configuration with undersampling (reduces majority class)
UNDERSAMPLING_CONFIG = ClassificationWorkflowConfig(
    # Basic settings
    n_iterations=50,
    test_size=0.3,
    random_seed=42,
    
    # Algorithms to test
    algorithms=["RandomForest", "DecisionTree", "XGBoost"],
    
    # Class balancing settings
    apply_class_balancing=True,
    min_class_balance_ratio=0.8,  # Accept up to 1:1.25 ratio (80% balance)
    sampling_method="undersampling",  # Reduce majority class
    
    # Other settings
    save_to_excel=True,
    verbose=True
)

# Configuration with oversampling (increases minority class)
OVERSAMPLING_CONFIG = ClassificationWorkflowConfig(
    # Basic settings
    n_iterations=50,
    test_size=0.3,
    random_seed=42,
    
    # Algorithms to test
    algorithms=["RandomForest", "DecisionTree", "XGBoost"],
    
    # Class balancing settings
    apply_class_balancing=True,
    min_class_balance_ratio=0.7,  # Accept up to 1:1.43 ratio (70% balance)
    sampling_method="oversampling",  # Increase minority class
    
    # Other settings
    save_to_excel=True,
    verbose=True
)

# Configuration without class balancing (for comparison)
NO_BALANCING_CONFIG = ClassificationWorkflowConfig(
    # Basic settings
    n_iterations=50,
    test_size=0.3,
    random_seed=42,
    
    # Algorithms to test
    algorithms=["RandomForest", "DecisionTree", "XGBoost"],
    
    # Class balancing settings
    apply_class_balancing=False,  # Disabled
    
    # Other settings
    save_to_excel=True,
    verbose=True
)

# Default balanced configuration (recommended)
BALANCED_CONFIG = UNDERSAMPLING_CONFIG

##### USAGE EXAMPLES #####

def example_usage():
    """
    Example of how to use the class balancing configurations.
    """
    import pandas as pd
    from classification_workflow import classification_run
    
    # Load your data
    # data = pd.read_csv("your_data.csv")
    
    # Example 1: Run with undersampling
    # results_undersampling = classification_run(
    #     input_data=data,
    #     config=UNDERSAMPLING_CONFIG
    # )
    
    # Example 2: Run with oversampling
    # results_oversampling = classification_run(
    #     input_data=data,
    #     config=OVERSAMPLING_CONFIG
    # )
    
    # Example 3: Run without balancing (for comparison)
    # results_no_balancing = classification_run(
    #     input_data=data,
    #     config=NO_BALANCING_CONFIG
    # )
    
    print("📋 Configuration examples created!")
    print("💡 Use these configurations with classification_run() function")
    print("📊 Compare results between balanced and unbalanced approaches")

if __name__ == "__main__":
    example_usage()