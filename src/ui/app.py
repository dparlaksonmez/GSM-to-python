import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.workflows.GSM_workflow import gsm_run
from src.workflows.GSM_workflow_config import (
    NUMBER_OF_ITERATIONS, TRAIN_TEST_SPLIT_RATIO, MODEL_NAME, 
    LABEL_COLUMN_NAME, NORMALIZATION_METHOD, GENE_COLUMN_NAME, GROUP_COLUMN_NAME,
    CLASS_LABELS_POSITIVE, CLASS_LABELS_NEGATIVE
)

# Page configuration
st.set_page_config(
    page_title="GSM Bioinformatics Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def smart_read_csv(file_input):
    """Reads a CSV/TXT file with automatic separator detection."""
    try:
        # Reset pointer if possible
        if hasattr(file_input, 'seek'):
            file_input.seek(0)
            
        # Try reading with Python engine and auto-detection
        return pd.read_csv(file_input, sep=None, engine='python')
    except Exception as e:
        # Fallback strategies
        try:
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
            return pd.read_csv(file_input, sep=',')
        except:
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
            return pd.read_csv(file_input, sep='\t')

class StreamlitLogHandler(logging.Handler):
    def __init__(self, container, state_key: str):
        super().__init__()
        self.container = container
        self.state_key = state_key

    def emit(self, record):
        msg = self.format(record)
        current = st.session_state.get(self.state_key, "")
        st.session_state[self.state_key] = current + msg + "\n"
        log_text = st.session_state[self.state_key].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.container.markdown(
            f"""
            <div id="log-box" style="height:240px; overflow-y:auto; border:1px solid #ddd; padding:8px; background:#0e1117; color:#e6e6e6; font-family:monospace; white-space:pre-wrap;">
{log_text}
            </div>
            <script>
            const logBox = document.getElementById('log-box');
            if (logBox) {{ logBox.scrollTop = logBox.scrollHeight; }}
            </script>
            """,
            unsafe_allow_html=True
        )

def main():
    st.title("🧬 GSM Bioinformatics Pipeline")
    st.markdown("### Grouping-Scoring-Modeling Analysis Tool")

    if "log_text" not in st.session_state:
        st.session_state["log_text"] = ""
    if "last_output_path" not in st.session_state:
        st.session_state["last_output_path"] = None
    
    # Sidebar for Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📁 Data Input")
        
        input_method = st.radio("Input Method", ["Select from 'data' folder", "Upload Files"])
        
        expression_file = None
        group_file = None
        
        if input_method == "Upload Files":
            expression_file = st.file_uploader("Expression Data (CSV)", type=['csv'])
            group_file = st.file_uploader("Group Data (CSV/TXT)", type=['csv', 'txt'])
        else:
            data_dir = project_root / "data"
            
            # Helper to find files
            def find_files(patterns, subdirs):
                files = {}
                for subdir in subdirs:
                    path = data_dir / subdir
                    if path.exists():
                        for pattern in patterns:
                            for f in path.glob(pattern):
                                files[f"{subdir}/{f.name}"] = f
                return files

            # Expression Files (main_data, test)
            expr_files = find_files(["*.csv"], ["main_data", "test"])
            if expr_files:
                selected_expr = st.selectbox("Expression Data", options=list(expr_files.keys()))
                expression_file = expr_files[selected_expr]
            else:
                st.warning("No CSV files found in data/main_data or data/test")

            # Group Files (grouping_data, test)
            group_files = find_files(["*.csv", "*.txt"], ["grouping_data", "test"])
            if group_files:
                selected_group = st.selectbox("Group Data", options=list(group_files.keys()))
                group_file = group_files[selected_group]
            else:
                st.warning("No Group files found in data/grouping_data or data/test")
        
        st.subheader("🔧 Parameters")
        
        # Pipeline Parameters
        n_iterations = st.number_input(
            "Number of Iterations", 
            min_value=1, 
            max_value=100, 
            value=NUMBER_OF_ITERATIONS
        )
        
        split_ratio = st.slider(
            "Train/Test Split Ratio",
            min_value=0.1,
            max_value=0.9,
            value=TRAIN_TEST_SPLIT_RATIO,
            step=0.05
        )
        
        norm_method = st.selectbox(
            "Normalization Method",
            options=['zscore', 'minmax', 'robust'],
            index=['zscore', 'minmax', 'robust'].index(NORMALIZATION_METHOD) if NORMALIZATION_METHOD in ['zscore', 'minmax', 'robust'] else 0
        )
        
        model_name = st.selectbox(
            "Machine Learning Model",
            options=['RandomForest', 'SVM', 'LogisticRegression'],
            index=0  # Default to RandomForest
        )
        
        st.subheader("🏷️ Column Names & Labels")
        label_col = st.text_input("Label Column", value=LABEL_COLUMN_NAME)
        pos_label = st.text_input("Positive Class Label", value=CLASS_LABELS_POSITIVE)
        neg_label = st.text_input("Negative Class Label", value=CLASS_LABELS_NEGATIVE)
        gene_col = st.text_input("Gene Column", value=GENE_COLUMN_NAME)
        group_col = st.text_input("Group Column", value=GROUP_COLUMN_NAME)

    # Main Content Area
    if expression_file and group_file:
        st.info("✅ Data files selected. Ready to run pipeline.")
        
        # Preview Data
        with st.expander("📊 Data Preview"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Expression Data**")
                try:
                    df_expr = smart_read_csv(expression_file)
                    preview_cols = list(df_expr.columns[:12])
                    if len(df_expr.columns) > 12:
                        st.caption("Showing first 12 columns for readability.")
                    st.dataframe(df_expr[preview_cols].head())
                    st.caption(f"Shape: {df_expr.shape}")
                    # Reset file pointer if it's an uploaded file
                    if hasattr(expression_file, 'seek'):
                        expression_file.seek(0)
                except Exception as e:
                    st.error(f"Error reading expression file: {e}")
            
            with col2:
                st.markdown("**Group Data**")
                try:
                    df_group = smart_read_csv(group_file)
                    preview_cols = list(df_group.columns[:6])
                    if len(df_group.columns) > 6:
                        st.caption("Showing first 6 columns for readability.")
                    st.dataframe(df_group[preview_cols].head())
                    st.caption(f"Shape: {df_group.shape}")
                    # Reset file pointer if it's an uploaded file
                    if hasattr(group_file, 'seek'):
                        group_file.seek(0)
                except Exception as e:
                    st.error(f"Error reading group file: {e}")

        # Run Button
        if st.button("🚀 Run GSM Pipeline"):
            try:
                st.session_state["log_text"] = ""
                st.session_state["last_output_path"] = None

                log_section = st.container()
                log_placeholder = log_section.empty()
                
                # Create handler
                st_handler = StreamlitLogHandler(log_placeholder, "log_text")
                st_handler.setLevel(logging.INFO)
                
                # Create a formatter
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                st_handler.setFormatter(formatter)

                with st.spinner("Running pipeline... This may take a while."):
                    # Load data for processing
                    input_data = smart_read_csv(expression_file)
                    group_data = smart_read_csv(group_file)
                    
                    # Run Pipeline
                    output_path = gsm_run(
                        input_data=input_data,
                        group_data=group_data,
                        sample_ratio=split_ratio,
                        n_iterations=n_iterations,
                        model_name=model_name,
                        label_column=label_col,
                        positive_class_label=pos_label,
                        negative_class_label=neg_label,
                        gene_column=gene_col,
                        group_column=group_col,
                        normalization_method=norm_method,
                        notebook_mode=False,
                        extra_handlers=[st_handler]
                    )
                    
                    st.session_state["last_output_path"] = output_path
                    st.success(f"Pipeline completed successfully! Results saved to: {output_path}")
                            
            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")
                st.exception(e)

        # Results Section (always below the run button)
        if st.session_state.get("last_output_path"):
            output_path = st.session_state["last_output_path"]
            st.markdown("## 📈 Analysis Results")

            summary_file = output_path / "summary_report.txt"
            if summary_file.exists():
                with st.expander("📄 Summary Report", expanded=True):
                    with open(summary_file, 'r') as f:
                        report_content = f.read()
                    st.text(report_content)

            st.markdown("### 📊 Visualizations")
            col1, col2 = st.columns(2)
            plot1 = output_path / "f1_scores_across_iterations.png"
            plot2 = output_path / "average_f1_scores_by_groups.png"

            with col1:
                if plot1.exists():
                    st.image(str(plot1), caption="F1 Scores across Iterations", use_column_width=True)

            with col2:
                if plot2.exists():
                    st.image(str(plot2), caption="Average F1 Scores by Groups", use_column_width=True)
                
    else:
        st.warning("👈 Please select or upload both Expression Data and Group Data files in the sidebar to proceed.")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. **Select Data**: Choose to select files from the 'data' folder or upload new ones.
        2. **Configure**: Adjust parameters like iterations, split ratio, and column names if needed.
        3. **Run**: Click the 'Run GSM Pipeline' button.
        4. **Analyze**: View the generated summary report and visualizations.
        """)

if __name__ == "__main__":
    main()
