"""
GSM Bioinformatics Pipeline — Streamlit UI 🧬

Purpose:
    Browser-based graphical interface for the Grouping-Scoring-Modeling
    pipeline.  Allows researchers without programming experience to
    configure, run, and inspect classification results.

Key Features:
    - File selection from data/ folder or direct upload
    - Full parameter configuration via sidebar
    - Live log streaming during pipeline execution
    - Rich results dashboard with all figures and metrics
    - Historical run browser with one-click result loading
"""

import json
import logging
import sys
import time
from pathlib import Path

import streamlit as st

##### PATH SETUP #####

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# NOTE: gsm_run is imported lazily inside the run handler to speed up page load
from src.workflows.GSM_workflow_config import (
    CLASS_LABELS_NEGATIVE,
    CLASS_LABELS_POSITIVE,
    GENE_COLUMN_NAME,
    GROUP_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    MODEL_NAME,
    NORMALIZATION_METHOD,
    NUMBER_OF_ITERATIONS,
    TRAIN_TEST_SPLIT_RATIO,
)


##### PAGE CONFIG #####

st.set_page_config(
    page_title="GSM Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clear any stale cached data from previous code versions on first load
if "_cache_cleared" not in st.session_state:
    st.cache_data.clear()
    st.session_state["_cache_cleared"] = True

##### CUSTOM CSS #####

st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Sidebar header */
section[data-testid="stSidebar"] h1 { font-size: 1.15rem !important; }

/* Primary action button */
div.stButton > button[kind="primary"] {
    width: 100%;
    background: linear-gradient(135deg, #1A237E 0%, #4A90D9 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 1.05rem;
    transition: opacity 0.2s;
}
div.stButton > button[kind="primary"]:hover { opacity: 0.88; }

/* Metric cards — theme-aware */
[data-testid="stMetric"] {
    border-radius: 10px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}
@media (prefers-color-scheme: dark) {
    [data-testid="stMetric"] { background: #1e2130; }
}
@media (prefers-color-scheme: light) {
    [data-testid="stMetric"] { background: #F4F6FA; }
}
/* Streamlit's own dark class override */
[data-testtheme="dark"] [data-testid="stMetric"],
.stApp[data-theme="dark"] [data-testid="stMetric"],
html[data-theme="dark"] [data-testid="stMetric"] {
    background: #1e2130;
}
</style>
""", unsafe_allow_html=True)


##### HELPER FUNCTIONS #####

def _get_pd():
    """Lazy pandas import to keep initial page load fast."""
    import pandas as pd          # noqa: delayed import
    return pd


def smart_read_csv(file_input):
    """Read CSV/TXT with automatic separator detection."""
    pd = _get_pd()
    if hasattr(file_input, "seek"):
        file_input.seek(0)
    try:
        return pd.read_csv(file_input, sep=None, engine="python")
    except Exception:
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        return pd.read_csv(file_input, sep="\t")


@st.cache_data(ttl=300, show_spinner=False)
def _discover_output_runs(output_dir: str) -> list[dict]:
    """Scan the output/ folder and return metadata dicts for each run.

    Returns plain dicts (not dataclasses) so st.cache_data can pickle
    them reliably across hot-reloads.
    Keys: path, timestamp, dataset_name, group_name.
    """
    runs: list[dict] = []
    out = Path(output_dir)
    if not out.exists():
        return runs
    for d in sorted(out.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("gsm_"):
            continue
        # Parse: gsm_YYYY_MM_DD-HH_MM_SS_dataname_groupname
        parts = d.name.split("_", 4)
        if len(parts) < 4:
            continue
        ts_raw = "_".join(parts[1:4])
        rest = parts[4] if len(parts) > 4 else ""
        if "-" in ts_raw:
            date_part = ts_raw[:10]
            time_rest = ts_raw[11:]
            time_part = time_rest[:8] if len(time_rest) >= 8 else time_rest
            rest_of_name = time_rest[9:] if len(time_rest) > 9 else ""
            ts_display = (f"{date_part.replace('_', '-')}  "
                          f"{time_part.replace('_', ':')}")
        else:
            ts_display = ts_raw
            rest_of_name = rest

        r = rest_of_name if rest_of_name else rest
        name_parts = r.rsplit("_", 1)
        ds_name = name_parts[0] if name_parts else r
        gr_name = name_parts[1] if len(name_parts) > 1 else ""

        runs.append({
            "path": str(d),
            "timestamp": ts_display,
            "dataset_name": ds_name.replace("_", " "),
            "group_name": gr_name.replace("_", " "),
        })
    return runs


@st.cache_data(ttl=600, show_spinner=False)
def _load_report_text(run_path: str) -> str:
    """Load summary_report.txt for a run (cached)."""
    f = Path(run_path) / "summary_report.txt"
    return f.read_text() if f.exists() else "Summary report not available."


def _find_figures(run_path: str) -> list[str]:
    """Return all PNG figure paths inside a run's figures/ directory."""
    fig_dir = Path(run_path) / "figures"
    if not fig_dir.exists():
        return []
    return [str(p) for p in sorted(fig_dir.glob("*.png"))]


@st.cache_data(ttl=600, show_spinner=False)
def _load_json_results(run_path: str) -> list[dict] | None:
    """Load modeling results JSON if present (cached)."""
    f = Path(run_path) / "modeling_results_all_iterations.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _cached_read_bytes(file_path: str) -> bytes:
    """Read file bytes with caching to avoid repeated disk I/O."""
    return Path(file_path).read_bytes()


##### LOG HANDLER #####

class StreamlitLogHandler(logging.Handler):
    """Routes log messages into a Streamlit container."""

    def __init__(self, container, state_key: str):
        super().__init__()
        self.container = container
        self.state_key = state_key

    def emit(self, record):
        msg = self.format(record)
        current = st.session_state.get(self.state_key, "")
        st.session_state[self.state_key] = current + msg + "\n"
        safe = (st.session_state[self.state_key]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        # flex-direction:column-reverse keeps the scroll pinned to the bottom
        self.container.markdown(
            f'<div style="display:flex; flex-direction:column-reverse;'
            f' height:260px; overflow-y:auto; border:1px solid #334;'
            f' padding:8px; background:#0e1117; color:#e6e6e6;'
            f' font-family:Consolas,monospace; font-size:0.82rem;'
            f' border-radius:6px;">'
            f'<div style="white-space:pre-wrap;">{safe}</div></div>',
            unsafe_allow_html=True,
        )


##### RESULTS DASHBOARD #####

def render_results_dashboard(run_path: str):
    """Display a comprehensive results dashboard for a given run."""

    # ---- Summary metrics row ----
    json_results = _load_json_results(run_path)

    if json_results:
        # Each entry has {"metadata": {...}, "results": [{...}]}
        last_iter = json_results[-1]
        best = (last_iter.get("results", [{}])[0]
                if "results" in last_iter else last_iter)
        f1 = best.get("f1_score", 0)
        auc = best.get("auc_roc", 0)
        acc = best.get("accuracy", 0)
        n_groups = best.get("num_groups_used", "?")
        n_feats = best.get("num_features_used", "?")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("F1 Score", f"{f1:.3f}")
        c2.metric("AUC-ROC", f"{auc:.3f}")
        c3.metric("Accuracy", f"{acc:.3f}")
        c4.metric("Groups", str(n_groups))
        c5.metric("Features", str(n_feats))

    # ---- Tabs ----
    tab_report, tab_figures, tab_files = st.tabs(
        ["📄 Summary Report", "📊 Figures", "📁 Output Files"]
    )

    with tab_report:
        st.code(_load_report_text(run_path), language=None)

    with tab_figures:
        figures = _find_figures(run_path)
        if not figures:
            st.info("No figures generated for this run.")
        else:
            for i in range(0, len(figures), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(figures):
                        caption = Path(figures[idx]).stem.replace("_", " ").title()
                        col.image(figures[idx], caption=caption,
                                  width="stretch")

    with tab_files:
        exts = {".txt", ".json", ".xlsx", ".csv", ".png", ".log"}
        run_dir = Path(run_path)

        def _list_files(base: Path, prefix: str = ""):
            for f in sorted(base.iterdir()):
                if f.is_dir():
                    _list_files(f, prefix=f"{prefix}{f.name}/")
                elif f.suffix in exts:
                    kb = f.stat().st_size / 1024
                    cn, cs, cd = st.columns([4, 1, 1])
                    cn.text(f"{prefix}{f.name}")
                    cs.text(f"{kb:.0f} KB")
                    with cd:
                        st.download_button(
                            "⬇", data=_cached_read_bytes(str(f)),
                            file_name=f.name,
                            key=f"dl_{run_dir.name}_{prefix}{f.name}",
                        )

        _list_files(run_dir)


##### CLINICAL INFERENCE UI #####

def _discover_bundles(output_dir: str) -> list[Path]:
    """Find all .gsm.zip bundles in the output directory tree."""
    out = Path(output_dir)
    if not out.exists():
        return []
    return sorted(out.rglob("*.gsm.zip"), reverse=True)


def render_inference_page():
    """Render the clinical inference page."""
    st.markdown("## 🏥 Clinical Inference")
    st.markdown(
        "Use a trained model bundle to predict disease status for "
        "new patient expression samples."
    )

    # ---- Bundle selection ----
    st.markdown("### 1. Select Model Bundle")
    bundle_method = st.radio(
        "Bundle source",
        ["Select from output/", "Upload .gsm.zip"],
        horizontal=True,
    )

    bundle_path = None
    if bundle_method == "Upload .gsm.zip":
        uploaded = st.file_uploader(
            "Upload model bundle", type=["zip"],
            help="Upload a .gsm.zip file generated by the training pipeline",
        )
        if uploaded:
            # Save to temp location
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_path = tmp_dir / uploaded.name
            tmp_path.write_bytes(uploaded.read())
            bundle_path = tmp_path
    else:
        output_dir = str(project_root / "output")
        bundles = _discover_bundles(output_dir)
        if bundles:
            labels = [f"{b.parent.parent.name} / {b.name}" for b in bundles]
            sel_idx = st.selectbox(
                "Available bundles", range(len(labels)),
                format_func=lambda i: labels[i],
            )
            bundle_path = bundles[sel_idx] if sel_idx is not None else None
        else:
            st.info(
                "No model bundles found. Run the training pipeline first "
                "to generate a bundle."
            )

    # ---- Bundle info ----
    if bundle_path:
        try:
            from src.inference.model_bundle import load_bundle
            bundle = load_bundle(bundle_path)
            meta = bundle.metadata

            with st.expander("📦 Bundle Details", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Models", str(meta.n_models_saved))
                c2.metric("Features", str(meta.n_features))
                c3.metric("F1 Score", f"{meta.ensemble_f1_mean:.3f}")
                c4.metric("AUC-ROC", f"{meta.ensemble_auc_mean:.3f}")

                st.caption(
                    f"Dataset: {meta.dataset_name} | "
                    f"Classifier: {meta.model_name} | "
                    f"Strategy: {meta.ensemble_strategy} | "
                    f"Created: {meta.created_at[:10]}"
                )
        except Exception as e:
            st.error(f"Failed to load bundle: {e}")
            bundle = None
    else:
        bundle = None

    # ---- Patient data upload ----
    st.markdown("### 2. Upload Patient Data")
    patient_file = st.file_uploader(
        "Patient expression data (CSV/TXT)",
        type=["csv", "txt"],
        help="Gene expression matrix with the same gene names as training data",
    )

    sample_id_col = st.text_input(
        "Sample ID column (optional)", value="",
        help="Column name containing patient/sample identifiers",
    )

    # ---- Run inference ----
    if bundle and patient_file:
        if st.button("🔬 Run Inference", type="primary"):
            pd = _get_pd()
            try:
                from src.inference.inference_engine import infer
                from src.inference.clinical_report import (
                    generate_clinical_report,
                    generate_report_dataframe,
                )
                from src.utils.logger import setup_logger

                patient_file.seek(0)
                patient_data = smart_read_csv(patient_file)

                st.markdown(f"**Patient data:** {patient_data.shape[0]} samples × "
                            f"{patient_data.shape[1]} features")

                # Create a logger
                import tempfile
                log_path = Path(tempfile.mktemp(suffix=".log"))
                logger = setup_logger(str(log_path), logger_name="inference_ui")

                with st.spinner("Running inference..."):
                    summary = infer(
                        bundle,
                        patient_data,
                        logger=logger,
                        sample_id_column=sample_id_col or None,
                    )

                # ---- Display results ----
                st.markdown("### 📊 Results")

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Samples", summary.n_samples)
                c2.metric("Positive (Disease)", summary.n_positive)
                c3.metric("Negative (Control)", summary.n_negative)
                c4.metric("Mean Confidence", f"{summary.mean_confidence:.1%}")

                # Results table
                df = generate_report_dataframe(summary)
                st.dataframe(df, use_container_width=True)

                # Download buttons
                col_txt, col_xlsx = st.columns(2)
                with col_txt:
                    report_text = generate_clinical_report(summary)
                    st.download_button(
                        "📄 Download Report (TXT)",
                        data=report_text,
                        file_name="clinical_report.txt",
                        mime="text/plain",
                    )
                with col_xlsx:
                    import io
                    buf = io.BytesIO()
                    df.to_excel(buf, index=False)
                    st.download_button(
                        "📊 Download Report (Excel)",
                        data=buf.getvalue(),
                        file_name="clinical_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # Full text report
                with st.expander("📋 Full Clinical Report"):
                    st.code(report_text, language=None)

            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.exception(e)
    elif bundle and not patient_file:
        st.info("👆 Upload patient expression data to run inference.")


##### MAIN APPLICATION #####

def main():
    # ---- Header ----
    st.markdown(
        '<h1 style="margin-bottom:0;">🧬 GSM Bioinformatics Pipeline</h1>'
        '<p style="color:#555; margin-top:0; font-size:1.05rem;">'
        'Grouping–Scoring–Modeling &nbsp;|&nbsp; Knowledge-driven '
        'feature selection for transcriptomic classification</p>',
        unsafe_allow_html=True,
    )

    # ---- Top-level navigation ----
    nav_tab = st.radio(
        "Mode",
        ["🧪 Training Pipeline", "🏥 Clinical Inference"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if nav_tab == "🏥 Clinical Inference":
        render_inference_page()
        return

    # ---- Session state ----
    if "log_text" not in st.session_state:
        st.session_state["log_text"] = ""
    if "last_output_path" not in st.session_state:
        st.session_state["last_output_path"] = None


    # ==================================================================
    #  SIDEBAR
    # ==================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### 📁 Data Input")
        input_method = st.radio(
            "Input method",
            ["Select from data/ folder", "Upload files"],
            horizontal=True,
        )

        expression_file = None
        group_file = None
        expr_name = None
        group_name = None

        if input_method == "Upload files":
            expression_file = st.file_uploader(
                "Expression data", type=["csv", "txt"],
                help="Gene-expression matrix (samples × genes)")
            group_file = st.file_uploader(
                "Group data", type=["csv", "txt"],
                help="Gene-to-group mapping file")
            if expression_file:
                expr_name = Path(expression_file.name).stem
            if group_file:
                group_name = Path(group_file.name).stem
        else:
            data_dir = project_root / "data"

            def _find(patterns, subdirs):
                found = {}
                for sd in subdirs:
                    p = data_dir / sd
                    if p.exists():
                        for pat in patterns:
                            for f in p.glob(pat):
                                found[f"{sd}/{f.name}"] = f
                return found

            expr_files = _find(["*.csv"], ["expression_data", "test"])
            if expr_files:
                sel = st.selectbox("Expression data",
                                   list(expr_files.keys()))
                expression_file = expr_files[sel]
                expr_name = Path(sel).stem
            else:
                st.warning("No CSV files in data/expression_data or data/test")

            group_files = _find(["*.csv", "*.txt"],
                                ["grouping_data", "test"])
            if group_files:
                sel = st.selectbox("Group data",
                                   list(group_files.keys()))
                group_file = group_files[sel]
                group_name = Path(sel).stem
            else:
                st.warning("No group files in data/grouping_data or data/test")

        st.divider()

        st.markdown("### 🔧 Parameters")
        n_iterations = st.number_input(
            "Iterations", min_value=1, max_value=100,
            value=NUMBER_OF_ITERATIONS,
            help="Number of independent train/test random splits",
        )
        split_ratio = st.slider(
            "Train / Test split", 0.50, 0.90,
            value=TRAIN_TEST_SPLIT_RATIO, step=0.05,
        )
        norm_method = st.selectbox(
            "Normalisation",
            ["zscore", "minmax", "robust"],
            index=(["zscore", "minmax", "robust"].index(NORMALIZATION_METHOD)
                   if NORMALIZATION_METHOD in ["zscore", "minmax", "robust"]
                   else 0),
        )
        model_name = st.selectbox(
            "Classifier",
            ["XGBoost", "RandomForest", "DecisionTree", "SVM", "KNN", "MLP"],
        )

        st.divider()

        st.markdown("### 🏷️ Label Mapping")
        label_col = st.text_input("Label column", LABEL_COLUMN_NAME)
        c1, c2 = st.columns(2)
        pos_label = c1.text_input("Positive class", CLASS_LABELS_POSITIVE)
        neg_label = c2.text_input("Negative class", CLASS_LABELS_NEGATIVE)
        gene_col = st.text_input("Gene column", GENE_COLUMN_NAME)
        group_col = st.text_input("Group column", GROUP_COLUMN_NAME)

    # ==================================================================
    #  MAIN AREA
    # ==================================================================
    if expression_file and group_file:
        # ---- Data preview ----
        with st.expander("📊 Data Preview", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Expression data**")
                try:
                    df_expr = smart_read_csv(expression_file)
                    st.dataframe(df_expr[list(df_expr.columns[:10])].head(6),
                                 width="stretch")
                    st.caption(f"{df_expr.shape[0]} samples × "
                               f"{df_expr.shape[1]} features")
                    if hasattr(expression_file, "seek"):
                        expression_file.seek(0)
                except Exception as e:
                    st.error(f"Error: {e}")
            with col2:
                st.markdown("**Group data**")
                try:
                    df_grp = smart_read_csv(group_file)
                    st.dataframe(df_grp.head(6),
                                 width="stretch")
                    st.caption(f"{df_grp.shape[0]} rows × "
                               f"{df_grp.shape[1]} columns")
                    if hasattr(group_file, "seek"):
                        group_file.seek(0)
                except Exception as e:
                    st.error(f"Error: {e}")

        # ---- Run button ----
        run_clicked = st.button("🚀  Run GSM Pipeline", type="primary")
        st.caption("Use the **Stop** button in the top-right corner of the "
                   "page to cancel a running pipeline.")

        # Show persisted log from a previous run
        if st.session_state.get("log_text") and not run_clicked:
            with st.expander("📋 Pipeline Log", expanded=False):
                _safe = (st.session_state["log_text"]
                         .replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
                st.markdown(
                    f'<div style="display:flex; flex-direction:column-reverse;'
                    f' height:260px; overflow-y:auto; border:1px solid #334;'
                    f' padding:8px; background:#0e1117; color:#e6e6e6;'
                    f' font-family:Consolas,monospace; font-size:0.82rem;'
                    f' border-radius:6px;">'
                    f'<div style="white-space:pre-wrap;">{_safe}</div></div>',
                    unsafe_allow_html=True,
                )

        if run_clicked:
            st.session_state["log_text"] = ""
            st.session_state["last_output_path"] = None

            log_placeholder = st.empty()
            handler = StreamlitLogHandler(log_placeholder, "log_text")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s  %(levelname)s  %(message)s",
                datefmt="%H:%M:%S"))

            with st.spinner("Pipeline running … use the Stop button "
                            "(top-right) to cancel."):
                try:
                    from src.workflows.GSM_workflow import gsm_run

                    input_data = smart_read_csv(expression_file)
                    group_data = smart_read_csv(group_file)

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
                        extra_handlers=[handler],
                        input_data_name=expr_name,
                        group_data_name=group_name,
                    )

                    st.session_state["last_output_path"] = str(output_path)
                    st.success(
                        f"✅  Pipeline completed — results at "
                        f"`{output_path.name}`")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.exception(e)

        # ---- Show latest results ----
        if st.session_state.get("last_output_path"):
            st.markdown("---")
            st.markdown("## 📈 Latest Results")
            render_results_dashboard(
                st.session_state["last_output_path"])

    else:
        st.info("👈  Select or upload **Expression data** and "
                "**Group data** in the sidebar to get started.")
        st.markdown("""
        ### Quick Start
        1. **Select data** — pick files from the `data/` folder or upload
        2. **Configure** — adjust iterations, split ratio, and classifier
        3. **Run** — click **Run GSM Pipeline** and watch the live log
        4. **Analyse** — browse metrics, figures, and downloadable files
        """)

    # ==================================================================
    #  HISTORICAL RUNS  (runs as a fragment — selecting a run only
    #  rerenders this section, not the entire page)
    # ==================================================================
    st.markdown("---")
    st.markdown("## 🕘 Run History")

    @st.fragment
    def _history_fragment():
        output_dir = str(project_root / "output")
        runs = _discover_output_runs(output_dir)

        if not runs:
            st.caption("No previous runs found in output/.")
            return

        st.caption(f"{len(runs)} previous run(s) found.")

        # Build labels for selectbox (lightweight — no file I/O)
        run_labels = [
            f"{r['timestamp']}  —  {r['dataset_name']}  /  {r['group_name']}"
            for r in runs
        ]

        selected_idx = st.selectbox(
            "Select a run to inspect",
            range(len(run_labels)),
            format_func=lambda i: run_labels[i],
            key="history_run_selector",
        )

        # Render dashboard only when user clicks Load
        if selected_idx is not None:
            if st.button("📂 Load Results", key="load_history_btn"):
                st.session_state["_loaded_run_idx"] = selected_idx

            loaded_idx = st.session_state.get("_loaded_run_idx")
            if loaded_idx is not None and loaded_idx < len(runs):
                with st.spinner("Loading run results…"):
                    render_results_dashboard(runs[loaded_idx]["path"])

    _history_fragment()


if __name__ == "__main__":
    main()
