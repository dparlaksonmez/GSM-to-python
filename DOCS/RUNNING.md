# Running the GSM Pipeline 🚀

This project can be run in multiple ways:

| Method | Best For | Time |
|--------|----------|------|
| **Quick Test** | Verifying setup works | ~1-2 min |
| **Single Dataset** | Full analysis on one dataset | 10-30 min |
| **Batch Runner** | All 7 datasets | 2-6 hours |
| **Streamlit UI** | Interactive exploration | Variable |

---

## Before You Run Anything

### VS Code (recommended)

Make sure VS Code is connected to WSL:
1. Bottom-left should show `WSL: Ubuntu`
2. If not: `Ctrl+Shift+P` → **WSL: New WSL Window** → open the project folder
3. Select the Python environment: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv` or `venv`

### Terminal

```bash
cd ~/GSM-to-python
source venv/bin/activate      # Adjust to .venv if that's what you used
```

If you don't have an environment yet, see [INSTALL_WSL.md](INSTALL_WSL.md).

---

## 1) Quick Test (Start Here!)

Before running on real data, verify everything works:

```bash
# Quick test with sample data (3 iterations, ~1-2 minutes)
python run_test.py

# Test with more iterations
python run_test.py --iterations 5

# Test with real data (small run)
python run_test.py --real-data --iterations 3
```

**Expected output:** A summary showing F1/AUC scores. If you see that, you're ready!

---

## 2) Single Dataset Run

Edit the configuration file first:

```bash
# Open in VS Code or any editor
code src/workflows/GSM_workflow_config.py
```

Key settings to check:
- `expression_file` — path to your GEO expression CSV
- `grouping_file` — path to DisGeNET grouping data
- `n_iterations` — number of iterations (default: 100)
- `classifier_type` — "random_forest" (default), "xgboost", etc.

Then run:

```bash
python src/workflows/GSM_workflow.py

# If import errors occur, use module form:
python -m src.workflows.GSM_workflow
```

---

## 3) Batch Run (Multiple Datasets)

Process multiple datasets in one go:

```bash
# All 7 datasets, 100 iterations each
python run_all_datasets.py

# Custom iterations
python run_all_datasets.py --iterations 50

# Specific datasets only
python run_all_datasets.py --datasets GDS2545 GDS3257

# List available datasets
python run_all_datasets.py --list
```

### Keep Batch Jobs Running After Disconnecting

For long jobs (2+ hours), use `screen` so the job survives terminal disconnects:

```
┌─────────────────────────────────────────────────────────────┐
│  Your Terminal (SSH/WSL)                                    │
│    └──► screen session (persists on server)                 │
│              └──► python run_all_datasets.py                │
│                   (keeps running even if you disconnect!)   │
└─────────────────────────────────────────────────────────────┘
```

**Step-by-step:**

```bash
# 1. Create a named session
screen -S gsm_batch

# 2. Start the job
cd ~/GSM-to-python
source venv/bin/activate
python run_all_datasets.py --iterations 100

# 3. Detach: press Ctrl+A, then D
#    You'll see: [detached from session gsm_batch]
#    Now you can close the terminal safely.

# 4. Reattach later
screen -r gsm_batch
```

**Quick reference:**

| Action | Command |
|--------|---------|
| Create session | `screen -S name` |
| Detach (leave running) | `Ctrl+A`, then `D` |
| List sessions | `screen -ls` |
| Reattach | `screen -r name` |
| Kill session (from inside) | `exit` |

**Common mistake:** Closing the terminal without detaching kills the job. Always `Ctrl+A, D` first!

**Alternative (simpler, less control):**

```bash
nohup python run_all_datasets.py --iterations 100 > batch_run.log 2>&1 &

# Monitor progress:
tail -f batch_run.log

# Find/kill the process:
ps aux | grep run_all_datasets
pkill -f run_all_datasets
```

---

## 4) Streamlit Web UI

```bash
streamlit run src/ui/app.py
```

Open the URL it prints (usually `http://localhost:8501`) in your browser.

**In the UI you can:**
- Upload expression data (CSV)
- Upload group definitions (CSV/TXT)
- Configure pipeline parameters
- Run the pipeline and view plots + summary

> **WSL tip:** If the URL doesn't auto-open, manually paste it into your Windows browser.
> If `localhost` doesn't work, try `http://127.0.0.1:8501`.
> See [INSTALL_WSL.md](INSTALL_WSL.md#cannot-connect-to-localhost--streamlit-url-doesnt-open) for more network fixes.

---

## Where Results Are Saved

Each run creates a **timestamped folder** under `output/`:

```
output/
└── gsm_2026_02_18-10_30_00_GDS2545_seed44_cancer-DisGeNET_gedinet/
    ├── summary_report.txt                    # Human-readable summary
    ├── modeling_results_all_iterations.json   # Detailed JSON results
    ├── modeling_results_statistics.xlsx       # Performance statistics
    ├── aggregated_group_ranking_rra.xlsx      # RRA group rankings
    ├── aggregated_feature_ranking_*.xlsx      # RRA feature rankings
    ├── run_parameters.txt                    # Config used
    ├── gsm_workflow.log                      # Detailed logs
    ├── figures/                              # 15+ plots
    │   ├── performance_boxplot.png
    │   ├── feature_importance.png
    │   └── statistics_*.xlsx
    └── biological_validation/                # If enabled
        ├── enrichr_results.xlsx
        ├── string_network.xlsx
        └── disgenet_results.xlsx
```

---

## Where Input Data Lives

The repository includes example data under `data/`:

| Folder | Contents |
|--------|----------|
| `data/main_data/` | GEO expression matrices (CSV, tracked by Git LFS) |
| `data/grouping_data/` | DisGeNET gene-disease group mappings |
| `data/test/` | Small test fixtures |

For your own experiments, place raw input files under `data/`.

---

## If Something Goes Wrong

| Problem | Quick Fix |
|---------|-----------|
| Import errors | Use `python -m src.workflows.GSM_workflow` |
| "No module named X" | `pip install -r dependencies.txt` |
| Streamlit won't open | Try `http://127.0.0.1:8501` |
| Port already in use | `streamlit run src/ui/app.py --server.port 8502` |
| Job killed on disconnect | Use `screen` (see above) |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more fixes.
