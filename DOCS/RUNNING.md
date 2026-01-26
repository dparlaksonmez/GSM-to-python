# Running the GSM Project

This project can be run in multiple ways:

1. **Quick Test** - Verify the pipeline works (recommended first!)
2. **Web UI (Streamlit)** - Interactive interface for single runs
3. **Command Line** - Full run with config file settings
4. **Batch Runner** - Process multiple datasets automatically

---

## Quick Test (Start Here!)

Before running on real data, verify everything works:

```bash
cd ~/GSM-to-python
source venv/bin/activate

# Quick test with sample data (3 iterations, ~1-2 minutes)
python run_test.py

# Test with more iterations
python run_test.py --iterations 5

# Test with real data (small run)
python run_test.py --real-data --iterations 3
```

If the test passes, you're ready for full runs!

---

## Before you run anything (VS Code way)

If you are using Windows, first make sure VS Code is connected to WSL:
- Bottom-left should show `WSL: Ubuntu`
- If not: press `Ctrl+Shift+P` → **WSL: New WSL Window** → open the project folder

Then select the project’s Python environment in VS Code:
1. Press `Ctrl+Shift+P`
2. Run: **Python: Select Interpreter**
3. Choose the interpreter from `.venv` (it usually shows something like `.venv/bin/python`)

This helps VS Code run and lint using the correct packages.

---

## (Fallback) Before you run anything in terminal

Open Ubuntu (WSL), go to the project root, and activate your environment:

```bash
cd ~/GSM-to-python
source .venv/bin/activate
```

If you do not have `.venv` yet, follow: [INSTALL_WSL.md](INSTALL_WSL.md)

---

## Option 1: Run the Web UI (Streamlit)

### Recommended: run from VS Code (integrated terminal)

In VS Code:
1. **Terminal → New Terminal**
2. Run:

```bash
streamlit run src/ui/app.py
```

Then open the URL that appears (usually `http://localhost:8501`).

Tip (WSL): If the URL does not open automatically, copy it into your Windows browser.

### What you do in the UI

- Upload **expression data** (CSV)
- Upload **group definitions** (CSV/TXT)
- Configure pipeline parameters (iterations, train/test split, model, etc.)
- Run and view plots + summary

More UI details are also described in: [src/ui/README.md](../src/ui/README.md)

---

## Option 2: Run from the command line

### Run from VS Code terminal

In VS Code:
1. **Terminal → New Terminal**
2. Run the workflow using one of these commands:

```bash
python src/workflows/GSM_workflow.py
```

If you get import/module errors, try:

```bash
python -m src.workflows.GSM_workflow
```

---

## Where results are saved

Outputs are stored under the `output/` folder.

Think of `output/` as the project’s “results notebook”: each run gets its own timestamped folder.

---

## Where the input data lives (suggestion)

This repository already contains example data under `data/`.

For your own experiments, a simple approach is:
- Put raw input files under `data/`
- Keep outputs automatically under `output/`

---

## Option 3: Running Long Batch Jobs (Multiple Datasets)

When processing multiple datasets, the pipeline can run for hours or even days. If you close your terminal or lose your SSH connection, the job will be killed. Use `screen` to keep jobs running in the background.

### What is `screen`?

`screen` is a terminal multiplexer that creates a **virtual terminal session** on the server. Think of it like leaving a TV playing in a room and closing the door - the TV keeps playing even though you left.

```
┌─────────────────────────────────────────────────────────────┐
│  Your Terminal (SSH/WSL)                                    │
│    │                                                        │
│    └──► screen session (lives on server)                    │
│              │                                              │
│              └──► python run_all_datasets.py                │
│                   (keeps running even if you disconnect!)   │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Guide

#### 1. Start a new screen session

```bash
screen -S gsm_batch
```

This creates a named session called `gsm_batch`.

#### 2. Activate your environment and run the job

```bash
cd ~/GSM-to-python
source venv/bin/activate
python run_all_datasets.py --datasets GDS2545 GDS2547 GDS3257 GDS3268 GDS3837 GDS4206 GDS4824 GDS5499
```

#### 3. Detach from the session (IMPORTANT!)

Press these keys in sequence:
1. `Ctrl+A`
2. Then press `D`

You'll see: `[detached from session gsm_batch]`

**Now you can safely close your terminal.** The job continues running.

#### 4. Reattach later to check progress

```bash
screen -r gsm_batch
```

### Quick Reference

| Action | Command |
|--------|---------|
| Create new session | `screen -S session_name` |
| Detach (leave running) | `Ctrl+A`, then `D` |
| List all sessions | `screen -ls` |
| Reattach to session | `screen -r session_name` |
| Kill session (from inside) | `exit` or `Ctrl+A`, then `K` |

### ⚠️ Common Mistakes

| Mistake | Result | Solution |
|---------|--------|----------|
| Close terminal without detaching | ❌ Job killed | Always detach first (`Ctrl+A`, `D`) |
| Forget session name | Can't find it | Use `screen -ls` to list all |
| Multiple sessions with same name | Confusion | Use unique names like `gsm_batch_jan25` |

### Alternative: Using `nohup` (simpler but less control)

If you just want to run and forget without learning screen:

```bash
nohup python run_all_datasets.py --datasets GDS2547 GDS3257 GDS3268 > batch_run.log 2>&1 &
```

- Output saved to `batch_run.log`
- Check progress: `tail -f batch_run.log`
- Find process: `ps aux | grep run_all_datasets`
- Kill process: `pkill -f run_all_datasets`

---

## If the browser cannot open Streamlit

Common causes:
- Streamlit server did not start (check terminal errors)
- Port is blocked or already used

See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
