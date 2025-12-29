# Running the GSM Project

This project can be run in two main ways:

1. **Web UI (recommended for most users)** using Streamlit
2. **Command line workflow** (useful for reproducible runs / automation)

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

## If the browser cannot open Streamlit

Common causes:
- Streamlit server did not start (check terminal errors)
- Port is blocked or already used

See: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
