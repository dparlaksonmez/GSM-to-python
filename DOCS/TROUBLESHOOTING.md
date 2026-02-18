# Troubleshooting

This page lists common problems when installing/running the project on WSL.

---

## 1) Git LFS – data files are empty / contain only pointer text

This project uses **Git LFS** for CSV data files. If you see tiny files
(~130 bytes) that start with `version https://git-lfs.github.com/spec/v1`,
LFS did not download the real content.

```bash
# Install Git LFS (once per machine)
sudo apt install -y git-lfs
git lfs install

# Pull all tracked data files
git lfs pull
```

If you cloned *before* installing `git-lfs`, run `git lfs pull` from the
repo root to replace pointer stubs with the real files.

---

## 2) "Missing column" error when loading grouping data

The pipeline **auto-detects** the separator (comma, tab, etc.) in both
expression and grouping files.  If you still get a "missing column" error:

- Open the file in a text editor and check whether it uses **commas**,
  **tabs**, or another delimiter.
- Make sure the header row contains the expected column names
  (`feature_id`, `group_name` for grouping files; `class` for expression
  data).
- Windows-created files with `\r\n` line endings work fine — no need to
  convert.

---

## 3) "command not found: python" or "python3 not found"

Install Python in Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

---

## 4) Virtual environment not activating

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `source` fails, check you are inside Ubuntu (WSL) and that `.venv/bin/activate` exists.

---

## 5) `pip install -r dependencies.txt` fails

### First try upgrading pip

```bash
python -m pip install --upgrade pip
pip install -r dependencies.txt
```

### If the error mentions build tools

Install build essentials:

```bash
sudo apt update
sudo apt install -y build-essential
```

Then retry:

```bash
pip install -r dependencies.txt
```

---

## 6) Streamlit does not start

Make sure you activated the environment:

```bash
source .venv/bin/activate
```

Then:

```bash
streamlit run src/ui/app.py
```

If Streamlit says the port is in use, try closing other Streamlit runs, or use another port:

```bash
streamlit run src/ui/app.py --server.port 8502
```

---

## 7) Browser cannot open `localhost:8501`

- Confirm Streamlit is still running in the terminal (no crash)
- Try opening the full URL Streamlit prints
- If you use a VPN/firewall, it can sometimes block local ports

---

## 8) Import errors when running the workflow

Try the module form (recommended):

```bash
python -m src.workflows.GSM_workflow
```

If errors persist, confirm you are running from the project root (`~/GSM-to-python`).

---

## 9) "Permission denied"

This is uncommon for Python scripts, but if you see it:
- Check file permissions
- Ensure you are not trying to run files from restricted locations

---

## 10) Still stuck

When asking for help, provide:
- The exact command you ran
- The full error message text
- Whether you are in WSL (Ubuntu)
