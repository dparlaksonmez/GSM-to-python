# Troubleshooting

This page lists common problems when installing/running the project on WSL.

---

## 1) “command not found: python” or “python3 not found”

Install Python in Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

---

## 2) Virtual environment not activating

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `source` fails, check you are inside Ubuntu (WSL) and that `.venv/bin/activate` exists.

---

## 3) `pip install -r dependencies.txt` fails

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

## 4) Streamlit does not start

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

## 5) Browser cannot open `localhost:8501`

- Confirm Streamlit is still running in the terminal (no crash)
- Try opening the full URL Streamlit prints
- If you use a VPN/firewall, it can sometimes block local ports

---

## 6) Import errors when running the workflow

Try the module form (recommended):

```bash
python -m src.workflows.GSM_workflow
```

If errors persist, confirm you are running from the project root (`~/GSM-to-python`).

---

## 7) “Permission denied”

This is uncommon for Python scripts, but if you see it:
- Check file permissions
- Ensure you are not trying to run files from restricted locations

---

## 8) Still stuck

When asking for help, provide:
- The exact command you ran
- The full error message text
- Whether you are in WSL (Ubuntu)
