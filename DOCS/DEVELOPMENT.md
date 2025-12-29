# Development Guide (Beginner-Friendly)

This guide explains how to *safely* make changes to the project.

If you are new to programming, think of this repo as:
- A **protocol** (the code)
- A **lab notebook** (the history in Git)
- A **shared freezer** (GitHub, where everyone stores the latest version)

---

## 1) Open the project in the right place (WSL)

Make sure you are working inside Ubuntu/WSL, not directly on Windows paths.

### VS Code way (recommended)

1. Open VS Code
2. Press `Ctrl+Shift+P`
3. Run: **WSL: New WSL Window**
4. In that window, open the project folder (`~/GSM-to-python`) via **File → Open Folder**

Bottom-left should show something like `WSL: Ubuntu`.

### (Fallback) Terminal way

In Ubuntu terminal:

```bash
cd ~/GSM-to-python
code .
```

---

## 2) Use the correct Python environment (VS Code)

In VS Code:
1. Press `Ctrl+Shift+P`
2. Run: **Python: Select Interpreter**
3. Choose the `.venv` interpreter for this project

This is the VS Code equivalent of “activating the environment”.

### (Fallback) Activate in terminal

Before running or developing (terminal):

```bash
cd ~/GSM-to-python
source .venv/bin/activate
```

---

## 3) High-level project structure (what is where)

- `src/` : the main code
  - `src/workflows/` : “entry points” that run end-to-end workflows
  - `src/ui/` : Streamlit web UI
  - Other folders under `src/` : pipeline stages and utilities
- `data/` : example input datasets
- `output/` : results from previous runs
- `DOCS/` : documentation for users and contributors

If you are unsure where to change something, start by searching inside `src/`.

---

## 4) A safe workflow for making changes

When you change code, you want to avoid breaking the main branch for others.

Use this pattern:
1. Pull latest changes
2. Create a new branch for your work
3. Make small changes
4. Run the UI or workflow to check
5. Commit and push
6. Open a Pull Request (PR)

The UI-first Git workflow is explained step-by-step in: [GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md)

---

## 5) Editing code: keep it simple

This project aims to be understandable for researchers.

Suggested habits:
- Make changes in small steps (one idea at a time)
- Prefer clear names over short names
- Add short docstrings to new functions
- Avoid creating “clever” code that is hard to read

---

## 6) Running after you change something

After changes, do a quick run:

### UI run (VS Code terminal)

```bash
streamlit run src/ui/app.py
```

### CLI workflow run (VS Code terminal)

```bash
python -m src.workflows.GSM_workflow
```

If something breaks, check errors carefully. Most issues are missing packages or import paths.

---

## 7) Share your changes on GitHub (integrate your code)

If you want your changes to become part of the shared project, you need to push them to GitHub and open a Pull Request.

### VS Code way (recommended)

Important: do **not** work directly on `main`.

Analogy: `main` is the “official lab protocol”. You do your experiments on a separate bench (a branch), then propose an update.

1. Create a new branch (before editing, if possible)
  - Look at the bottom-left of VS Code: it shows your current branch (often `main`)
  - Click the branch name
  - Choose **Create new branch…**
  - Name it like `yourname-short-task` (example: `yasin-fix-upload`)

2. Make your changes in code

3. Open **Source Control** (left sidebar)

4. Review the changed files (click to see the diff)

5. Stage files with **+** (or **Stage All**)

6. Write a commit message and click **Commit**

7. Click **Sync Changes** / **Push** to send your branch to GitHub

8. Open a **Pull Request** on GitHub (or using the VS Code PR extension)

This process is explained in detail (with screenshots placeholders and fallbacks) in:
- [GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md)

---

## 8) Asking for help (recommended)

If you get stuck, copy:
- The command you ran
- The full error message

Then ask a colleague (or GitHub Copilot / ChatGPT) for help.

Important: avoid pasting sensitive patient data into AI tools.
