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

## 6) Running tests before you change something

Tests verify that core functions still work correctly after your edits.
Think of tests as a **checklist** that runs automatically and tells you
"everything still works" or "something broke here".

### Run tests manually (recommended after every change)

```bash
# From the project root:
pytest
```

This runs all 29+ unit tests and takes about 2 seconds.
You will see green PASSED / red FAILED next to each test.

### What happens automatically

| Trigger | What runs | How |
|---------|-----------|-----|
| `git push` to `main` or `develop` | Full test suite | GitHub Actions (`.github/workflows/tests.yml`) |
| Pull Request to `main` or `develop` | Full test suite | GitHub Actions |
| `git push` (local, if pre-commit installed) | Full test suite | pre-commit hook |

**GitHub Actions**: Every push and PR automatically runs the tests on GitHub.
If tests fail, the PR will show a red ✗. You can see the details
in the "Actions" tab on GitHub.

**Pre-commit hook** (optional, local): Runs tests before every `git push`
so broken code never reaches GitHub. To set up:

```bash
pip install pre-commit
pre-commit install --hook-type pre-push
```

After this, `git push` will automatically run the tests first.
If any test fails, the push is blocked until you fix it.

### Writing new tests

Tests live in `tests/test_core_functions.py`. To add a test:

```python
def test_my_new_function():
    """Describe what you are testing."""
    result = my_function(input_data)
    assert result == expected, "Helpful message if it fails"
```

Guidelines:
- Test name must start with `test_`
- Keep each test short and focused on one thing
- Use small, synthetic data (not real datasets)

---

## 7) Running after you change something

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

## 8) Share your changes on GitHub (integrate your code)

If you want your changes to become part of the shared project, the safe way is:
**Branch → Commit → Push → Pull Request → Review → Merge**.

### Why we do it this way (with lab analogies)

If you skip these steps, it’s easy to accidentally break the project for everyone.

- `main` is the **official lab protocol** (the “approved” version).
- A **branch** is a **separate bench / draft copy** of the protocol.
  - You can try changes without touching the official protocol.
  - Everyone can work in parallel without overwriting each other.
- A **commit** is a **lab notebook entry**: “I changed X because Y”.
  - Small, clear commits make it easier to understand and undo mistakes.
- **Push** is **uploading your bench work to the shared freezer (GitHub)**.
  - It also prevents losing work if your laptop dies.
- A **Pull Request (PR)** is a **formal request**: “Please review my proposed protocol update.”
  - It creates a discussion thread, shows the diffs, and allows approvals.
- **Review** is the **buddy-check** step: another person (or you, later) verifies it makes sense.
- **Merge** is when your update becomes the **new official protocol** on `main`.

### VS Code way (recommended, step-by-step)

Important: do **not** work directly on `main`.

#### Step A — Create a branch (your private bench)

1. Look at the bottom-left of VS Code: it shows your current branch (often `main`)
2. Click the branch name
3. Choose **Create new branch…**
4. Name it like:
   - `yourname-short-task` (example: `yasin-fix-upload`)
   - or `feature-short-task` (example: `feature-new-plot`)

Rule of thumb: one branch = one idea.

#### Step B — Make and check your changes

1. Edit files
2. Run the UI or workflow to ensure it still works (see section 6)

#### Step C — Commit (write a notebook entry)

1. Open **Source Control** (left sidebar)
2. Click files to review the diff
3. Stage the files that belong to this change:
   - Click **+** next to a file to stage it
   - Or use **Stage All** if everything is part of the same change
4. Write a short commit message in the message box
   - Good: `Fix upload validation in UI`
   - Avoid: `update` or `changes`
5. Click **Commit**

If VS Code asks you to configure your name/email, follow the prompt.

#### Step D — Push (upload your branch to GitHub)

1. In **Source Control**, click **Sync Changes** or **Push**
2. Sign in to GitHub if prompted

After this, your work is safely on GitHub.

#### Step E — Pull Request (ask for review)

1. Open GitHub in your browser
2. You will often see a banner suggesting: **Compare & pull request**
3. Create the PR
4. In the PR description, write:
   - What you changed
   - Why you changed it
   - How someone can test it

#### Step F — After merge (get the official version back)

Once your PR is merged by a maintainer:
1. Switch back to `main` (click branch name bottom-left)
2. Pull the latest `main` (Source Control → **…** → Pull)

Optional cleanup: you can delete your old branch after merge.

### Full GitHub guide

This process is explained in more detail (with screenshot placeholders and terminal fallbacks) in:
- [GITHUB_WORKFLOW.md](GITHUB_WORKFLOW.md)

---

## 9) Asking for help (recommended)

If you get stuck, copy:
- The command you ran
- The full error message

Then ask a colleague (or GitHub Copilot / ChatGPT) for help.

Important: avoid pasting sensitive patient data into AI tools.
